from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import click
import polars as pl
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from instanovo.inference.beam_search import BeamSearchDecoder
from instanovo.inference.knapsack import Knapsack
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo

# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import yaml
# from sklearn.metrics import auc
# from sklearn.metrics import roc_curve
# from instanovo.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
# from instanovo.utils.metrics import Metrics
# from instanovo.utils.residues import ResidueSet

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# flake8: noqa: CR001
def get_preds(
    data_path: str,
    model: InstaNovo,
    config: dict[str, Any],
    denovo: bool = False,
    output_path: str | None = None,
    knapsack_path: str | None = None,
    save_beams: bool = False,
    n_beams: int = 5,
    filter_precursor_ppm: float = 20.0,
    model_confidence_no_pred: float = 0.0001,
    fp16: bool = False,
    fdr: float = 0.05,
    use_basic_logging: bool = True,
    device: str = "cuda",
) -> None:
    """Get predictions from a trained model."""
    if denovo and output_path is None:
        raise ValueError(
            "Must specify an output path in denovo mode. Specify an output csv file with --output_path"
        )

    if Path(data_path).suffix.lower() != ".ipc":
        raise ValueError(
            f"Unknown filetype of {data_path}. Only Polars .ipc is currently supported."
        )

    logging.info(f"Loading data from {data_path}")
    df = pl.read_ipc(data_path)
    col_map = {
        "Modified sequence": "modified_sequence",
        "Precursor m/z": "precursor_mz",
        "MS/MS m/z": "precursor_mz",
        "m/z": "precursor_mz",
        "Mass": "precursor_mass",
        "Precursor charge": "precursor_charge",
        "Charge": "precursor_charge",
        "Mass values": "mz_array",
        "Mass spectrum": "mz_array",
        "Intensity": "intensity_array",
        "Raw intensity spectrum": "intensity_array",
        "Result file index": "file_index",
        "Scan number": "scan_number",
    }
    # Manually rename Intensity column, so it doesn't overlap with the Raw intensity spectrum column
    # TODO: Needs to be removed in the future
    # df = df.rename({"Intensity": "precursor_intensity"})

    # if "m/z" in df.columns or "MS/MS m/z" in df.columns:
    #     if "MS/MS m/z" in df.columns:
    #         col_map["m/z"] = "calc_precursor_mz"
    #     df = df.rename({k: v for k, v in col_map.items() if k in df.columns})
    #     df = df.with_columns(
    #         pl.col("modified_sequence").map_elements(lambda x: x[1:-1], return_dtype=pl.Utf8)
    #     )
    df = df.rename({k: v for k, v in col_map.items() if k in df.columns})  # Rename in any case

    df = df.sample(fraction=config["subset"], seed=0)
    logging.info(
        f"Data loaded, evaluating {config['subset']*100:.1f}%, {df.shape[0]:,} samples in total."
    )

    if not denovo and (df["modified_sequence"] == "").all():
        raise ValueError(
            "The modified_sequence column is empty, are you trying to run de novo prediction? Add the --denovo flag"
        )

    residue_set = model.residue_set
    logging.info(f"Vocab: {residue_set.index_to_residue}")

    # TODO: find a better place for this, maybe config?
    residue_set.update_remapping(
        {
            # "M(+15.99)": "M(ox)",
            "M(ox)": "M(+15.99)",
            "C(+57.02)": "C",
            # "C": "C(+57.02)",
            "S(p)": "S(+79.97)",
            "T(p)": "T(+79.97)",
            "Y(p)": "Y(+79.97)",
            "Q(+0.98)": "Q(+.98)",
            "N(+0.98)": "N(+.98)",
        }
    )

    #   # Terminal modifications
    #   "(+42.01)": 42.010565 # Acetylation
    #   "(+43.01)": 43.005814 # Carbamylation
    #   "(-17.03)": -17.026549 # NH3 loss
    #   "(+25.98)": 25.980265 # Carbamylation & NH3 loss

    logging.info(f"Residue set: {residue_set.index_to_residue}")

    if not denovo:
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        df = df.with_columns(
            pl.col("modified_sequence")
            .map_elements(
                lambda x: all([y in supported_residues for y in residue_set.tokenize(x)]),
                return_dtype=pl.Boolean,
            )
            .alias("supported")
        )
        if (~df["supported"]).sum() > 0:
            logger.warning(
                "Unsupported residues found in evaluation set! These rows will be dropped."
            )
            df_residues = set()
            for x in df["modified_sequence"]:
                df_residues.update(set(residue_set.tokenize(x)))
            logger.warning(f"Residues found: \n{df_residues-supported_residues}")
            logger.warning(f"Residues supported: \n{supported_residues}")
            logger.warning(
                f"Please check residue remapping if a different convention has been used."
            )
            original_size = df.shape[0]
            df = df.filter(pl.col("supported"))
            logger.warning(f"{original_size-df.shape[0]:,d} rows have been dropped.")
            logger.warning(f"Peptide recall should be manually updated accordingly.")

    ds = SpectrumDataset(df, residue_set, config["n_peaks"], return_str=True, annotated=not denovo)

    dl = DataLoader(
        ds,
        batch_size=config["predict_batch_size"],
        num_workers=config["n_workers"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = model.to(device)
    model = model.eval()

    # Setup decoder
    # TODO: Add flag to choose decoding type (greedy, beam, knapsack beam)
    # if knapsack_path is None or not os.path.exists(knapsack_path):
    #     logging.info("Knapsack path missing or not specified, generating...")
    #     knapsack = _setup_knapsack(model)
    #     decoder = KnapsackBeamSearchDecoder(model, knapsack)
    #     if knapsack_path is not None:
    #         logging.info(f"Saving knapsack to {knapsack_path}")
    #         knapsack.save(knapsack_path)
    # else:
    #     logging.info("Knapsack path found. Loading...")
    #     decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_path)
    decoder = BeamSearchDecoder(model=model)

    index_cols = [
        "id",
        "experiment_name",
        "evidence_index",
        "scan_number",
        "global_index",
        "spectrum_index",
        "file_index",
        "sample",
        "file",
        "index",
        "fileno",
        "precursor_mz",
        "precursor_charge",
    ]
    cols = [x for x in df.columns if x in index_cols]

    pred_df = df.to_pandas()[cols].copy()

    preds: dict[int, list[str]] = {i: [] for i in range(n_beams)}
    targs: list[str] = []
    probs: dict[int, list[float]] = {i: [] for i in range(n_beams)}

    start = time.time()

    iter_dl = enumerate(dl)
    if not use_basic_logging:
        iter_dl = tqdm(enumerate(dl), total=len(dl))

    logging.info("Starting evaluation...")
    logging.info(f"Using n_beams={n_beams}, save_beams={save_beams}, fp16={fp16}")

    for i, batch in iter_dl:
        spectra, precursors, spectra_mask, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)
        spectra_mask = spectra_mask.to(device)
        # Log the first batch
        if i == 0:
            logging.info(f"First batch: {spectra.shape}, {precursors.shape}, {spectra_mask.shape}")
            logging.info(f"First batch spectra: {spectra[0][:10]}")
            logging.info(f"First batch precursors: {precursors[0]}")
            logging.info(f"First batch spectra_mask: {spectra_mask[0][:10]}")
            logging.info(f"First batch peptides: {peptides[0]}")

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=fp16):
            p = decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=n_beams,
                max_length=config["max_length"],
                return_all_beams=save_beams,
            )

        if save_beams:
            for x in p:
                for j in range(n_beams):
                    if j >= len(x):
                        preds[j].append("")
                        probs[j].append(-1e6)
                    else:
                        preds[j].append("".join(x[j].sequence))
                        probs[j].append(x[j].log_probability)
        else:
            preds[0] += [x.sequence if type(x) != list else "" for x in p]
            probs[0] += [x.log_probability if type(x) != list else -1e6 for x in p]
            targs += list(peptides)

        if use_basic_logging and (i + 1) % 10 == 0:
            delta = time.time() - start
            est_total = delta / (i + 1) * (len(dl) - i - 1)
            logging.info(
                f"Batch {i+1:05d}/{len(dl):05d}, [{_format_time(delta)}/{_format_time(est_total)}, {(delta / i):.3f}s/it]"
            )

    delta = time.time() - start

    logging.info(f"Time taken for {data_path} is {delta:.1f} seconds")
    logging.info(
        f"Average time per batch (bs={config['predict_batch_size']}): {delta/len(dl):.1f} seconds"
    )

    if not denovo:
        pred_df["targets"] = targs
        # Map targets to the correct residue set
        # (p) -> (+79.97), (ox) -> (+15.99), etc.
        pred_df["targets"] = [
            re.split(r"(?<=.)(?=[A-Z])", x) if isinstance(x, str) else x
            for x in pred_df["targets"].tolist()
        ]
        pred_df["targets"] = pred_df["targets"].map(
            lambda x: "".join([residue_set.residue_remapping.get(y, y) for y in x])
        )

    pred_df["preds"] = ["".join(x) for x in preds[0]]
    pred_df["preds_tokenised"] = [", ".join(x) for x in preds[0]]
    pred_df["log_probs"] = probs[0]

    # Save predictions before metrics
    if output_path is not None:
        pred_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
    logging.info("Predictions saved. Exiting...")

    # if save_beams:
    #     for i in range(n_beams):
    #         pred_df[f"preds_beam_{i}"] = ["".join(x) for x in preds[i]]
    #         pred_df[f"log_probs_beam_{i}"] = probs[i]

    # try:
    #     ptms = config["ptms"]
    # except KeyError:
    #     ptms = {"(p)": "(+79.97)", "(ox)": "(+15.99)"}  # Hardcoded PTMs
    #     logging.info("No PTMs found in config, using default PTMs")
    # logging.info(f"ptms: {ptms}")
    # logging.info(f"first 10 preds: {preds[0][:10]}")
    # logging.info(f"first 10 targets: {pred_df['targets'][:10]}")

    # # Calculate metrics
    # if not denovo:
    #     # Hardcode ptms:

    #     metrics = Metrics(residue_set, config["isotope_error_range"], ptms=ptms)

    #     # Make sure we pass preds[0] without joining on ""
    #     # This is to handle cases where n-terminus modifications could be accidentally joined
    #     (
    #         aa_prec,
    #         aa_recall,
    #         pep_recall,
    #         pep_prec,
    #         ptm_prec,
    #         ptm_recall,
    #         pred_bool,
    #     ) = metrics.compute_precision_recall_ptm(pred_df["targets"], preds[0])
    #     aa_er = metrics.compute_aa_er(pred_df["targets"], preds[0])
    #     auc = metrics.calc_auc(pred_df["targets"], preds[0], np.exp(pred_df["log_probs"]))
    #     pred_df[
    #         "pred_bool"
    #     ] = pred_bool  # Save for later use with ROC curve and precision-recall curve

    #     logging.info(f"ptm_prec: {ptm_prec}")
    #     logging.info(f"ptm_recall: {ptm_recall}")

    #     logging.info(f"Performance on {data_path}:")
    #     logging.info(f"  aa_er       {aa_er:.5f}")
    #     logging.info(f"  aa_prec     {aa_prec:.5f}")
    #     logging.info(f"  aa_recall   {aa_recall:.5f}")
    #     logging.info(f"  pep_prec    {pep_prec:.5f}")
    #     logging.info(f"  pep_recall  {pep_recall:.5f}")
    #     for ptm in ptms.keys():
    #         logging.info(f"  {ptm}_prec    {ptm_prec[ptm]:.5f}")
    #         logging.info(f"  {ptm}_recall  {ptm_recall[ptm]:.5f}")
    #     logging.info(f"  auc         {auc:.5f}")

    #     _, threshold = metrics.find_recall_at_fdr(
    #         pred_df["targets"], preds[0], np.exp(pred_df["log_probs"]), fdr=fdr
    #     )
    #     (
    #         aa_prec,
    #         aa_recall,
    #         pep_recall,
    #         pep_prec,
    #         ptm_prec,
    #         ptm_recall,
    #         pred_bool,
    #     ) = metrics.compute_precision_recall_ptm(
    #         pred_df["targets"], preds[0], np.exp(pred_df["log_probs"]), threshold=threshold
    #     )
    #     logging.info(f"Performance at {fdr*100:.1f}% FDR:")
    #     logging.info(f"  aa_prec     {aa_prec:.5f}")
    #     logging.info(f"  aa_recall   {aa_recall:.5f}")
    #     logging.info(f"  pep_prec    {pep_prec:.5f}")
    #     logging.info(f"  pep_recall  {pep_recall:.5f}")
    #     logging.info(f"  confidence  {threshold:.5f}")
    #     for ptm in ptms.keys():
    #         logging.info(f"  {ptm}_prec    {ptm_prec[ptm]:.5f}")
    #         logging.info(f"  {ptm}_recall  {ptm_recall[ptm]:.5f}")

    #     # Calculate some additional information for filtering:
    #     # pred_df["delta_mass_ppm"] = pred_df.apply(
    #     #     lambda row: np.min(
    #     #         np.abs(
    #     #             metrics.matches_precursor(
    #     #                 preds[0][row.name], row["precursor_mz"], row["precursor_charge"]
    #     #             )[1]
    #     #         )
    #     #     ),
    #     #     axis=1,
    #     # )
    #     pred_df["delta_mass_ppm"] = np.nan

    #     for index, row in pred_df.iterrows():
    #         try:
    #             pred_value = preds[0][index]
    #             if pred_value == "":
    #                 pred_value = []
    #             precursor_mz = row["precursor_mz"]
    #             precursor_charge = row["precursor_charge"]

    #             delta_mass = np.min(
    #                 np.abs(metrics.matches_precursor(pred_value, precursor_mz, precursor_charge)[1])
    #             )
    #             pred_df.at[index, "delta_mass_ppm"] = delta_mass

    #         except Exception as e:
    #             logging.info(f"Error processing row {index}: {row.to_dict()}")
    #             logging.info(f"Exception: {e}")

    #     idx = pred_df["delta_mass_ppm"] < filter_precursor_ppm
    #     filtered_preds = pd.Series(preds[0])
    #     filtered_preds[~idx] = ""
    #     (
    #         aa_prec,
    #         aa_recall,
    #         pep_recall,
    #         pep_prec,
    #         ptm_prec,
    #         ptm_recall,
    #         pred_bool,
    #     ) = metrics.compute_precision_recall_ptm(pred_df["targets"], filtered_preds)
    #     logging.info(f"Performance with filtering at {filter_precursor_ppm} ppm delta mass:")
    #     logging.info(f"  aa_prec     {aa_prec:.5f}")
    #     logging.info(f"  aa_recall   {aa_recall:.5f}")
    #     logging.info(f"  pep_prec    {pep_prec:.5f}")
    #     logging.info(f"  pep_recall  {pep_recall:.5f}")
    #     for ptm in ptms.keys():
    #         logging.info(f"  {ptm}_prec    {ptm_prec[ptm]:.5f}")
    #         logging.info(f"  {ptm}_recall  {ptm_recall[ptm]:.5f}")
    #     logging.info(
    #         f"Rows filtered: {df.shape[0]-np.sum(idx)} ({(df.shape[0]-np.sum(idx))/df.shape[0]*100:.2f}%)"
    #     )

    #     idx = np.exp(pred_df["log_probs"]) > model_confidence_no_pred
    #     filtered_preds = pd.Series(preds[0])
    #     filtered_preds[~idx] = ""
    #     (
    #         aa_prec,
    #         aa_recall,
    #         pep_recall,
    #         pep_prec,
    #         ptm_prec,
    #         ptm_recall,
    #         pred_bool,
    #     ) = metrics.compute_precision_recall_ptm(pred_df["targets"], filtered_preds)
    #     logging.info(f"Performance with filtering confidence < {model_confidence_no_pred}")
    #     logging.info(f"  aa_prec     {aa_prec:.5f}")
    #     logging.info(f"  aa_recall   {aa_recall:.5f}")
    #     logging.info(f"  pep_prec    {pep_prec:.5f}")
    #     logging.info(f"  pep_recall  {pep_recall:.5f}")
    #     for ptm in ptms.keys():
    #         logging.info(f"  {ptm}_prec    {ptm_prec[ptm]:.5f}")
    #         logging.info(f"  {ptm}_recall  {ptm_recall[ptm]:.5f}")
    #     logging.info(
    #         f"Rows filtered: {df.shape[0]-np.sum(idx)} ({(df.shape[0]-np.sum(idx))/df.shape[0]*100:.2f}%)"
    #     )

    # # Save output
    # if output_path is not None:
    #     pred_df.to_csv(output_path, index=False)
    #     logging.info(f"Predictions saved to {output_path}")

    #     # Upload to Aichor
    #     if s3._s3_enabled():
    #         s3.upload(output_path, s3.convert_to_s3_output(output_path))


@click.command()
@click.argument("data-path")
@click.argument("model-path")
@click.option("--output-path", "-o", default=None)
@click.option("--denovo", "-n", is_flag=True, default=False)
@click.option("--subset", "-s", default=1.0)
@click.option("--knapsack-path", "-k", default=None)
@click.option("--n-workers", "-w", default=16)
@click.option("--batch-size", "-b", default=128)
@click.option("--save-beams", "-a", is_flag=True, default=False)
@click.option("--n-beams", "-m", default=5)
@click.option("--filter-precursor-ppm", "-f", default=20)
@click.option("--fp16", "-d", is_flag=True, default=False)
def main(
    data_path: str,
    model_path: str,
    output_path: str,
    denovo: bool,
    subset: float,
    knapsack_path: str,
    n_workers: int,
    batch_size: int,
    save_beams: bool,
    n_beams: int,
    filter_precursor_ppm: float,
    fp16: bool,
) -> None:
    """Predict with the model."""
    logging.info("Initializing inference.")

    logging.info(f"Loading model from {model_path}")
    model, config = InstaNovo.load(model_path)
    logging.info(f"Config:\n{config}")

    try:
        with open_dict(config):
            config["n_workers"] = int(n_workers)
            config["subset"] = float(subset)
            config["predict_batch_size"] = int(batch_size)
    except:
        logging.error("Error loading config with OmegaConf. Assuming normal dict.")
        assert isinstance(config, dict)
        config["n_workers"] = int(n_workers)
        config["subset"] = float(subset)
        config["predict_batch_size"] = int(batch_size)

    get_preds(
        data_path,
        model,
        config,
        denovo,
        output_path,
        knapsack_path,
        save_beams,
        n_beams,
        filter_precursor_ppm,
        fp16=fp16,
    )


def _setup_knapsack(model: InstaNovo) -> Knapsack:
    MASS_SCALE = 10000
    residue_masses = dict(model.residue_set.residue_masses.copy())
    for special_residue in list(model.residue_set.residue_to_index.keys())[:3]:
        residue_masses[special_residue] = 0
    residue_indices = model.residue_set.residue_to_index
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=4000.00,
        mass_scale=MASS_SCALE,
    )


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"


if __name__ == "__main__":
    main()
