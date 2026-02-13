from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional

import finetuning_scheduler as fts
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
import yaml
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch import Tensor
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from instanovo.inference.beam_search import BeamSearchDecoder
from instanovo.transformer.dataset import collate_batch
from instanovo.transformer.dataset import SpectrumDataset
from instanovo.transformer.model import InstaNovo
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from instanovo.utils.schedulers import STLRScheduler
from instanovo.utils.schedulers import STLRSchedulerEpochBased

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PTModule(ptl.LightningModule):
    """PTL wrapper for model."""

    def __init__(
        self,
        config: dict[str, Any],
        model: InstaNovo,
        decoder: BeamSearchDecoder,
        metrics: Metrics,
        sw: SummaryWriter,
        checkpoint: dict[str, Any],
        # optim: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler._LRScheduler,
        # device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.decoder = decoder
        self.metrics = metrics
        self.sw = sw
        self.checkpoint = checkpoint
        # self.optim = optim
        # self.scheduler = scheduler

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        self.running_loss = None
        self._reset_valid_metrics()
        self.steps = 0
        self.validation_numerator = 0

        # Update rates based on bs=32
        self.step_scale = 32 / config["train_batch_size"]

    def forward(
        self,
        spectra: Tensor,
        precursors: Tensor,
        peptides: list[str] | Tensor,
        spectra_mask: Tensor,
        peptides_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Model forward pass."""
        return self.model(spectra, precursors, peptides, spectra_mask, peptides_mask)  # type: ignore

    def training_step(  # need to update this
        self,
        batch: tuple[Tensor, Tensor, Tensor, list[str] | Tensor, Tensor],
    ) -> torch.Tensor:
        """A single training step.

        Args:
            batch (tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]) :
                A batch of MS/MS spectra, precursor information, and peptide
                sequences as torch Tensors.

        Returns:
            torch.FloatTensor: training loss
        """
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        # Adding dropout to the spectra_mask
        spectra_mask = nn.functional.dropout(
            spectra_mask.float(), p=self.config["peak_dropout"]
        ).bool()
        peptides_mask = peptides_mask.to(self.device)

        if isinstance(peptides, Tensor):
            peptides = peptides.to(self.device)

            preds, truth = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
            # cut off EOS's prediction, ignore_index should take care of masking
            preds = preds[0][:, :-1].reshape(-1, preds[0].shape[-1])
            loss = self.loss_fn(preds, peptides.flatten())
        else:
            preds, truth = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
            preds = preds[:, :-1, :].reshape(-1, self.model.decoder.vocab_size + 1)
            loss = self.loss_fn(preds, truth.flatten())

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        if (
            (self.steps + 1) % int(self.config.get("console_logging_steps", 2000) * self.step_scale)
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            logging.info(
                f"[Step {self.steps+1:06d}]: train_loss_raw={loss.item():.4f}, running_loss={self.running_loss:.4f}, LR={lr}"
            )

        if (self.steps + 1) % int(
            self.config.get("tensorboard_logging_steps", 500) * self.step_scale
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.sw.add_scalar("train/loss_raw", loss.item(), self.steps + 1)
            self.sw.add_scalar("train/loss_smooth", self.running_loss, self.steps + 1)
            self.sw.add_scalar(
                "optim/lr", lr, self.steps + 1
            )  # Might be wrong because of the scheduler
            self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps + 1)
            # logging.info(f"Logged to Tensorboard")

        # Logging LR over global step
        self.sw.add_scalar(
            "global/lr",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            self.steps + 1,
        )

        self.steps += 1

        # Logging for use with finetuning_scheduler
        # On step or on epoch? (or both)
        # self.log_dict({"train_loss": loss.item(),
        #                "lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
        #                },
        #                     on_step=False,
        #                     on_epoch=True)
        self.log("train_loss", loss.item())
        self.log(
            "lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], prog_bar=False
        )

        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, list[str] | Tensor, Tensor], *args: Any
    ) -> torch.Tensor:
        """Single validation step."""
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        # peptides = peptides.to(self.device)
        # peptides_mask = peptides_mask.to(self.device)

        # Loss
        # with torch.no_grad():
        #     preds, truth = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        # preds = preds[:, :-1, :].reshape(-1, self.model.decoder.vocab_size + 1)
        # loss = self.loss_fn(preds, truth.flatten())
        if isinstance(peptides, Tensor):
            peptides = peptides.to(self.device)

            with torch.no_grad():
                preds, truth = self.forward(
                    spectra, precursors, peptides, spectra_mask, peptides_mask
                )
            # cut off EOS's prediction, ignore_index should take care of masking
            preds = preds[0][:, :-1].reshape(-1, preds[0].shape[-1])
            loss = self.loss_fn(preds, peptides.flatten())
        else:
            with torch.no_grad():
                preds, truth = self.forward(
                    spectra, precursors, peptides, spectra_mask, peptides_mask
                )
            preds = preds[:, :-1, :].reshape(-1, self.model.decoder.vocab_size + 1)
            loss = self.loss_fn(preds, truth.flatten())

        # Greedy decoding
        with torch.no_grad():
            # y, _ = decoder(spectra, precursors, spectra_mask)
            p = self.decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=self.config["n_beams"],
                max_length=self.config["max_length"],
            )

        y = ["".join(x.sequence) if type(x) != list else "" for x in p]
        if isinstance(peptides, Tensor):
            # When peptides is a Tensor, it is reversed. We must reverse it again.
            targets = ["".join(s) for s in self.model.batch_idx_to_aa(peptides, reverse=True)]
        else:
            targets = peptides

        (
            aa_prec,
            aa_recall,
            pep_prec,
            pep_recall,
            ptm_prec,
            ptm_recall,
            pred_bool,
        ) = self.metrics.compute_precision_recall_ptm(targets, y)
        aa_er = self.metrics.compute_aa_er(targets, y)

        self.valid_metrics["valid_loss"].append(loss.item())
        self.valid_metrics["aa_er"].append(aa_er)
        self.valid_metrics["aa_prec"].append(aa_prec)
        self.valid_metrics["aa_recall"].append(aa_recall)
        self.valid_metrics["pep_prec"].append(pep_prec)
        self.valid_metrics["pep_recall"].append(pep_recall)

        # Handle PTM metrics
        # compute_precision_recall_ptm returns a dict like {"(+79.97)": ..., "(+15.99)": ...}
        # Here we revert it to "(p)" and "(ox)" for clarity in the logs
        for ptm in self.config["ptms"].keys():
            self.valid_metrics[f"{ptm}_prec"].append(ptm_prec[ptm])
            self.valid_metrics[f"{ptm}_recall"].append(ptm_recall[ptm])

        return loss

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        epoch = self.trainer.current_epoch
        self.sw.add_scalar(f"eval/train_loss", self.running_loss, epoch)
        # print(f"results_metrics_train: {self.trainer._results.result_metrics}")
        self.running_loss = None

    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics at the end of each epoch."""
        epoch = self.trainer.current_epoch
        val_num = self.validation_numerator  # To keep track of multiple validations per epoch
        logging.info(f"[Epoch {epoch:02d}][Step {self.steps:05d}][Val_num {val_num:02d}] Metrics:")
        for k, v in self.valid_metrics.items():
            logging.info(f"[Epoch {epoch:02d}] - {k:<12s} {np.mean(v):>6.3f}")
            self.sw.add_scalar(f"eval/{k}", np.mean(v), val_num)
            self.log(f"{k}", np.mean(v))

        logging.info(
            f"[Epoch {epoch:02d}] train_loss={self.running_loss if self.running_loss else 0:.5f}, valid_loss={np.mean(self.valid_metrics['valid_loss']):.5f}"
        )

        self.validation_numerator += 1
        self._reset_valid_metrics()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save config with checkpoint."""
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attempt to load config with checkpoint."""
        self.config = checkpoint["config"]

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        # TODO: Add argument to specify optimizer and scheduler as well as their parameters
        # TODO: If LinearLR, initialize with multiplied LR...

        # Initialize optimizer here to filter out non-trainable parameters specified by the finetuning_scheduler
        if self.config["optimizer"] == "AdamW":
            optim = torch.optim.AdamW(
                params=list(filter(lambda x: x.requires_grad, self.model.parameters())),
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"]),
            )
            logging.info("Using AdamW optimizer")
        elif self.config["optimizer"] == "Adam":
            optim = torch.optim.Adam(
                params=list(filter(lambda x: x.requires_grad, self.model.parameters())),
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"]),
            )
            logging.info("Using Adam optimizer")
        else:
            raise ValueError(f"Optimizer {self.optim} not supported")

        if self.config["scheduler"] == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optim,
                T_0=self.config["T_0"],
                T_mult=self.config["T_mult"],
                eta_min=self.config["eta_min"],
                last_epoch=-1,
            )
            logging.info("Using CosineAnnealingWarmRestarts scheduler")

        elif self.config["scheduler"] == "LinearLR":
            # Emulation of WarmupScheduler
            # Max start_factor is 1, so we need to multiply the LR if we want to start higher and decrease to the specified LR
            scheduler = lr_scheduler.LinearLR(
                optim,
                start_factor=self.config["LinearLR_start_factor"],
                end_factor=self.config["LinearLR_end_factor"],
                total_iters=self.config["LinearLR_total_iters"],
            )
            logging.info("Using LinearLR scheduler")

        elif self.config["scheduler"] == "STLRScheduler":
            scheduler = STLRScheduler(
                optim,
                last_epoch=-1,
                ratio=float(self.config["STLR_ratio"]),
                max_lr=float(self.config["learning_rate"]),
                full_cycle_length=int(self.config["STLR_cycle_length"]),
                cycle_midpoint_fraction=float(self.config["STLR_midpoint_fraction"]),
            )
            logging.info("Using STLR Scheduler")

        elif self.config["scheduler"] == "STLRSchedulerEpochBased":
            scheduler = STLRSchedulerEpochBased(
                optim,
                max_lrs_list=[float(val) for val in self.config["STLR_maxlr_list"]],
                last_epoch=-1,
                num_epochs=self.config["epochs"],
                ratio=float(self.config["STLR_ratio"]),
                full_cycle_length=int(self.config["STLR_cycle_length"]),
                cycle_midpoint_fraction=float(self.config["STLR_midpoint_fraction"]),
            )
            logging.info("Using Epoch based STLR Scheduler")
        # TODO: Add ReduceLROnPlateau

        else:
            raise ValueError(f"Scheduler {self.scheduler} not supported")

        # Interval: step or epoch? When set to step, it still says "epoch" in the logs. But step is probably correct.
        return [optim], {"scheduler": scheduler, "interval": self.config["scheduler_interval"]}

    def _reset_valid_metrics(self) -> None:
        valid_metrics = ["valid_loss", "aa_er", "aa_prec", "aa_recall", "pep_prec", "pep_recall"]
        # Add PTM metrics depending on the config
        valid_metrics += [f"{ptm}_prec" for ptm in self.config["ptms"].keys()]
        valid_metrics += [f"{ptm}_recall" for ptm in self.config["ptms"].keys()]
        self.valid_metrics: dict[str, list[float]] = {x: [] for x in valid_metrics}


# flake8: noqa: CR001
def train(
    train_path: str,
    valid_path: str,
    config: dict,
    schedule_path: str,
    checkpoint_path: str | None = None,
) -> None:
    """Training function."""
    config["tb_summarywriter"] = config["tb_summarywriter"] + datetime.datetime.now().strftime(
        "_%y_%m_%d_%H_%M"
    )
    sw = SummaryWriter(config["tb_summarywriter"])
    residue_set = ResidueSet(
        residue_masses=config["residues"],
        residue_remapping={
            "M(ox)": "M(+15.99)",
            "S(p)": "S(+79.97)",
            "T(p)": "T(+79.97)",
            "Y(p)": "Y(+79.97)",
        },
    )
    logging.info(f"Vocab: {residue_set.index_to_residue}")
    logging.info("Loading data")

    train_df = pl.read_ipc(train_path)
    train_df = train_df.sample(fraction=config["train_subset"], seed=0)
    valid_df = pl.read_ipc(valid_path)
    valid_df = valid_df.sample(fraction=config["valid_subset"], seed=0)
    train_ds = SpectrumDataset(train_df, residue_set, config["n_peaks"], return_str=False)
    valid_ds = SpectrumDataset(valid_df, residue_set, config["n_peaks"], return_str=False)

    logging.info(
        f"Data loaded: {len(train_ds):,} training samples; {len(valid_ds):,} validation samples"
    )
    logging.info(f"Train columns: {train_df.columns}")
    logging.info(f"Valid columns: {valid_df.columns}")

    logging.info("Checking if any validation set overlaps with training set...")
    leakage = any(valid_df["modified_sequence"].is_in(train_df["modified_sequence"]))
    if leakage:
        raise ValueError("Portion of validation set sequences overlaps with training set.")
    else:
        logging.info("No data leakage!")

    expected_cols = [
        "mz_array",
        "intensity_array",
        "precursor_mz",
        "precursor_charge",
        "modified_sequence",
    ]
    if any([x not in list(train_df.columns) for x in expected_cols]):
        raise ValueError(f"Column(s) missing from train_df! Expected: {expected_cols}")
    if any([x not in list(valid_df.columns) for x in expected_cols]):
        raise ValueError(f"Column(s) missing from valid_df! Expected: {expected_cols}")

    train_dl = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        num_workers=config["n_workers"],
        shuffle=True,
        collate_fn=collate_batch,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=config["predict_batch_size"],
        num_workers=config["n_workers"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Update rates based on bs=32
    # TODO: Check if this is correct
    step_scale = 32 / config["train_batch_size"]
    logging.info(f"Updates per epoch: {len(train_dl):,}, step_scale={step_scale}")

    batch = next(iter(train_dl))
    spectra, precursors, spectra_mask, peptides, peptides_mask = batch

    logging.info("Sample batch:")
    logging.info(f" - spectra.shape={spectra.shape}")
    logging.info(f" - precursors.shape={precursors.shape}")
    logging.info(f" - spectra_mask.shape={spectra_mask.shape}")
    logging.info(f" - len(peptides)={len(peptides)}")
    logging.info(f" - peptides_mask.shape={peptides_mask.shape}")

    # init model
    model = InstaNovo(
        residue_set=residue_set,
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        max_charge=config["max_charge"],
        apply_lora=config["apply_lora"],
        lora_rank=config["lora_rank"],
    )

    if checkpoint_path is not None:
        logging.info(f"Loading model checkpoint from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logging.info(f"Checkpoint keys: {checkpoint.keys()}")

        # Reformat the loaded model_state keys
        if "state_dict" in checkpoint:
            model_state = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

            # Formatting
            model_state = {k.replace("transformer_decoder.", ""): v for k, v in model_state.items()}
            model_state = {
                k.replace("decoder.charge_encoder.", "charge_encoder."): v
                for k, v in model_state.items()
            }
            # Leave out embedding and head layers when loading from the base InstaNovo checkpoint
            # because the phospho model has an extended vocabulary
            model_state = {
                k: v
                for k, v in model_state.items()
                if k not in ["aa_embed.weight", "head.weight", "head.bias"]
            }

        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys ({len(missing_keys)}): {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")

    # logging.info(f"Model: \n{model}")
    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    # logging.info("Test forward pass:")
    # with torch.no_grad():
    #     y, _ = model(spectra, precursors, peptides, spectra_mask, peptides_mask)
    #     logging.info(f" - y.shape={y.shape}")

    # Train on GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # decoder = GreedyDecoder(model, i2s, max_length=config["max_length"])
    decoder = BeamSearchDecoder(model=model)
    metrics = Metrics(residue_set, config["isotope_error_range"], config["ptms"])
    strategy = _get_strategy()
    ptmodel = PTModule(config, model, decoder, metrics, sw, checkpoint)

    if config["fine_tune"]:
        callbacks = [
            fts.FinetuningScheduler(
                ft_schedule=schedule_path,
                base_max_lr=config["learning_rate"],
                restore_best=config["restore_best"],
                epoch_transitions_only=config["epoch_transitions_only"],
                allow_untested=True,
                apply_lambdas_new_pgs=True,
            ),
            fts.FTSCheckpoint(
                monitor="valid_loss",
                dirpath=config["model_save_folder_path"],
                filename="ft_phospho-epoch:{epoch:02d}-step:{step:05d}-train_loss:{train_loss:.4f}-valid_loss={valid_loss:.4f}",
                auto_insert_metric_name=False,
                save_top_k=-1,
                save_weights_only=config["save_weights_only"],
                save_on_train_epoch_end=False,  # Save on validation epoch end
                verbose=True,
            ),
        ]
        if config["epoch_transitions_only"] == False:
            # Early stopping is needed to determine phase transitions unless epoch_transitions_only is set to True in the FinetuningScheduler
            # In that case, the scheduler will handle the phase transitions and EarlyStopping is ignored
            callbacks.append(
                fts.FTSEarlyStopping(
                    monitor="valid_loss",
                    min_delta=config["min_delta"],
                    patience=config["patience"],
                    verbose=True,
                    mode="min",
                    strict=True,  # Stops training if monitored value not found in logs
                )
            )

        logging.info(f"Fine-tuning enabled. Schedule:")
        with open(schedule_path) as f:
            schedule = yaml.safe_load(f)
            for phase in schedule:
                logging.info(f"Phase {phase}:")
                for key, val in schedule[phase].items():
                    logging.info(f"\t{key}: {val}")

    else:
        logging.info("Model saving disabled")
        callbacks = None

    logging.info("Initializing PL trainer.")
    trainer = ptl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        callbacks=callbacks,
        devices="auto",
        logger=config["logger"],
        max_epochs=config["epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        accumulate_grad_batches=config["grad_accumulation"],
        gradient_clip_val=config["gradient_clip_val"],
        strategy=strategy,
        enable_progress_bar=config["progress_bar"],
        val_check_interval=config["val_check_interval"],
    )

    # Train the model.
    trainer.fit(ptmodel, train_dl, valid_dl)

    logging.info("Training complete.")


def _get_strategy() -> DDPStrategy | None:
    """Get the strategy for the Trainer.

    The DDP strategy works best when multiple GPUs are used. It can work for
    CPU-only, but definitely fails using MPS (the Apple Silicon chip) due to
    Gloo.

    Returns
    -------
    Optional[DDPStrategy]
        The strategy parameter for the Trainer.
    """
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)

    return None


def main() -> None:
    """Train the model."""
    logging.info("Initializing training.")

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path")
    parser.add_argument("--valid_path")
    parser.add_argument("--config", default="base.yaml")
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--n_workers", default=8)
    # parser.add_argument("--schedule")
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"../../configs/instanovo/{args.config}"
    )

    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)

    config["residues"] = {str(aa): float(mass) for aa, mass in config["residues"].items()}
    config["n_gpu"] = int(args.n_gpu)
    config["n_workers"] = int(args.n_workers)
    schedule_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f"../../configs/finetune_scheduler/{config['schedule']}",
    )

    if config["n_gpu"] > 1:
        raise Exception("n_gpu > 1 currently not supported.")

    if not config["train_from_scratch"]:
        checkpoint_path = config["resume_checkpoint"]
    else:
        checkpoint_path = None

    train(args.train_path, args.valid_path, config, schedule_path, checkpoint_path)


if __name__ == "__main__":
    main()
