from __future__ import annotations

import bisect
import re

import jiwer
import numpy as np

from instanovo.constants import CARBON_MASS_DELTA
from instanovo.utils.residues import ResidueSet


class Metrics:
    """Peptide metrics class."""

    def __init__(
        self,
        residue_set: ResidueSet,
        isotope_error_range: list[int],
        ptms: dict[str, str],
        cum_mass_threshold: float = 0.5,
        ind_mass_threshold: float = 0.1,
    ) -> None:
        self.residue_set = residue_set
        self.isotope_error_range = isotope_error_range
        self.ptms = ptms
        self.cum_mass_threshold = cum_mass_threshold
        self.ind_mass_threshold = ind_mass_threshold

    @staticmethod
    def _split_sequences(seq: list[str] | list[list[str]]) -> list[list[str]]:
        return [re.split(r"(?<=.)(?=[A-Z])", x) if isinstance(x, str) else x for x in seq]

    @staticmethod
    def _split_peptide(peptide: str | list[str]) -> list[str]:
        if not isinstance(peptide, str):
            return peptide
        return re.split(r"(?<=.)(?=[A-Z])", peptide)

    def matches_precursor(
        self, seq: str | list[str], prec_mass: float, prec_charge: int, prec_tol: int = 50
    ) -> tuple[bool, list[float]]:
        """Check if a sequence matches the precursor mass within some tolerance."""
        seq_mass = self._mass(seq, charge=prec_charge)
        delta_mass_ppm = [
            self._calc_mass_error(seq_mass, prec_mass, prec_charge, isotope)
            for isotope in range(
                self.isotope_error_range[0],
                self.isotope_error_range[1] + 1,
            )
        ]
        return any(abs(d) < prec_tol for d in delta_mass_ppm), delta_mass_ppm

    def compute_aa_er(
        self,
        peptides_truth: list[str] | list[list[str]],
        peptides_predicted: list[str] | list[list[str]],
    ) -> float:
        """Compute amino-acid level error-rate."""
        # Ensure amino acids are separated
        peptides_truth = self._split_sequences(peptides_truth)
        peptides_predicted = self._split_sequences(peptides_predicted)

        return float(
            jiwer.wer(
                [" ".join(x) for x in peptides_truth], [" ".join(x) for x in peptides_predicted]
            )
        )

    # Adapted from https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/evaluate.py
    def compute_precision_recall(
        self,
        targets: list[str] | list[list[str]],
        predictions: list[str] | list[list[str]],
        confidence: list[float] | None = None,
        threshold: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Calculate precision and recall at peptide- and AA-level.

        Args:
            targets (list[str] | list[list[str]]): Target peptides.
            predictions (list[str] | list[list[str]]): Model predicted peptides.
            confidence (list[float] | None): Optional model confidence.
            threshold (float | None): Optional confidence threshold.
        """
        targets = self._split_sequences(targets)
        predictions = self._split_sequences(predictions)

        n_targ_aa, n_pred_aa, n_match_aa = 0, 0, 0
        n_pred_pep, n_match_pep = 0, 0

        if confidence is None or threshold is None:
            threshold = 0
            confidence = np.ones(len(predictions))

        for i in range(len(targets)):
            targ = self._split_peptide(targets[i])
            pred = self._split_peptide(predictions[i])
            conf = confidence[i]  # type: ignore

            if pred[0] == "":
                pred = []

            n_targ_aa += len(targ)
            if conf >= threshold and len(pred) > 0:
                n_pred_aa += len(pred)
                n_pred_pep += 1

                # pred = [x.replace('I', 'L') for x in pred]
                # n_match_aa += np.sum([m[0]==' ' for m in difflib.ndiff(targ,pred)])
                n_match = self._novor_match(targ, pred)
                n_match_aa += n_match

                if len(pred) == len(targ) and len(targ) == n_match:
                    n_match_pep += 1

        pep_recall = n_match_pep / len(targets)
        aa_recall = n_match_aa / n_targ_aa

        if n_pred_pep == 0:
            pep_precision = 1.0
            aa_prec = 1.0
        else:
            pep_precision = n_match_pep / n_pred_pep
            aa_prec = n_match_aa / n_pred_aa

        return aa_prec, aa_recall, pep_recall, pep_precision

    def compute_precision_recall_ptm(
        self,
        targets: list[str] | list[list[str]],
        predictions: list[str] | list[list[str]],
        confidence: list[float] | None = None,
        threshold: float | None = None,
    ) -> tuple[float, float, float, float, dict, dict, np.ndarray]:
        """Calculate precision and recall at peptide-, AA- and PTM-level.

        Args:
            targets (list[str] | list[list[str]]): Target peptides.
            predictions (list[str] | list[list[str]]): Model predicted peptides.
            confidence (list[float] | None): Optional model confidence.
            threshold (float | None): Optional confidence threshold.
        """
        # print(predictions[0])
        # print(targets[0])
        targets = self._split_sequences(targets)
        predictions = self._split_sequences(predictions)

        # print(predictions[0])
        # print(targets[0])

        n_targ_aa, n_pred_aa, n_match_aa = 0, 0, 0
        n_pred_pep, n_match_pep = 0, 0
        n_targ_ptm = {ptm: 0 for ptm in self.ptms}
        n_pred_ptm = {ptm: 0 for ptm in self.ptms}
        n_match_ptm = {ptm: 0 for ptm in self.ptms}

        pred_bool = np.zeros(len(predictions))

        if confidence is None or threshold is None:
            threshold = 0
            confidence = np.ones(len(predictions))

        for i in range(len(targets)):
            targ = self._split_peptide(targets[i])
            pred = self._split_peptide(predictions[i])
            # print(pred)
            # print(type(pred))
            conf = confidence[i]  # type: ignore

            if isinstance(pred, float) or pred[0] == "":
                pred = []
            # if pred == float("nan"):
            #     pred = []
            # elif not isinstance(pred, float) and pred[0] == "":
            #     pred = []
            # print(pred)

            n_targ_aa += len(targ)
            for ptm, ptm_weight in self.ptms.items():
                n_targ_ptm[ptm] += sum([ptm_weight in x for x in targ])
            if conf >= threshold and len(pred) > 0:
                n_pred_aa += len(pred)
                for ptm, ptm_weight in self.ptms.items():
                    n_pred_ptm[ptm] += sum([ptm_weight in x for x in pred])
                n_pred_pep += 1

                n_match, n_match_ptm_instance = self._novor_match_ptm(targ, pred)
                n_match_aa += n_match
                for ptm in self.ptms:
                    n_match_ptm[ptm] += n_match_ptm_instance[ptm]

                if len(pred) == len(targ) and len(targ) == n_match:
                    n_match_pep += 1
                    pred_bool[i] = 1

        pep_recall = n_match_pep / len(targets) if targets else 0
        aa_recall = n_match_aa / n_targ_aa if n_targ_aa else 0
        ptm_recall = {
            ptm: n_match_ptm[ptm] / n_targ_ptm[ptm] if n_targ_ptm[ptm] else 0 for ptm in self.ptms
        }

        pep_prec = n_match_pep / n_pred_pep if n_pred_pep else 1.0
        aa_prec = n_match_aa / n_pred_aa if n_pred_aa else 1.0
        ptm_prec = {
            ptm: n_match_ptm[ptm] / n_pred_ptm[ptm] if n_pred_ptm[ptm] else 1.0 for ptm in self.ptms
        }

        return aa_prec, aa_recall, pep_prec, pep_recall, ptm_prec, ptm_recall, pred_bool

    def calc_auc(
        self,
        targs: list[str] | list[list[str]],
        preds: list[str] | list[list[str]],
        conf: list[float],
    ) -> float:
        """Calculate the peptide-level AUC."""
        x, y = self._get_pr_curve(targs, preds, conf)
        recall, precision = np.array(x)[::-1], np.array(y)[::-1]

        width = recall[1:] - recall[:-1]
        height = np.minimum(precision[1:], precision[:-1])
        top = np.maximum(precision[1:], precision[:-1])
        side = top - height
        return (width * height).sum() + 0.5 * (side * width).sum()  # type: ignore

    def find_recall_at_fdr(
        self,
        targs: list[str] | list[list[str]],
        preds: list[str] | list[list[str]],
        conf: list[float],
        fdr: float = 0.05,
    ) -> tuple[float, float]:
        """Get model recall and threshold for specified FDR."""
        conf = np.array(conf)
        order = conf.argsort()[::-1]
        matches = np.array(self._get_peptide_matches(targs, preds))
        matches = matches[order]
        conf = conf[order]

        csum = np.cumsum(matches)
        precision = csum / (np.arange(len(matches)) + 1)
        recall = csum / len(matches)

        # if precision never greater than FDR
        if all(precision < (1 - fdr)):
            # recall = 0, threshold = 1
            return 0.0, 1.0

        # bisect requires ascending order
        idx = len(precision) - bisect.bisect_right(precision[::-1], 1 - fdr) - 1
        return recall[idx], conf[idx]

    def _get_pr_curve(
        self,
        targs: list[str] | list[list[str]],
        preds: list[str] | list[list[str]],
        conf: np.ndarray,
        N: int = 20,  # noqa: N803
    ) -> tuple[list[float], list[float]]:
        x, y = [], []
        t_idx = np.argsort(np.array(conf))
        t_idx = t_idx[~conf[t_idx].isna()]
        t_idx = list(t_idx[(t_idx.shape[0] * np.arange(N) / N).astype(int)]) + [t_idx[-1]]
        for t in conf[t_idx]:
            _, _, recall, precision = self.compute_precision_recall(targs, preds, conf, t)
            x.append(recall)
            y.append(precision)
        return x, y

    def _mass(self, seq: str | list[str], charge: int | None = None) -> float:
        """Calculate a peptide's mass or m/z."""
        seq = self._split_peptide(seq)
        return self.residue_set.get_sequence_mass(seq, charge)  # type: ignore
        # calc_mass = sum([self.residues[aa] for aa in seq]) + H2O_MASS

        # if charge is not None:
        #     # Neutral mass
        #     calc_mass = (calc_mass / charge) + PROTON_MASS_AMU

        # return calc_mass

    def _calc_mass_error(
        self, mz_theoretical: float, mz_measured: float, charge: int, isotope: int = 0
    ) -> float:
        """Calculate the mass error between theoretical and actual mz in ppm."""
        return float(mz_theoretical - (mz_measured - isotope * CARBON_MASS_DELTA / charge)) / mz_measured * 10**6  # type: ignore

    # Adapted from https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/evaluate.py
    def _novor_match(
        self,
        a: list[str],
        b: list[str],
    ) -> int:
        """Number of AA matches with novor method."""
        n = 0

        mass_a: list[float] = [self.residue_set.get_mass(x) for x in a]
        mass_b: list[float] = [self.residue_set.get_mass(x) for x in b]
        cum_mass_a = np.cumsum(mass_a)
        cum_mass_b = np.cumsum(mass_b)

        i, j = 0, 0
        while i < len(a) and j < len(b):
            if abs(cum_mass_a[i] - cum_mass_b[j]) < self.cum_mass_threshold:
                n += int(abs(mass_a[i] - mass_b[j]) < self.ind_mass_threshold)
                i += 1
                j += 1
            elif cum_mass_a[i] > cum_mass_b[j]:
                i += 1
            else:
                j += 1
        return n

    def _novor_match_ptm(
        self,
        a: list[str],
        b: list[str],
    ) -> tuple[int, dict]:
        """Number of AA matches and PTM matches with novor method."""
        # flake8: noqa: C901
        n = 0
        n_ptm = {ptm: 0 for ptm in self.ptms}

        mass_a: list[float] = [self.residue_set.get_mass(x) for x in a]
        mass_b: list[float] = [self.residue_set.get_mass(x) for x in b]
        cum_mass_a = np.cumsum(mass_a)
        cum_mass_b = np.cumsum(mass_b)

        i, j = 0, 0
        while i < len(a) and j < len(b):
            if abs(cum_mass_a[i] - cum_mass_b[j]) < self.cum_mass_threshold:
                if abs(mass_a[i] - mass_b[j]) < self.ind_mass_threshold:
                    n += 1
                    for ptm, ptm_weight in self.ptms.items():
                        # Using (+79.97) and (+15.99) for both target and prediction
                        if ptm_weight in a[i] and ptm_weight in b[j]:
                            n_ptm[ptm] += 1
                i += 1
                j += 1
            elif cum_mass_a[i] > cum_mass_b[j]:
                i += 1
            else:
                j += 1
        return n, n_ptm

    def _count_ptms(self, peptide: list[str]) -> dict[str, int]:
        ptm_counts = {ptm: 0 for ptm in self.ptms}
        for aa in peptide:
            for ptm, mass in self.ptms.items():
                if mass in aa:
                    ptm_counts[ptm] += 1
        return ptm_counts

    def _get_peptide_matches(
        self, targets: list[str] | list[list[str]], predictions: list[str] | list[list[str]]
    ) -> list[bool]:
        matches: list[bool] = []
        for i in range(len(targets)):
            targ = self._split_peptide(targets[i])
            pred = self._split_peptide(predictions[i])
            if len(pred) > 0 and pred[0] == "":
                pred = []
            n_match = self._novor_match(targ, pred)
            matches.append(len(pred) == len(targ) and len(targ) == n_match)
        return matches


if __name__ == "__main__":
    print("debugging metrics")
    residue_set = ResidueSet(
        residue_masses={
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "S(p)": 166.998028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "T(p)": 181.01367,
            # "C(+57.02)": 160.030649,
            "C": 160.030649,  # V1
            "L": 113.084064,
            "I": 113.084064,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "Y(p)": 243.029329,
            "W": 186.079313,
            # "M(+15.99)": 147.035400,
            "M(ox)": 147.035400,  # V1
            "N(+.98)": 115.026943,
            "Q(+.98)": 129.042594,
        }
    )
    metrics = Metrics(
        residue_set, isotope_error_range=[0, 1], ptms={"(ox)": "(ox)", "(p)": "(p)"}
    )  # Hacked for testing
    # targets = ["PEPTIDE", "LINDQKEM(ox)H", "LINDQKEMH", "Y"]
    # predictions = ["PEPTIDE", "LINDQKEM(ox)H", "LINDQKEMF", "Y"]
    # targets = ["Y(p)AAAAAYF", "Y(p)AAAAAYF"]
    # predictions = ["Y(p)AAAAAY(p)H", "Y(p)AAAAAYF"]
    targets = ["Y(p)AAAAAYFM(ox)", "Y(p)AAAAAYFM(ox)"]
    predictions = ["Y(p)AAAAAY(p)FM(ox)", "Y(p)AAAAAYFM"]

    (
        aa_prec,
        aa_recall,
        pep_prec,
        pep_recall,
        ptm_prec,
        ptm_recall,
        pred_bool,
    ) = metrics.compute_precision_recall_ptm(targets, predictions)
    print(
        f"aa_prec: \t{aa_prec:.2f}\naa_recall: \t{aa_recall:.2f}\npep_recall: \t{pep_recall:.2f}\nphos_prec: \t{ptm_prec}\nphos_recall: \t{ptm_recall}"
    )

    print(metrics.ptms)

    # print("debugging matches_precursor")
    # pred_value = []
    # precursor_mz = 1000
    # precursor_charge = 2
    # delta_mass = np.min(
    #     np.abs(metrics.matches_precursor(pred_value, precursor_mz, precursor_charge)[1])
    # )
    # print(f"delta_mass: {delta_mass}")
