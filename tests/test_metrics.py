"""Regression tests for the residue matching in instanovo.utils.metrics (issue #13)."""

from __future__ import annotations

import pytest

from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

RESIDUES = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "T": 101.047670,
    "I": 113.084064,
    "L": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "E": 129.042593,
    "K": 128.094963,
    "S(+79.97)": 166.998028,
}


@pytest.fixture
def metrics() -> Metrics:
    return Metrics(ResidueSet(RESIDUES), isotope_error_range=[0, 1], ptms={"(p)": "(+79.97)"})


def test_identical_peptides_match_every_residue(metrics: Metrics) -> None:
    peptide = list("PEPTIDEK")
    assert metrics._novor_match(peptide, peptide) == len(peptide)


@pytest.mark.parametrize(
    ("a", "b"),
    [(list("GGPEPTIDE"), list("NPEPTIDE")), (list("NPEPTIDE"), list("GGPEPTIDE"))],
)
def test_matching_resynchronises_after_a_mismatch(metrics: Metrics, a: list[str], b: list[str]) -> None:
    # GG and N have the same mass, so the prefixes realign and PEPTIDE matches.
    assert metrics._novor_match(a, b) == 7


def test_ptm_matching_resynchronises_after_a_mismatch(metrics: Metrics) -> None:
    target = ["G", "G", "P", "E", "P", "S(+79.97)", "I", "D", "E"]
    prediction = ["N", "P", "E", "P", "S(+79.97)", "I", "D", "E"]
    n, n_ptm = metrics._novor_match_ptm(target, prediction)
    assert n == 7
    assert n_ptm == {"(p)": 1}
