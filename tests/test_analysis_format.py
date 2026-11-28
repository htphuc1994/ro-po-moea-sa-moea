"""Formatting of p-values in the analysis tables."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis  # noqa: E402


def test_p_value_just_below_005_is_printed_below_005():
    assert analysis.p_text(0.0498) == "0.0498"
    assert analysis.p_text(0.04996) == "0.04996"
    assert analysis.p_text(0.0501) == "0.050"
    assert analysis.p_text(0.0106) == "0.011"
    assert analysis.p_text(1.0) == "1.000"


def test_small_p_values_use_powers_of_ten():
    assert analysis.tex_p(0.0498) == "0.0498"
    assert analysis.tex_p(2.4e-15) == "$2.4\\times10^{-15}$"
