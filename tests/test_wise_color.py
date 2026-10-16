"""The WISE-colour convention must match how the model was trained: no WISE source
at all -> colour 0.0 (training writes the flag into all four mags, and flag - flag
= 0.0); a source with a band unmeasured -> flag (that band is NaN in training, which
maps to the flag); both bands present -> the real difference."""

import pytest

# oracle_boom_ztf imports the oracle package at module load; skip where it is absent
# (e.g. a bare env) rather than fail collection.
oracle_boom_ztf = pytest.importorskip("oracle_support.oracle_boom_ztf")
_wise_color = oracle_boom_ztf._wise_color
flag_value = oracle_boom_ztf.flag_value


def test_no_wise_source_gives_zero_colour():
    # all four bands flagged -> the no-source case -> 0.0, not flag
    assert _wise_color(flag_value, flag_value, all_missing=True) == 0.0


def test_masked_band_gives_flag():
    assert _wise_color(15.0, flag_value, all_missing=False) == flag_value
    assert _wise_color(flag_value, 12.0, all_missing=False) == flag_value


def test_both_bands_present_gives_real_difference():
    assert _wise_color(15.0, 12.0, all_missing=False) == 3.0
