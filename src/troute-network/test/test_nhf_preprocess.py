"""Tests for ``troute.nhf_preprocess``.

Covers ``_validate_flowpaths_channel_params`` — the load-time guard that
rejects ``flowpaths`` containing non-finite (NaN/Inf) channel parameters
which would otherwise propagate into NaN routing output via ``sqrt(s0)``
in the Muskingum-Cunge kernel.
"""

import numpy as np
import pandas as pd
import pytest

from troute.nhf_preprocess import (
    _BAD_FPID_PREVIEW_LIMIT,
    _FLOWPATHS_CHANNEL_COLS,
    _validate_flowpaths_channel_params,
)


def _make_flowpaths(n=5, extra_cols=None):
    """Build a minimal valid flowpaths DataFrame with all channel cols finite."""
    data = {c: np.linspace(0.1, 1.0, n) for c in _FLOWPATHS_CHANNEL_COLS}
    data["fp_id"] = np.arange(1000, 1000 + n)
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


# ----- no-op cases ----------------------------------------------------------

def test_validator_none_is_noop():
    # Must not raise on missing input.
    _validate_flowpaths_channel_params(None)


def test_validator_empty_df_is_noop():
    _validate_flowpaths_channel_params(pd.DataFrame())


def test_validator_no_channel_cols_is_noop():
    # If none of the kernel-critical columns are present, nothing to validate.
    df = pd.DataFrame({"fp_id": [1, 2, 3], "something_else": [10.0, 20.0, 30.0]})
    _validate_flowpaths_channel_params(df)


def test_validator_all_finite_passes():
    df = _make_flowpaths(n=10)
    _validate_flowpaths_channel_params(df)


# ----- raising cases --------------------------------------------------------

def test_validator_raises_on_nan_slope():
    df = _make_flowpaths(n=4)
    df.loc[1, "slope"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "1 of 4 segments" in msg
    assert "slope" in msg
    assert "Muskingum-Cunge" in msg
    # The bad fp_id (1001) must appear in the preview.
    assert "1001" in msg


def test_validator_raises_on_positive_inf():
    df = _make_flowpaths(n=3)
    df.loc[2, "n"] = np.inf
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    assert "'n': 1" in str(excinfo.value)


def test_validator_raises_on_negative_inf():
    df = _make_flowpaths(n=3)
    df.loc[0, "btmwdth"] = -np.inf
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    assert "btmwdth" in str(excinfo.value)


def test_validator_reports_multiple_bad_columns():
    df = _make_flowpaths(n=4)
    df.loc[0, "slope"] = np.nan
    df.loc[1, "n"] = np.nan
    df.loc[2, "topwdth"] = np.inf
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    # 3 distinct affected rows.
    assert "3 of 4 segments" in msg
    # Each bad column should be reported individually.
    assert "'slope': 1" in msg
    assert "'n': 1" in msg
    assert "'topwdth': 1" in msg


def test_validator_counts_one_row_with_multiple_bad_cols_as_one_segment():
    # A single row with NaNs in two columns counts as 1 segment (row-wise)
    # but contributes 1 to each affected-column count.
    df = _make_flowpaths(n=3)
    df.loc[1, "slope"] = np.nan
    df.loc[1, "n"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "1 of 3 segments" in msg
    assert "'slope': 1" in msg
    assert "'n': 1" in msg


def test_validator_preview_capped_with_overflow_marker():
    # More than _BAD_FPID_PREVIEW_LIMIT bad fp_ids → preview is truncated
    # and the message includes an "(and N more)" marker.
    n = _BAD_FPID_PREVIEW_LIMIT + 5
    df = _make_flowpaths(n=n)
    df.loc[:, "slope"] = np.nan  # mark all rows bad
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert f"and {n - _BAD_FPID_PREVIEW_LIMIT} more" in msg


def test_validator_handles_missing_fp_id_column():
    # If fp_id is absent (e.g. test fixture), validation should still raise
    # but without listing IDs.
    df = _make_flowpaths(n=3).drop(columns=["fp_id"])
    df.loc[0, "slope"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "1 of 3 segments" in msg
    # Preview list should be empty (no fp_id column to draw from).
    assert "Affected fp_ids: []" in msg


def test_validator_ignores_non_channel_column_nans():
    # NaN in a column that is NOT a Muskingum-Cunge input should not trigger.
    df = _make_flowpaths(n=3, extra_cols={"unrelated_col": [1.0, np.nan, 3.0]})
    _validate_flowpaths_channel_params(df)


def test_validator_works_with_partial_channel_columns():
    # Real geopackages may not contain every entry in _FLOWPATHS_CHANNEL_COLS.
    # The validator should restrict to whichever subset is present.
    cols = ["fp_id", "slope", "n"]
    df = pd.DataFrame({
        "fp_id": [10, 20, 30],
        "slope": [0.1, np.nan, 0.3],
        "n": [0.03, 0.04, 0.05],
    })
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "'slope': 1" in msg
    assert "20" in msg
