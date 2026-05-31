"""Tests for ``troute.nhf_preprocess``.

Covers:
  * ``_validate_flowpaths_channel_params`` -- the load-time guard that
    rejects ``flowpaths`` containing non-finite (NaN/Inf) channel parameters
    which would otherwise propagate into NaN routing output via ``sqrt(s0)``
    in the Muskingum-Cunge kernel.
  * ``_validate_required_columns`` -- the load-time guard that rejects input
    geopackages missing columns the NHF build requires (e.g. ``segment_order``
    in ``reference_flowpaths``), replacing cryptic downstream ``KeyError``s.
"""

import numpy as np
import pandas as pd
import pytest

from troute.nhf_preprocess import (
    _BAD_FPID_PREVIEW_LIMIT,
    _FLOWPATHS_CHANNEL_COLS,
    LAYERS_TO_READ,
    _groupby_to_list_dict,
    _validate_flowpaths_channel_params,
    _validate_required_columns,
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


# ----- _validate_required_columns ------------------------------------------
# Guards that the input geopackage carries every column the NHF build needs.
# The reported production failure was a missing ``segment_order`` column in
# ``reference_flowpaths`` surfacing as ``KeyError: ['segment_order'] not in
# index`` deep inside discretization.


def _required_for(layer):
    """Mirror the validator's rule for a layer's required column set: a
    layer's explicit ``columns`` list (None-columns layers are not validated)."""
    cols = layer["columns"]
    return list(cols) if isinstance(cols, list) else []


def _make_valid_table_dict():
    """Build a table_dict satisfying every layer's required columns.

    Required layers get a (zero-row) DataFrame carrying exactly their required
    columns; non-required layers (lakes, gages, ...) get bare empty frames, as
    they would when absent from a geopackage.
    """
    table_dict = {}
    for layer in LAYERS_TO_READ:
        required = _required_for(layer)
        if required:
            table_dict[layer["name"]] = pd.DataFrame(columns=required)
        else:
            table_dict[layer["name"]] = pd.DataFrame()
    return table_dict


def _layers_with_requirements():
    return [layer for layer in LAYERS_TO_READ if _required_for(layer)]


def test_required_columns_valid_dataset_passes():
    _validate_required_columns(_make_valid_table_dict())


def test_required_columns_reference_flowpaths_spec_includes_segment_order():
    # Guard the spec itself: the layer that caused the production bug must
    # declare segment_order as required.
    rf = next(l for l in LAYERS_TO_READ if l["name"] == "reference_flowpaths")
    assert "segment_order" in _required_for(rf)


def test_required_columns_missing_segment_order_raises():
    # The exact production failure: reference_flowpaths without segment_order.
    td = _make_valid_table_dict()
    td["reference_flowpaths"] = td["reference_flowpaths"].drop(columns=["segment_order"])
    with pytest.raises(ValueError) as excinfo:
        _validate_required_columns(td)
    msg = str(excinfo.value)
    assert "reference_flowpaths" in msg
    assert "segment_order" in msg
    # Message should be actionable about hydrofabric versioning.
    assert "hydrofabric" in msg


def test_required_columns_missing_flowpaths_col_raises():
    # flowpaths requirements are derived from its explicit `columns` list.
    td = _make_valid_table_dict()
    assert "n" in td["flowpaths"].columns  # sanity
    td["flowpaths"] = td["flowpaths"].drop(columns=["n"])
    with pytest.raises(ValueError) as excinfo:
        _validate_required_columns(td)
    msg = str(excinfo.value)
    assert "flowpaths" in msg
    assert "'n'" in msg


def test_required_columns_aggregates_across_layers():
    # Multiple layers missing columns -> a single error listing all of them.
    td = _make_valid_table_dict()
    td["reference_flowpaths"] = td["reference_flowpaths"].drop(columns=["segment_order"])
    td["virtual_flowpaths"] = td["virtual_flowpaths"].drop(columns=["dn_virtual_nex_id"])
    with pytest.raises(ValueError) as excinfo:
        _validate_required_columns(td)
    msg = str(excinfo.value)
    assert "reference_flowpaths" in msg and "segment_order" in msg
    assert "virtual_flowpaths" in msg and "dn_virtual_nex_id" in msg


def test_required_columns_optional_empty_layers_do_not_raise():
    # lakes/gages/hydrolocations/virtual_nexus are used conditionally and may
    # be legitimately empty/absent; they must not trigger validation.
    td = _make_valid_table_dict()
    for name in ("lakes", "gages", "hydrolocations", "virtual_nexus"):
        assert td[name].empty
    _validate_required_columns(td)  # must not raise


def test_required_columns_absent_layer_reports_full_required_set():
    # A layer entirely missing from the geopackage is replaced upstream with a
    # bare empty DataFrame (no columns) -> its full required set is reported.
    td = _make_valid_table_dict()
    td["reference_flowpaths"] = pd.DataFrame()  # simulate missing layer
    with pytest.raises(ValueError) as excinfo:
        _validate_required_columns(td)
    msg = str(excinfo.value)
    for col in _required_for(
        next(l for l in LAYERS_TO_READ if l["name"] == "reference_flowpaths")
    ):
        assert col in msg


def test_required_columns_none_layer_reports_full_required_set():
    # Defensive: a layer key absent from the dict entirely (None) is treated as
    # missing all its required columns rather than crashing.
    td = _make_valid_table_dict()
    td["reference_flowpaths"] = None
    with pytest.raises(ValueError) as excinfo:
        _validate_required_columns(td)
    assert "reference_flowpaths" in str(excinfo.value)


def test_required_columns_zero_row_layer_with_schema_passes():
    # A present-but-empty layer still carries its schema columns, so a 0-row
    # frame with the right columns must pass (distinct from an absent layer).
    td = _make_valid_table_dict()
    for layer in _layers_with_requirements():
        assert td[layer["name"]].empty  # zero rows...
    _validate_required_columns(td)  # ...but columns present -> passes


def test_required_columns_extra_columns_are_ignored():
    # Datasets carrying additional columns beyond the required set are fine.
    td = _make_valid_table_dict()
    td["reference_flowpaths"]["some_extra_col"] = pd.Series(dtype=float)
    _validate_required_columns(td)


# ----- _groupby_to_list_dict (Step N2 helper) ------------------------------
# These tests exist to guarantee parity with the pandas idiom
#   df.groupby(key)[val].apply(list).to_dict()
# that this helper replaces. Anything pandas does (NaN-key handling, key
# unboxing, empty-frame behavior) the helper must match for the
# crosswalk_nex_flowpath_poi callers in nhf_preprocess to stay correct.


def _pandas_reference(df, key, val):
    """The legacy pandas idiom the helper replaces."""
    return df.groupby(key)[val].apply(list).to_dict()


def test_groupby_helper_matches_pandas_basic_int_keys():
    df = pd.DataFrame({"k": [10, 20, 10, 30, 20], "v": [1, 2, 3, 4, 5]})
    out = _groupby_to_list_dict(df, "k", "v")
    assert out == _pandas_reference(df, "k", "v")
    # And the keys are python ints, not numpy scalars.
    assert all(type(k) is int for k in out)


def test_groupby_helper_matches_pandas_with_nan_keys():
    # Pandas' default groupby drops NaN keys; the numpy helper must too.
    df = pd.DataFrame({
        "k": [10.0, np.nan, 10.0, 20.0, np.nan, 30.0],
        "v": [1, 2, 3, 4, 5, 6],
    })
    out = _groupby_to_list_dict(df, "k", "v")
    ref = _pandas_reference(df, "k", "v")
    assert out == ref
    # Explicitly: no NaN key in the output.
    assert not any(isinstance(k, float) and pd.isna(k) for k in out)


def test_groupby_helper_matches_pandas_all_nan_keys():
    df = pd.DataFrame({
        "k": [np.nan, np.nan, np.nan],
        "v": [1, 2, 3],
    })
    out = _groupby_to_list_dict(df, "k", "v")
    assert out == {}
    assert out == _pandas_reference(df, "k", "v")


def test_groupby_helper_matches_pandas_object_keys():
    df = pd.DataFrame({
        "k": ["a", "b", "a", "c"],
        "v": [1, 2, 3, 4],
    })
    out = _groupby_to_list_dict(df, "k", "v")
    assert out == _pandas_reference(df, "k", "v")


def test_groupby_helper_empty_frame():
    df = pd.DataFrame({"k": [], "v": []}, dtype=float)
    assert _groupby_to_list_dict(df, "k", "v") == {}
