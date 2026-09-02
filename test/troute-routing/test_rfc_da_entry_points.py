"""The two RFC entry points must ingest on the same schedule.

`mc_reach` calls the DA with `dt * timestep`, the end of the interval it just routed,
so `model_reservoir` must pass the same instant rather than its own start-of-interval
clock. A one step lag lands on a different forecast sample once per cadence: 24 of 288
steps over a day at dt=300. Sharing the parser is not enough; the callers also have to
agree on what "model time" means.

Discovery has the same shape of split: both must fall back to an older issue that
covers t0 rather than stopping at the newest filename and running level pool.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from troute.DataAssimilation import _read_timeseries_files, assemble_rfc_dataframes
from troute.routing.fast_reach.reservoir_RFC_da import preprocess_RFC_data, reservoir_RFC_da

_FIXTURE = (
    Path(__file__).parents[1] / "BMI" / "rfc_timeseries"
    / "2021-10-22_12.60min.KNFC1.RFCTimeSeries.ncdf"
)
_KERNEL = {
    "rfc_forecast_persist_seconds": 11 * 86400, "reservoir_type": 4, "inflow": 1.0,
    "water_elevation": 10.0, "levelpool_outflow": -1.0, "levelpool_water_elevation": 10.0,
    "lake_area": 1e9, "max_water_elevation": 20.0, "rfc_file": "f",
}


def _drive(series, idx, update_time, *, routing_period, hours, first_step):
    """Step the DA, with `first_step` selecting the caller's clock convention."""
    for step in range(int(hours * 3600 / routing_period)):
        _, _, update_time, idx, *_ = reservoir_RFC_da(
            True, series, idx, len(series) - 1, routing_period=float(routing_period),
            current_time=float(routing_period * (step + first_step)),
            update_time=update_time, DA_time_step=3600, **_KERNEL
        )
    return idx


def _kernel_seed():
    """What assemble_rfc_dataframes hands the routing kernel for the fixture gage."""
    t0 = datetime(2021, 10, 21, 12)
    dates, d = [], t0
    while d <= t0 + timedelta(hours=28):
        dates.append(d.strftime("%Y-%m-%d_%H"))
        d += timedelta(hours=1)
    raw = _read_timeseries_files(str(_FIXTURE.parent), dates, t0, t0 + timedelta(days=11))
    crosswalk = pd.DataFrame(
        {"rfc_gage_id": ["KNFC1"], "rfc_lake_id": [1]}
    ).set_index("rfc_lake_id")
    obs, params = assemble_rfc_dataframes(
        raw, crosswalk, pd.Timestamp(t0), {"reservoir_rfc_forecast_persist_days": 11}
    )
    return (
        obs.to_numpy(dtype="float32")[0],
        int(params["timeseries_idx"].iloc[0]),
        int(params["totalCounts"].iloc[0]),
        float(params["update_time"].iloc[0]),
        int(params["da_timestep"].iloc[0]),
    )


def _standalone_seed():
    """What preprocess_RFC_data hands the standalone reservoir BMI for the same gage."""
    use_rfc, series, idx, update_time, cadence, counts, _ = preprocess_RFC_data(
        "2021-10-21_12:00:00", 28, "KNFC1", f"{_FIXTURE.parent}/", 1, 300
    )
    assert use_rfc is True
    return np.asarray(series, dtype="float32"), idx, counts, float(update_time), cadence


@pytest.mark.parametrize("routing_period", [300, 900, 1800, 3600])
def test_both_entry_points_reach_the_same_forecast_sample(routing_period):
    """Same seeding, same clock, so the same sample at any dt up to the cadence.

    Seeding one index before t0 and relying on the first call to burn an advance is
    invisible while dt < cadence and one sample late at it, so the drivers would agree
    only below the cadence.

    Both arms are seeded from the real functions, not from a literal: seeding them alike
    by hand would make the comparison a tautology and miss exactly the divergence this
    exists to catch.
    """
    kernel_series, k_idx, _, k_update, cadence = _kernel_seed()
    standalone_series, s_idx, _, s_update, s_cadence = _standalone_seed()
    assert (k_idx, k_update, cadence) == (s_idx, s_update, s_cadence)

    hours = 24
    kernel = _drive(
        kernel_series, k_idx, k_update, routing_period=routing_period, hours=hours, first_step=1
    )
    standalone = _drive(
        standalone_series, s_idx, s_update, routing_period=routing_period,
        hours=hours, first_step=1,
    )
    assert kernel == standalone
    # And it is the sample the model time asks for: i0 + floor(elapsed / cadence).
    assert kernel == k_idx + (hours * 3600) // cadence


def test_standalone_falls_back_to_an_older_covering_forecast(tmp_path):
    shutil.copy(_FIXTURE, tmp_path / "2021-10-21_12.60min.KNFC1.RFCTimeSeries.ncdf")
    newest = tmp_path / "2021-10-22_12.60min.KNFC1.RFCTimeSeries.ncdf"
    shutil.copy(_FIXTURE, newest)
    with netCDF4.Dataset(str(newest), "a") as ds:
        ds.sliceStartTimeUTC = "2021-10-22_00:00:00"  # begins after t0

    use_rfc, _, idx, _, _, _, chosen = preprocess_RFC_data(
        "2021-10-21_12:00:00", 28, "KNFC1", f"{tmp_path}/", 1, 300
    )
    assert chosen == "2021-10-21_12.60min.KNFC1.RFCTimeSeries.ncdf"
    assert use_rfc is True
    assert idx == 24


def test_standalone_still_reports_no_usable_forecast(tmp_path):
    """Nothing covering t0 anywhere means level pool, not a wrong file."""
    only = tmp_path / "2021-10-22_12.60min.KNFC1.RFCTimeSeries.ncdf"
    shutil.copy(_FIXTURE, only)
    with netCDF4.Dataset(str(only), "a") as ds:
        ds.sliceStartTimeUTC = "2021-10-22_00:00:00"
    use_rfc, *_ = preprocess_RFC_data(
        "2021-10-21_12:00:00", 28, "KNFC1", f"{tmp_path}/", 1, 300
    )
    assert use_rfc is False
