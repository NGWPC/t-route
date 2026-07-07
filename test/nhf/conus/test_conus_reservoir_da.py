from pathlib import Path

import pytest
import geopandas as gpd
import numpy as np
from troute.nhf_preprocess import WATERBODY_DF_FIELDS
from ..utils.integration_helpers import (
    assert_output_dimensions_and_validity,
    delete_outputs,
    has_files,
    load_lakeout,
    run_troute,
    skip_if_not_built,
)
from ..utils.make_configs import Config, DataAssimilationParameters
from ..utils.make_forcing import build_forcing_dataset
from ..utils.make_da import DAConfig, build_da_dataset

START_TIME = "2000-01-01 00:00"
END_TIME = "2000-01-01 04:00"
FORCING_MODE = "constant"
CONSTANT_QLAT = 100

DATA_DIR = Path(__file__).parent / "data" / "reservoir_da"
DA_PARAMS = DataAssimilationParameters(
    reservoir_persistence_usgs=True,
    reservoir_persistence_usace=True,
    reservoir_persistence_usbr=True,
    reservoir_rfc_forecasts=True,
    timeslice_lookback_hours=1,
    usgs_timeslices_folder="reservoir_da/usgs_da",
    usace_timeslices_folder="reservoir_da/usace_da",
    usbr_timeslices_folder="reservoir_da/usbr_da",
    reservoir_rfc_forecasts_time_series_path="reservoir_da/rfc_da",
    reservoir_rfc_forecasts_lookback_hours=28,
    reservoir_rfc_forecasts_offset_hours=0,
    reservoir_rfc_forecast_persist_days=11,
)
CFG = Config(
    DATA_DIR,
    START_TIME,
    END_TIME,
    data_assimilation_parameters=DA_PARAMS,
    lakeout_output="lakeout",
    max_loop_size=1,
)
RESERVOIR_TYPE_MAPPING = {"usgs": 2, "usace": 3, "rfc": 4, "usbr": 7}


def assert_da_outflows(
    lakeout_dir: Path, domain_path: Path, expected: float, atol: float
) -> None:
    """Assert outflow equals expected for DA-forced lakes, excluding timesteps where water_sfc_elev > LkMxE."""
    # Load data
    ds_lake = load_lakeout(lakeout_dir)
    forced_ids = (
        gpd.read_file(
            domain_path,
            layer="reservoir_da",
            where="da_type != 1",
            ignore_geometry=True,
            columns=["nhf_lake_id"],
        )["nhf_lake_id"]
        .dropna()
        .astype(int)
        .values
    )
    # Note: T-route may drop some lakes, so this intersection may silently hide dropped lakes.
    valid_ids = list(set(forced_ids).intersection(ds_lake["feature_id"].values))
    ds_lake = ds_lake.sel(feature_id=valid_ids)

    # Load lakes and their properties
    lakes = gpd.read_file(
        domain_path, layer="lakes", ignore_geometry=True, columns=WATERBODY_DF_FIELDS
    )
    lkmxe = (
        lakes.dropna()
        .set_index("nhf_lake_id")["LkMxE"]
        .reindex(ds_lake["feature_id"].values)
        .values
    )

    wse = ds_lake["water_sfc_elev"].values
    exempt = (wse > lkmxe[np.newaxis, :]) | np.isnan(lkmxe[np.newaxis, :])

    outflow = ds_lake["outflow"].values.copy()
    outflow[exempt] = expected
    ds_lake.close()

    # Assert
    np.testing.assert_allclose(
        outflow,
        expected,
        atol=atol,
        err_msg="Lakeout outflow values are not near expected DA discharge (excluding weir overflow)",
    )


def get_lake_ids_for_da_type(gpkg_path: Path, da_type: int) -> list[int]:
    where = f"da_type = {da_type}"
    return (
        gpd.read_file(
            gpkg_path,
            layer="reservoir_da",
            where=where,
            ignore_geometry=True,
            columns=["site_no"],
        )["site_no"]
        .dropna()
        .astype(str)
        .values
    )


def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing for a standard test case."""
    if refresh or not CFG.config_path.exists():
        CFG.write_yaml()

    if refresh or not CFG.domain_path.exists():
        if CFG.domain_path.exists():
            CFG.domain_path.unlink()
        CFG.domain_path.symlink_to(source_gpkg)

    if refresh or not has_files(CFG.channel_forcing_dir, CFG.qlat_file_pattern):
        build_forcing_dataset(
            FORCING_MODE,
            START_TIME,
            END_TIME,
            CFG.channel_forcing_dir,
            CFG.domain_path,
            constant_qlat=CONSTANT_QLAT,
        )

    # Generate DA data for each persistence type (USGS, USACE, USBR).
    for da_name in RESERVOIR_TYPE_MAPPING:
        if da_name == "rfc":
            continue
        target_dir = getattr(CFG, f"{da_name}_timeslices_dir")
        suffix_map = {
            "usgs": "usgsTimeSlice",
            "usace": "usaceTimeSlice",
            "usbr": "usbrTimeSlice",
        }
        if refresh or not has_files(target_dir, f"*.{suffix_map[da_name]}.ncdf"):
            ids = get_lake_ids_for_da_type(
                CFG.domain_path, RESERVOIR_TYPE_MAPPING[da_name]
            )
            build_da_dataset(
                DAConfig(
                    da_type=da_name,
                    station_ids=ids,
                    start_time=START_TIME,
                    end_time=END_TIME,
                    output_dir=target_dir,
                    discharge=0.1,
                )
            )

    # Generate RFC DA data.
    if refresh or not has_files(CFG.rfc_timeslices_dir, "*.RFCTimeSeries.ncdf"):
        ids = get_lake_ids_for_da_type(CFG.domain_path, RESERVOIR_TYPE_MAPPING["rfc"])
        for idx in ids:
            build_da_dataset(
                DAConfig(
                    da_type="rfc",
                    station_ids=[idx],
                    start_time=START_TIME,
                    end_time=END_TIME,
                    output_dir=CFG.rfc_timeslices_dir,
                    discharge=0.1,
                )
            )


@pytest.mark.integration
def test_conus_reservoir_da():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    delete_outputs(CFG.lakeout_dir)
    run_troute(CFG.config_path)
    assert_output_dimensions_and_validity(
        CFG.output_dir, CFG.domain_path, CFG.config_path
    )
    assert_da_outflows(CFG.lakeout_dir, CFG.domain_path, expected=0.1, atol=0.05)
