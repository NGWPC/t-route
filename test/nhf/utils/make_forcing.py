"""Generate channel forcing files and a config YAML for running an NHF test case.

Three forcing modes are available via --forcing-mode:

  retro   (default) Pull lateral inflows from the NWM v3.0 retrospective Zarr
            store on S3.  Requires --start-time, --end-time, and an NHF gpkg
            with a ``reference_flowpaths`` layer.

  pulse   Apply a synthetic unit-hydrograph pulse scaled to --peak-qlat
            (m³/s, default 10 000) uniformly across all reaches.  The pulse
            shape is a 22-step rising/falling limb padded with leading zeros
            and a long recession tail.  Only --start-time is used to set the
            output filename timestamps.

  constant  Apply a constant qlat of --constant-qlat (m³/s, default 1.0)
            to every reach for every timestep in the [start-time, end-time]
            window.
"""

import argparse
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


### CONSTANT DEFINITION ###

RETRO_PATH = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
RETROSPECTIVE_LATERAL_FIELD = "q_lateral"

# Unit-hydrograph shape shared by pulse forcing
_PULSE_SHAPE = np.array(
    [
        0.0,
        0.03,
        0.10,
        0.19,
        0.31,
        0.47,
        0.66,
        0.82,
        0.93,
        0.99,
        1.00,
        0.99,
        0.93,
        0.86,
        0.78,
        0.68,
        0.56,
        0.42,
        0.27,
        0.18,
        0.08,
        0.03,
    ]
)


def _write_runout_steps(
    last_time: pd.Timestamp,
    feature_ids,
    forcing_dir: Path,
    runout_time: int,
    forcing_file_pattern: str,
) -> None:
    """Append zero-qlat runout CSVs after the primary simulation window."""
    for i in range(1, runout_time + 1):
        print(f"Processing runout time step {i}...")
        t_str = (last_time + pd.Timedelta(hours=i)).strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: 0.0})
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)


def create_pulse_forcing_dataset(
    t_start: str,
    forcing_dir: str,
    hydrofabric_path: str,
    t_end: str,
    peak_qlat: float = 10000.0,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
    runout_time: int = 0,
) -> None:
    """Write per-timestep CSVs driven by a synthetic pulse scaled to *peak_qlat*.

    The pulse is applied uniformly to every reach in the hydrofabric.  When
    *t_end* is provided the unit-hydrograph shape is linearly interpolated to
    span exactly the [t_start, t_end] window; otherwise the native shape length
    determines the number of timesteps.
    """
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    feature_ids = fps["fp_id"].astype(int).values

    times = pd.date_range(t_start, t_end, freq="h")
    n = len(times)
    # Linearly interpolate the unit-hydrograph shape to n samples
    xp = np.linspace(0, 1, len(_PULSE_SHAPE))
    xi = np.linspace(0, 1, n)
    shape = np.interp(xi, xp, _PULSE_SHAPE)


    inflows = shape * peak_qlat

    for t, q in zip(times, inflows):
        t_str = t.strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: q})
        df.to_csv(
            forcing_dir / f"{t_str}.{forcing_file_pattern}.csv",
            index=False,
            float_format="%.15g",
        )
        print(f"Processing time step {t}...")
    _write_runout_steps(
        times[-1], feature_ids, forcing_dir, runout_time, forcing_file_pattern
    )


def create_constant_forcing_dataset(
    t_start: str,
    t_end: str,
    forcing_dir: str,
    hydrofabric_path: str,
    constant_qlat: float = 1.0,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
    runout_time: int = 0,
) -> None:
    """Write per-timestep CSVs with a constant *constant_qlat* at every reach."""
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    feature_ids = fps["fp_id"].astype(int).values

    times = pd.date_range(t_start, t_end, freq="h")
    for t in times:
        t_str = t.strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: constant_qlat})
        df.to_csv(
            forcing_dir / f"{t_str}.{forcing_file_pattern}.csv",
            index=False,
            float_format="%.15g",
        )
        print(f"Processing time step {t}...")
    _write_runout_steps(
        times[-1], feature_ids, forcing_dir, runout_time, forcing_file_pattern
    )


def create_forcing_dataset(
    t_start: str,
    t_end: str,
    forcing_dir: str,
    hydrofabric_path: str,
    retrospective_path: str,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
    runout_time: int = 0,
):
    """Create a dataset of channel forcing files from retrospective data."""
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    crosswalk = gpd.read_file(hydrofabric_path, layer="reference_flowpaths")
    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    retro = xr.open_zarr(
        retrospective_path,
        storage_options={"anon": True},
    )

    # Post-process
    feature_ids_retro = crosswalk["ref_fp_id"].values
    crosswalk = pd.merge(
        crosswalk[["ref_fp_id", "div_id"]],
        fps[["fp_id", "div_id"]],
        left_on="div_id",
        right_on="div_id",
        how="left",
    )

    # Generate dataset
    iterator = pd.date_range(start=t_start, end=t_end, freq="h")
    for i in iterator:
        print(f"Processing time step {i}...")
        qlat = retro.sel(feature_id=feature_ids_retro, time=i)[
            RETROSPECTIVE_LATERAL_FIELD
        ].reset_coords(drop=True)
        t_str = i.strftime("%Y%m%d%H%M")
        df = qlat.to_dataframe()
        df = pd.merge(
            df,
            crosswalk[["ref_fp_id", "fp_id"]],
            left_index=True,
            right_on="ref_fp_id",
            how="left",
        ).rename(columns={"fp_id": "feature_id", RETROSPECTIVE_LATERAL_FIELD: t_str})[
            ["feature_id", t_str]
        ]
        df["feature_id"] = df["feature_id"].astype(int)
        df = df.groupby("feature_id").sum().reset_index()
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)
    _write_runout_steps(
        iterator[-1],
        fps["fp_id"].astype(int).values,
        forcing_dir,
        runout_time,
        forcing_file_pattern,
    )


def build_forcing_dataset(
    forcing_mode: Literal["retro", "pulse", "constant"],
    t_start: str,
    t_end: str,
    forcing_dir: Path,
    hydrofabric_path: Path,
    runout_period: int = 0,
    peak_qlat: float = 0,
    constant_qlat: float = 0,
) -> None:
    """Build the forcing dataset and config YAML described by config."""
    if forcing_mode == "retro":
        create_forcing_dataset(
            t_start=t_start,
            t_end=t_end,
            forcing_dir=forcing_dir,
            hydrofabric_path=hydrofabric_path,
            retrospective_path=RETRO_PATH,
            runout_time=runout_period,
        )
    elif forcing_mode == "pulse":
        create_pulse_forcing_dataset(
            t_start=t_start,
            t_end=t_end,
            forcing_dir=forcing_dir,
            hydrofabric_path=hydrofabric_path,
            peak_qlat=peak_qlat,
            runout_time=runout_period,
        )
    elif forcing_mode == "constant":
        create_constant_forcing_dataset(
            t_start=t_start,
            t_end=t_end,
            forcing_dir=forcing_dir,
            hydrofabric_path=hydrofabric_path,
            constant_qlat=constant_qlat,
            runout_time=runout_period,
        )
    else:
        raise ValueError(
            f"Unknown forcing_mode '{forcing_mode}'. Choose retro, pulse, or constant."
        )


def main():
    """Enter via CLI."""
    parser = argparse.ArgumentParser(description="Generate forcing dataset for a case.")

    parser.add_argument(
        "--hf-path",
        required=True,
        help="Path to the NHF GeoPackage.",
    )

    parser.add_argument(
        "--forcing-dir",
        required=True,
        help="Directory to write per-timestep forcing CSVs.",
    )

    parser.add_argument(
        "--start-time",
        default="2009-12-12 00:00",
        help="Simulation start time (e.g. '2009-12-12' or '2009-12-12 06:00').",
    )

    parser.add_argument(
        "--end-time",
        default="2009-12-29 00:00",
        help="Simulation end time (e.g. '2009-12-29' or '2009-12-29 12:00').",
    )

    parser.add_argument(
        "--forcing-mode",
        default="retro",
        choices=["retro", "pulse", "constant"],
        help=(
            "Forcing generation mode. "
            "'retro' (default): pull lateral inflows from the NWM v3 retrospective store. "
            "'pulse': apply a synthetic unit-hydrograph pulse to all reaches. "
            "'constant': apply a constant qlat to all reaches for every timestep."
        ),
    )

    parser.add_argument(
        "--runout-period",
        type=int,
        default=0,
        help="Hours of zero-qlat runout to append after the primary simulation window.",
    )

    parser.add_argument(
        "--peak-qlat",
        type=float,
        default=10_000.0,
        help="Peak discharge (m³/s) for the synthetic pulse. Only used when --forcing-mode=pulse.",
    )

    parser.add_argument(
        "--constant-qlat",
        type=float,
        default=1.0,
        help="Constant lateral inflow (m³/s) per reach. Only used when --forcing-mode=constant.",
    )

    args = parser.parse_args()

    build_forcing_dataset(
        forcing_mode=args.forcing_mode,
        t_start=args.start_time,
        t_end=args.end_time,
        forcing_dir=Path(args.forcing_dir),
        hydrofabric_path=Path(args.hf_path),
        runout_period=args.runout_period,
        peak_qlat=args.peak_qlat,
        constant_qlat=args.constant_qlat,
    )


if __name__ == "__main__":
    main()
