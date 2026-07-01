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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pyogrio.errors
import xarray as xr
import yaml
from dataretrieval import nwis

### CONSTANT DEFINITION ###

RETRO_PATH = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
RETROSPECTIVE_LATERAL_FIELD = "q_lateral"
RETROSPECTIVE_FLOW_FIELD = "streamflow"

# Unit-hydrograph shape shared by pulse forcing
_PULSE_SHAPE = np.array([
    0.0, 0.03, 0.10, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99,
    1.00, 0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.42, 0.27, 0.18,
    0.08, 0.03,
])
_PULSE_SHAPE = np.concatenate([np.zeros(3), _PULSE_SHAPE, np.zeros(25)])

### CONFIGURATION DATA CLASS ###

@dataclass
class ForcingConfig:
    """All parameters needed to build a forcing dataset and config YAML."""

    case_id: str
    hf_file: str
    run_id: str = "retro"
    start_time: str = "2009-12-12 00:00"
    end_time:   str = "2009-12-29 00:00"
    forcing_mode: Literal["retro", "pulse", "constant"] = "retro"

    # retro options
    generate_reference_data: bool = False
    add_runout_period:        bool = False
    # pulse options
    peak_qlat: float = 10_000.0
    # constant options
    constant_qlat: float = 1.0

    # path management
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def start_dt(self) -> pd.Timestamp:
        """Parsed start time as a pandas Timestamp."""
        return pd.to_datetime(self.start_time)

    @property
    def end_dt(self) -> pd.Timestamp:
        """Parsed end time as a pandas Timestamp."""
        return pd.to_datetime(self.end_time)

    @property
    def run_dir(self) -> Path:
        """Root directory for this case (base_dir / case_id)."""
        return self.base_dir / self.case_id

    @property
    def forcing_subdir(self) -> str:
        """Name of the forcing subdirectory within run_dir."""
        return f"channel_forcing_{self.run_id}"

    @property
    def config_path(self) -> Path:
        """Path to the generated t-route config YAML."""
        return self.run_dir / f"{self.run_id}.yaml"

    @property
    def forcing_dir(self) -> Path:
        """Absolute path to the directory where forcing CSVs will be written."""
        return self.run_dir / self.forcing_subdir

    @property
    def hf_path(self) -> Path:
        """Absolute path to the hydrofabric geopackage."""
        return self.run_dir / "domain" / self.hf_file

    @property
    def output_dir(self) -> str:
        """Output directory string written into the config YAML."""
        return f"output_{self.run_id}/"

    @property
    def hf_path_relative(self) -> str:
        """Hydrofabric path relative to run_dir, as used in the config YAML."""
        return f"domain/{self.hf_file}"

    @property
    def qlat_input_folder(self) -> str:
        """Forcing folder path with trailing slash, as used in the config YAML."""
        return f"{self.forcing_subdir}/"

    @property
    def restart_time(self) -> str:
        """start_dt formatted as the config YAML restart datetime string."""
        return self.start_dt.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def reference_dir(self) -> Path | None:
        """Path for reference output files, or None if not requested."""
        return (self.run_dir / self.run_id) if self.generate_reference_data else None

    @property
    def runout_time(self) -> int:
        """Extra hours of zero-qlat runout appended after end_time."""
        if self.add_runout_period:
            return int(((self.end_dt - self.start_dt) / 2).total_seconds() / 3600)
        return 0

    @property
    def nts(self) -> int:
        """Total number of 5-minute model timesteps."""
        dt = 300
        sim_time = (self.end_dt - self.start_dt).total_seconds()
        if self.forcing_mode != "pulse" and self.add_runout_period:
            sim_time += (self.runout_time + 1) * 3600
        return int(sim_time / dt)


def create_pulse_forcing_dataset(
    t_start: str,
    forcing_dir: str,
    hydrofabric_path: str,
    t_end: str | None = None,
    peak_qlat: float = 10_000.0,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
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
    feature_ids = fps["fp_id"].values

    if t_end is not None:
        times = pd.date_range(t_start, t_end, freq="h")
        n = len(times)
        # Linearly interpolate the unit-hydrograph shape to n samples
        xp = np.linspace(0, 1, len(_PULSE_SHAPE))
        xi = np.linspace(0, 1, n)
        shape = np.interp(xi, xp, _PULSE_SHAPE)
    else:
        times = pd.date_range(t_start, periods=len(_PULSE_SHAPE), freq="h")
        shape = _PULSE_SHAPE

    inflows = shape * peak_qlat

    for t, q in zip(times, inflows):
        t_str = t.strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: q})
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False, float_format="%.15g")
        print(f"Processing time step {t}...")


def create_constant_forcing_dataset(
    t_start: str,
    t_end: str,
    forcing_dir: str,
    hydrofabric_path: str,
    constant_qlat: float = 1.0,
    forcing_file_pattern: str = "CHRTOUT_DOMAIN1",
) -> None:
    """Write per-timestep CSVs with a constant *constant_qlat* at every reach."""
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    feature_ids = fps["fp_id"].values

    times = pd.date_range(t_start, t_end, freq="h")
    for t in times:
        t_str = t.strftime("%Y%m%d%H%M")
        df = pd.DataFrame({"feature_id": feature_ids, t_str: constant_qlat})
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False, float_format="%.15g")
        print(f"Processing time step {t}...")


def create_forcing_dataset(t_start: str, t_end: str, forcing_dir: str, hydrofabric_path: str, retrospective_path: str, forcing_file_pattern: str = "CHRTOUT_DOMAIN1", generate_reference_data: bool = False, reference_dir: Union[str, None] = None, runout_time: int = 0):
    """Create a dataset of channel forcing files from retrospective data."""
    forcing_dir = Path(forcing_dir)
    forcing_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    crosswalk = gpd.read_file(hydrofabric_path, layer="reference_flowpaths")
    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    retro = xr.open_zarr(retrospective_path, storage_options={"anon": True},)

    # Post-process
    feature_ids_retro = crosswalk["ref_fp_id"].values
    crosswalk = pd.merge(crosswalk[["ref_fp_id", "div_id"]], fps[["fp_id", "div_id"]], left_on="div_id", right_on="div_id", how="left")

    # Generate dataset
    iterator = pd.date_range(
        start=t_start,
        end=t_end,
        freq="h"
    )
    for i in iterator:
        print(f"Processing time step {i}...")
        qlat = retro.sel(feature_id=feature_ids_retro, time=i)[RETROSPECTIVE_LATERAL_FIELD].reset_coords(drop=True)
        t_str = i.strftime("%Y%m%d%H%M")
        df = qlat.to_dataframe()
        df = pd.merge(df, crosswalk[["ref_fp_id", "fp_id"]], left_index=True, right_on="ref_fp_id", how="left").rename(columns={"fp_id": "feature_id", RETROSPECTIVE_LATERAL_FIELD: t_str})[["feature_id", t_str]]
        df = df.groupby("feature_id").sum().reset_index()
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)
    for i in range(1, runout_time + 1):
        print(f"Processing runout time step {i}...")
        t_str = (iterator[-1] + pd.Timedelta(hours=i)).strftime("%Y%m%d%H%M")
        df = pd.DataFrame({
            "feature_id": fps["fp_id"],
            t_str: [0.0] * len(fps),
        })
        df.to_csv(forcing_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)

    # Generate reference outputs if requested
    if generate_reference_data and reference_dir is not None:
        try:
            gages = gpd.read_file(hydrofabric_path, sql="SELECT * FROM gages WHERE status = 'USGS-active'", ignore_geometry=True)
        except pyogrio.errors.DataLayerError:
            return
        data_vars = {}
        fp_ids = []
        site_nos = []
        first = True
        for _, gage in gages.iterrows():
            # Load retrospective flow for the gage's reference flowpath
            if pd.isna(gage["fp_id"]):
                continue
            fp_id = int(gage["fp_id"])
            ref_fp_id = crosswalk.loc[crosswalk["fp_id"] == fp_id, "ref_fp_id"].values[0]
            retro_q = retro.sel(feature_id=ref_fp_id, time=slice(t_start, t_end))[RETROSPECTIVE_FLOW_FIELD].reset_coords(drop=True).to_dataframe()
            retro_q.index = retro_q.index.tz_localize("UTC")

            # Load USGS data, if available
            site_no = gage["site_no"]
            usgs_raw = nwis.get_iv(site=site_no, start=t_start.strftime("%Y-%m-%dT%H:%MZ"), end=t_end.strftime("%Y-%m-%dT%H:%MZ"), parameterCd="00060")[0]
            if "00060" in usgs_raw.columns:
                usgs_q = usgs_raw.rename(columns={"00060": "usgs_q"})
                usgs_q.index = pd.to_datetime(usgs_q.index)
                usgs_q = usgs_q.reindex(retro_q.index, method="nearest", tolerance=pd.Timedelta("15min"))
            else:
                usgs_q = pd.DataFrame({"usgs_q": np.nan}, index=retro_q.index)

            if first:
                time_index = pd.DatetimeIndex(retro_q.index).tz_localize(None)
                first = False
            # Log metadata
            fp_ids.append(fp_id)
            site_nos.append(site_no)

            # Stack along gage dimension
            data_vars.setdefault("retrospective_q", []).append(retro_q[RETROSPECTIVE_FLOW_FIELD].values)
            data_vars.setdefault("usgs_q", []).append(usgs_q["usgs_q"].values)

        # Convert to arrays (gage x time)
        retrospective_array = np.stack(data_vars["retrospective_q"], axis=0)
        usgs_array = np.stack(data_vars["usgs_q"], axis=0)

        # Build xarray dataset
        ds = xr.Dataset(
            {
                "retrospective_q": (("gage", "time"), retrospective_array),
                "usgs_q": (("gage", "time"), usgs_array),
            },
            coords={
                "site_no": ("gage", site_nos),
                "fp_id": ("gage", fp_ids),
                "time": time_index,
            },
        )

        # Save single NetCDF
        ds.to_netcdf(Path(reference_dir) / "gage_reference_data.nc")



def make_config_yaml(config_path: str, hydrofabric_path: str, qlat_input_folder: str, nts: int, restart_time: str, output_dir: str, file_pattern_filter: str = "*.CHRTOUT_DOMAIN1.csv", max_loop_size: int = 288):
    """Create a config YAML for running the test case."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "log_parameters": {
            "showtiming": True,
            "log_level": "DEBUG",
        },
        "network_topology_parameters": {
            "supernetwork_parameters": {
                "geo_file_path": hydrofabric_path,
                "network_type": "NHF"
            },
            "waterbody_parameters": {
                "break_network_at_waterbodies": True,
            },
        },
        "compute_parameters": {
            "parallel_compute_method": "by-subnetwork-jit-clustered",
            "compute_kernel": "V02-structured",
            "assume_short_ts": True,
            "subnetwork_target_size": 10000,
            "cpu_pool": 1,
            "restart_parameters": {
                "start_datetime": restart_time,
            },
            "forcing_parameters": {
                "dt": 300,
                "qlat_input_folder": qlat_input_folder,
                "qlat_file_pattern_filter": file_pattern_filter,
                "nts": nts,
                "max_loop_size": max_loop_size,
            },
            "data_assimilation_parameters": {
                "streamflow_da": {
                    "streamflow_nudging": False,
                    "diffusive_streamflow_nudging": False,
                },
                "reservoir_da": {
                    "reservoir_persistence_da": {
                        "reservoir_persistence_usgs": False,
                    },
                    "reservoir_rfc_da": {
                        "reservoir_rfc_forecasts": False,
                    },
                },
            },
        },
        "output_parameters": {
            "stream_output": {
                "stream_output_directory": output_dir,
                "stream_output_time": -1,
                "stream_output_type": ".nc",
                "stream_output_internal_frequency": 60,
            },
        },
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

def build_forcing_dataset(cfg: ForcingConfig) -> None:
    """Build the forcing dataset and config YAML described by config."""
    if cfg.generate_reference_data and cfg.reference_dir is not None:
        cfg.reference_dir.mkdir(parents=True, exist_ok=True)

    if cfg.forcing_mode == "retro":
        create_forcing_dataset(
            t_start=cfg.start_dt,
            t_end=cfg.end_dt,
            forcing_dir=cfg.forcing_dir,
            hydrofabric_path=cfg.hf_path,
            retrospective_path=RETRO_PATH,
            generate_reference_data=cfg.generate_reference_data,
            reference_dir=cfg.reference_dir,
            runout_time=cfg.runout_time,
        )
    elif cfg.forcing_mode == "pulse":
        create_pulse_forcing_dataset(
            t_start=cfg.start_dt,
            t_end=cfg.end_dt,
            forcing_dir=cfg.forcing_dir,
            hydrofabric_path=cfg.hf_path,
            peak_qlat=cfg.peak_qlat,
        )
    elif cfg.forcing_mode == "constant":
        create_constant_forcing_dataset(
            t_start=cfg.start_dt,
            t_end=cfg.end_dt,
            forcing_dir=cfg.forcing_dir,
            hydrofabric_path=cfg.hf_path,
            constant_qlat=cfg.constant_qlat,
        )
    else:
        raise ValueError(f"Unknown forcing_mode '{cfg.forcing_mode}'. Choose retro, pulse, or constant.")

    make_config_yaml(
        config_path=cfg.config_path,
        hydrofabric_path=cfg.hf_path_relative,
        qlat_input_folder=cfg.qlat_input_folder,
        nts=cfg.nts,
        restart_time=cfg.restart_time,
        output_dir=cfg.output_dir,
    )


def main():
    """Enter via CLI."""
    parser = argparse.ArgumentParser(
        description="Generate forcing dataset and config YAML for a case."
    )

    parser.add_argument(
        "--start-time",
        default="2009-12-12 00:00",
        help="Simulation start time (e.g. '2009-12-12' or '2009-12-12 06:00')",
    )

    parser.add_argument(
        "--end-time",
        default="2009-12-29 00:00",
        help="Simulation end time (e.g. '2009-12-29' or '2009-12-29 12:00')",
    )

    parser.add_argument(
        "--case-id",
        default="conecuh_case",
        help="Case directory name",
    )

    parser.add_argument(
        "--hf-file",
        default="02374250.gpkg",
        help="Hydrofabric file inside domain directory",
    )

    parser.add_argument(
        "--run-id",
        default="retro",
        help="Run identifier.  There can be multiple runs per case.",
    )

    parser.add_argument(
        "--generate-reference-data",
        action="store_true",
        help="Generate reference data (USGS and retrospective outputs) for testing.",
    )

    parser.add_argument(
        "--no-runout-period",
        action="store_false",
        help="Add a runout period after the primary simulation window.",
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

    cfg = ForcingConfig(
        case_id=args.case_id,
        hf_file=args.hf_file,
        run_id=args.run_id,
        start_time=args.start_time,
        end_time=args.end_time,
        forcing_mode=args.forcing_mode,
        generate_reference_data=args.generate_reference_data,
        add_runout_period=args.no_runout_period,
        peak_qlat=args.peak_qlat,
        constant_qlat=args.constant_qlat,
    )

    build_forcing_dataset(cfg)


if __name__ == "__main__":
    main()
