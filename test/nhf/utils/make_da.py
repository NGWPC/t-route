"""Generate DA (data assimilation) forcing files for NHF test cases.

Supports four persistence DA types (USGS, USACE, USBR, Canada/WSC) written as
15-minute timeslice NetCDF files, and RFC forecast files written as hourly
per-station NetCDF files.

Usage examples
--------------
# Write USGS and USACE timeslices for station list:
python make_da.py --da-type usgs usace \\
    --station-ids 04127885 04159130 \\
    --start-time "2020-01-01 00:00" --end-time "2020-01-02 00:00" \\
    --output-dir reservoir_da/

# Write RFC files for a single station:
python make_da.py --da-type rfc \\
    --station-ids RFC000000000004 \\
    --start-time "2020-01-01 00:00" --end-time "2020-01-01 01:00" \\
    --output-dir reservoir_da/ --discharge 12345
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

# DA type → subdirectory name
DA_SUBDIR = {
    "usgs": "usgs_da",
    "usace": "usace_da",
    "usbr": "usbr_da",
    "canada": "canada_da",
    "rfc": "rfc_da",
}

# DA type → timeslice filename suffix
DA_SUFFIX = {
    "usgs": "usgsTimeSlice",
    "usace": "usaceTimeSlice",
    "usbr": "usbrTimeSlice",
    "canada": "wscTimeSlice",
}

DAType = Literal["usgs", "usace", "usbr", "canada", "rfc"]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DAConfig:
    """Parameters describing a single DA forcing generation run."""

    da_type: DAType
    """One of 'usgs', 'usace', 'usbr', 'canada', or 'rfc'."""

    station_ids: list[str]
    """List of station identifiers (padded to 15 chars inside the files)."""

    start_time: str
    """Simulation start time, e.g. '2020-01-01 00:00:00'."""

    end_time: str
    """Simulation end time, e.g. '2020-01-02 00:00:00'."""

    output_dir: Path = field(default_factory=lambda: Path("reservoir_da"))
    """Root directory under which per-type subdirs are created."""

    discharge: float = 1.0
    """Constant discharge value written to every observation (m³/s)."""

    discharge_quality: int = 100
    """Quality flag written alongside every discharge value."""

    # RFC-specific options
    rfc_lookback_hours: int = 28
    """Hours of observed lookback written into each RFC file."""

    rfc_forecast_hours: int = 12
    """Hours of synthetic forecast written into each RFC file."""

    rfc_timeslice_resolution_minutes: int = 60
    """Timeslice resolution reported in RFC file attributes (minutes)."""

    @property
    def start_dt(self) -> pd.Timestamp:
        return pd.to_datetime(self.start_time)

    @property
    def end_dt(self) -> pd.Timestamp:
        return pd.to_datetime(self.end_time)



# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")


def write_timeslice_da(cfg: DAConfig) -> None:
    """Write 15-minute timeslice NetCDF files for persistence DA types.

    Covers 'usgs', 'usace', 'usbr', and 'canada' (WSC) DA types.
    Files are written at 15-minute intervals spanning a one-hour buffer
    on each side of [start_time, end_time].
    """
    if cfg.da_type not in DA_SUFFIX:
        raise ValueError(
            f"write_timeslice_da does not handle da_type='{cfg.da_type}'. "
            "Use write_rfc_da for rfc."
        )

    suffix = DA_SUFFIX[cfg.da_type]
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(cfg.station_ids)
    station_ids_bytes = np.asarray(
        [sid.rjust(15) for sid in cfg.station_ids], dtype="S15"
    )
    discharge = np.full(n, cfg.discharge, dtype=np.float32)
    quality = np.full(n, cfg.discharge_quality, dtype=np.int16)

    encoding = {
        "stationId": {"dtype": "S15"},
        "time": {"dtype": "S19"},
        "discharge": {"dtype": "float32"},
        "discharge_quality": {"dtype": "int16"},
    }

    ts_range = pd.date_range(
        cfg.start_dt - pd.Timedelta(hours=1),
        cfg.end_dt + pd.Timedelta(hours=1),
        freq="15min",
    )

    now_str = _now_utc()
    for t in ts_range:
        time_str = t.strftime("%Y-%m-%d_%H:%M:%S")
        out_path = out_dir / f"{time_str}.15min.{suffix}.ncdf"
        ds = xr.Dataset(
            data_vars={
                "stationId": (["stationIdInd"], station_ids_bytes),
                "time": (
                    ["stationIdInd"],
                    np.asarray([time_str] * n, dtype="S19"),
                ),
                "discharge": (["stationIdInd"], discharge),
                "discharge_quality": (["stationIdInd"], quality),
            },
            attrs={
                "fileUpdateTimeUTC": now_str,
                "sliceCenterTimeUTC": time_str,
                "sliceTimeResolutionMinutes": "15",
            },
        )
        ds.to_netcdf(out_path, encoding=encoding)


def write_rfc_da(cfg: DAConfig) -> None:
    """Write hourly RFC forecast NetCDF files (one per hour per station).

    Files span a one-hour buffer on each side of [start_time, end_time].
    """
    if cfg.da_type != "rfc":
        raise ValueError(
            f"write_rfc_da requires da_type='rfc', got '{cfg.da_type}'."
        )

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lookback = cfg.rfc_lookback_hours
    forecast = cfg.rfc_forecast_hours
    n_steps = lookback + forecast + 1

    encoding = {
        "stationId": {"dtype": "S15"},
        "issueTimeUTC": {"dtype": "S19"},
        "discharges": {"dtype": "float32"},
        "synthetic_values": {"dtype": "int8"},
        "totalCounts": {"dtype": "int16"},
        "observedCounts": {"dtype": "int16"},
        "forecastCounts": {"dtype": "int16"},
        "discharge_qualities": {"dtype": "int16"},
    }

    hour_range = pd.date_range(
        cfg.start_dt - pd.Timedelta(hours=1),
        cfg.end_dt + pd.Timedelta(hours=1),
        freq="h",
    )

    now_str = _now_utc()
    for t in hour_range:
        time_str = t.strftime("%Y-%m-%d_%H:%M:%S")
        hour_str = t.strftime("%Y-%m-%d_%H")
        slice_start_str = (t - pd.Timedelta(hours=lookback)).strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        for sid in cfg.station_ids:
            sid_clean = sid.strip()
            out_path = (
                out_dir
                / f"{hour_str}.{cfg.rfc_timeslice_resolution_minutes}min.{sid_clean}.RFCTimeSeries.ncdf"
            )
            discharges = np.full((1, n_steps), cfg.discharge, dtype=np.float32)
            synthetic = np.zeros((1, n_steps), dtype=np.int8)
            synthetic[0, lookback + 1 :] = 1
            ds = xr.Dataset(
                data_vars={
                    "stationId": np.asarray(sid.rjust(15), dtype="S15"),
                    "issueTimeUTC": (
                        ["nseries"],
                        np.asarray([time_str], dtype="S19"),
                    ),
                    "discharges": (["nseries", "forecastInd"], discharges),
                    "synthetic_values": (["nseries", "forecastInd"], synthetic),
                    "totalCounts": (
                        ["nseries"],
                        np.asarray([n_steps], dtype=np.int16),
                    ),
                    "observedCounts": (
                        ["nseries"],
                        np.asarray([lookback + 1], dtype=np.int16),
                    ),
                    "forecastCounts": (
                        ["nseries"],
                        np.asarray([forecast], dtype=np.int16),
                    ),
                    "timeSteps": (
                        ["nseries"],
                        np.asarray(
                            [np.timedelta64(1, "h")], dtype="timedelta64[ns]"
                        ),
                    ),
                    "discharge_qualities": (
                        ["nseries"],
                        np.asarray([cfg.discharge_quality], dtype=np.int16),
                    ),
                    "queryTime": (
                        ["nseries"],
                        np.asarray(
                            [np.datetime64(t.to_pydatetime(), "ns")],
                            dtype="datetime64[ns]",
                        ),
                    ),
                },
                attrs={
                    "fileUpdateTimeUTC": now_str,
                    "sliceStartTimeUTC": slice_start_str,
                    "sliceTimeResolutionMinutes": str(
                        cfg.rfc_timeslice_resolution_minutes
                    ),
                    "missingValue": "-999",
                    "newest_forecast": "0",
                    "NWM_version_number": "v3.0",
                },
            )
            ds.to_netcdf(out_path, encoding=encoding)


def build_da_dataset(cfg: DAConfig) -> None:
    """Dispatch to the appropriate writer based on cfg.da_type."""
    if cfg.da_type == "rfc":
        write_rfc_da(cfg)
    else:
        write_timeslice_da(cfg)


def write_lake_ontario_outflow(
    out_path: Path,
    start_time: str,
    end_time: str,
    outflow: float = 2000.0,
) -> None:
    """Write a Lake Ontario outflow CSV spanning [start_time, end_time]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start_time, end_time, freq="h")
    pd.DataFrame({
        "Date": times.strftime("%Y-%m-%d"),
        "Hour": times.strftime("%H:00"),
        "Outflow(m3/s)": np.repeat(outflow, len(times)),
    }).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DA forcing files for an NHF test case."
    )
    parser.add_argument(
        "--da-type",
        required=True,
        choices=list(DA_SUBDIR),
        help="DA type to generate: usgs, usace, usbr, canada, or rfc.",
    )
    parser.add_argument(
        "--station-ids",
        nargs="+",
        required=True,
        metavar="STATION_ID",
        help="One or more station identifiers.",
    )
    parser.add_argument(
        "--start-time",
        required=True,
        help="Simulation start time (e.g. '2020-01-01 00:00:00').",
    )
    parser.add_argument(
        "--end-time",
        required=True,
        help="Simulation end time (e.g. '2020-01-02 00:00:00').",
    )
    parser.add_argument(
        "--output-dir",
        default="reservoir_da",
        help="Root output directory (default: reservoir_da/).",
    )
    parser.add_argument(
        "--discharge",
        type=float,
        default=1.0,
        help="Constant discharge value (m³/s) written to every observation.",
    )
    parser.add_argument(
        "--discharge-quality",
        type=int,
        default=100,
        help="Quality flag written alongside every discharge value (default: 100).",
    )
    parser.add_argument(
        "--rfc-lookback-hours",
        type=int,
        default=28,
        help="Hours of observed lookback in RFC files (default: 28).",
    )
    parser.add_argument(
        "--rfc-forecast-hours",
        type=int,
        default=12,
        help="Hours of synthetic forecast in RFC files (default: 12).",
    )
    args = parser.parse_args()

    cfg = DAConfig(
        da_type=args.da_type,
        station_ids=args.station_ids,
        start_time=args.start_time,
        end_time=args.end_time,
        output_dir=Path(args.output_dir),
        discharge=args.discharge,
        discharge_quality=args.discharge_quality,
        rfc_lookback_hours=args.rfc_lookback_hours,
        rfc_forecast_hours=args.rfc_forecast_hours,
    )

    build_da_dataset(cfg)


if __name__ == "__main__":
    main()
