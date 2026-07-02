#!/usr/bin/env python3
"""prep_tests.py — Run preprocessing for NHF tests without running the tests.

Usage
-----
    python prep_tests.py [OPTIONS]

Options
-------
    --nhf-gpkg PATH       Path to your NHF geopackage (required for tests that
                          need it: conecuh, patuxent, lake_creek, great_lakes)
    --test NAME [NAME ...] One or more test names to prep.  If omitted, all
                          tests are prepped.
                          Choices: conecuh patuxent lake_creek great_lakes
                                   ciss_creek hot_brook
    --refresh             Force regeneration even if output already exists
    -h / --help           Show this help message

Tests that need no preprocessing (data is committed):
    four_lakes, richelieu

Examples
--------
    # Prep all tests
    python prep_tests.py --nhf-gpkg /data/nhf.gpkg

    # Prep only conecuh and force-refresh
    python prep_tests.py --nhf-gpkg /data/nhf.gpkg --test conecuh --refresh

    # Prep only tests that don't need a gpkg
    python prep_tests.py --test ciss_creek hot_brook

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_forcing import ForcingConfig, build_forcing_dataset
from four_lakes.setup import (
    RunContext as FourLakesRunContext,
    make_channel_forcing_data,
    make_config as make_four_lakes_config,
    make_reservoir_da_data,
    modify_lakes_table,
)
from typing import Literal


NHF_GPKG_DEFAULT = "/hydrofabric/nhf_1.2.1.gpkg"


@dataclass
class TestConfig:
    """Configuration for an NHF preprocessing test."""

    case_id: str
    outlet_fp_id: int | None = None  # None = no NHF subset step needed
    start_time: str = ""
    end_time: str = ""
    hf_file: str = "nhf.gpkg"
    run_id: str = "test_run"
    forcing_mode: Literal["retro", "pulse", "constant"] = "retro"
    peak_qlat: float = 10_000.0
    constant_qlat: float = 1.0


@dataclass
class GreatLakesConfig:
    """Configuration for the great_lakes test (uses build_domain.py)."""

    case_id: str = "great_lakes"
    hf_file: str = "nhf.gpkg"


### TESTS ###

CONECUH = TestConfig(
    case_id="conecuh_case",
    outlet_fp_id=1270581653591645,
    start_time="2009-12-12 00:00",
    end_time="2009-12-29 00:00",
)

PATUXENT = TestConfig(
    case_id="patuxent",
    outlet_fp_id=1284196257037837,
    start_time="2011-09-05 00:00",
    end_time="2011-09-15 00:00",
)

CISS_CREEK = TestConfig(
    case_id="ciss_creek",
    outlet_fp_id=1288454913281725,
    start_time="2000-01-01 00:00",
    end_time="2000-01-03 00:00",
)

GREAT_LAKES = GreatLakesConfig()

FOUR_LAKES_OUTLET_FP_ID = 1276182780176988

ALL_TESTS = [
    "conecuh",
    "patuxent",
    "lake_creek",
    "great_lakes",
    "ciss_creek",
    "hot_brook",
    "four_lakes",
]

### HELPERS ###

HERE = Path(__file__).resolve().parent


def info(msg: str) -> None:
    """Print a prefixed info message to stdout."""
    print(f"[INFO]  {msg}", flush=True)


def run(cmd: list[str | Path]) -> None:
    """Log and execute a subprocess command, raising on non-zero exit."""
    info("Running: " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def forcing_is_empty(path: Path) -> bool:
    """Return True if the forcing directory is absent or contains no files."""
    return not path.exists() or not any(path.iterdir())


### SPECIFIC TEST BUILDERS ###


def prep_test(cfg: TestConfig, nhf_gpkg: str, refresh: bool) -> None:
    """Subset the NHF domain and generate forcing for a standard test case."""
    case_dir = HERE / cfg.case_id
    domain_gpkg = case_dir / "domain" / cfg.hf_file
    forcing_dir = case_dir / f"channel_forcing_{cfg.run_id}"

    info(f"=== {cfg.case_id} ===")

    if refresh or not domain_gpkg.exists():
        info(f"Subsetting NHF for {cfg.case_id} outlet fp_id={cfg.outlet_fp_id} ...")
        domain_gpkg.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                HERE / "subset_nhf.py",
                "--source-gpkg",
                nhf_gpkg,
                "--out-gpkg",
                domain_gpkg,
                "--outlet-fp-id",
                cfg.outlet_fp_id,
            ]
        )
    else:
        info("Domain gpkg already exists; skipping subset (use --refresh to redo).")

    if refresh or forcing_is_empty(forcing_dir):
        info(
            f"Generating forcing for {cfg.case_id} ({cfg.start_time} to {cfg.end_time}) ..."
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        build_forcing_dataset(
            ForcingConfig(
                case_id=cfg.case_id,
                hf_file=cfg.hf_file,
                run_id=cfg.run_id,
                start_time=cfg.start_time,
                end_time=cfg.end_time,
                forcing_mode=cfg.forcing_mode,
                peak_qlat=cfg.peak_qlat,
                constant_qlat=cfg.constant_qlat,
            )
        )
    else:
        info("Forcing already exists; skipping (use --refresh to redo).")


def prep_great_lakes(cfg: GreatLakesConfig, nhf_gpkg: str, refresh: bool) -> None:
    """Build the Great Lakes domain geopackage via build_domain.py."""
    case_dir = HERE / cfg.case_id
    domain_gpkg = case_dir / "domain" / cfg.hf_file

    info(f"=== {cfg.case_id} ===")

    if refresh or not domain_gpkg.exists():
        info(f"Building Great Lakes domain from {nhf_gpkg} ...")
        domain_gpkg.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                case_dir / "build_domain.py",
                "--source-gpkg",
                nhf_gpkg,
                "--out-gpkg",
                domain_gpkg,
            ]
        )
    else:
        info("Domain gpkg already exists; skipping (use --refresh to redo).")


def prep_four_lakes(nhf_gpkg: str, refresh: bool) -> None:
    """Prep the four_lakes reservoir DA test."""
    rc = FourLakesRunContext()
    info("=== four_lakes ===")

    if refresh:
        import shutil

        for p in [rc.forcing_dir, rc.da_dir]:
            shutil.rmtree(p, ignore_errors=True)

    if refresh or not rc.hf_path.exists():
        info(
            f"Subsetting NHF for four_lakes outlet fp_id={FOUR_LAKES_OUTLET_FP_ID} ..."
        )
        rc.hf_path.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                HERE / "subset_nhf.py",
                "--source-gpkg",
                nhf_gpkg,
                "--out-gpkg",
                rc.hf_path,
                "--outlet-fp-id",
                FOUR_LAKES_OUTLET_FP_ID,
            ]
        )
    else:
        info("Domain gpkg already exists; skipping subset (use --refresh to redo).")

    modify_lakes_table(rc)
    if refresh or not any(rc.forcing_dir.glob("*.csv")):
        make_channel_forcing_data(rc)
    if refresh or not any(rc.da_dir.rglob("*.ncdf")):
        make_reservoir_da_data(rc)
    make_four_lakes_config(rc)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nhf-gpkg",
        default=NHF_GPKG_DEFAULT,
        help="Path to your NHF geopackage.",
    )
    parser.add_argument(
        "--test",
        nargs="+",
        choices=ALL_TESTS,
        metavar="NAME",
        help=f"Tests to prep (default: all).  Choices: {', '.join(ALL_TESTS)}",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force regeneration even if output already exists.",
    )
    args = parser.parse_args()

    tests = set(args.test) if args.test else set(ALL_TESTS)

    # Validate gpkg for tests that need it
    gpkg_path = Path(args.nhf_gpkg)
    if not gpkg_path.exists():
        parser.error(f"--nhf-gpkg '{args.nhf_gpkg}' does not exist. ")

    if "conecuh" in tests:
        prep_test(CONECUH, args.nhf_gpkg, args.refresh)
    if "patuxent" in tests:
        prep_test(PATUXENT, args.nhf_gpkg, args.refresh)
    if "ciss_creek" in tests:
        prep_test(CISS_CREEK, args.nhf_gpkg, args.refresh)
    if "great_lakes" in tests:
        prep_great_lakes(GREAT_LAKES, args.nhf_gpkg, args.refresh)
    if "four_lakes" in tests:
        prep_four_lakes(args.nhf_gpkg, args.refresh)

    info("=== Preprocessing complete ===")


if __name__ == "__main__":
    main()
