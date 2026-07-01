"""End-to-end NHF integration tests.

Each test runs t-route against pre-built forcing/domain data and checks the
result.  Tests are marked ``integration`` so they can be run selectively:

    pytest -m integration          # run only these tests
    pytest -m "not integration"    # skip these tests

All tests perform a pre-flight data check and skip with an informational
message if the required files have not been built yet.  Run ``prep_tests.py``
first to generate the necessary inputs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from four_lakes.setup import RunContext as FourLakesRunContext
from four_lakes._reservoir_da import review_results, run_troute

HERE = Path(__file__).resolve().parent
RUN_ID = "test_run"
CONFIG = f"{RUN_ID}.yaml"
FORCING_DIR = f"channel_forcing_{RUN_ID}"

def _run_troute(config_path: Path) -> None:
    """Run t-route from config_path's parent directory, raising on failure."""
    subprocess.run(
        [sys.executable, "-m", "nwm_routing", "-V5", "-f", config_path.name],
        cwd=config_path.parent,
        check=True,
    )


def _has_files(directory: Path, pattern: str = "*") -> bool:
    """Return True if *directory* exists and contains at least one matching file."""
    return directory.is_dir() and any(directory.glob(pattern))


@pytest.mark.integration
def test_conecuh() -> None:
    """Route the December 2009 Conecuh River flood event."""
    case_dir    = HERE / "conecuh_case"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / FORCING_DIR

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "conecuh_case data not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test conecuh"
        )

    _run_troute(config_path)


@pytest.mark.integration
def test_patuxent() -> None:
    """Route the September 2011 Patuxent Reservoir event."""
    case_dir    = HERE / "patuxent"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / FORCING_DIR

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "patuxent data not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test patuxent"
        )

    _run_troute(config_path)


@pytest.mark.integration
def test_lake_creek() -> None:
    """Route the March 1987 Lake Creek event (two lakes on the same flowpath)."""
    case_dir    = HERE / "lake_creek"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / FORCING_DIR

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "lake_creek data not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test lake_creek"
        )

    _run_troute(config_path)


@pytest.mark.integration
def test_ciss_creek() -> None:
    """Route a synthetic pulse through Ciss Creek and regenerate the diagnostic plot."""
    case_dir    = HERE / "ciss_creek"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / FORCING_DIR

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "ciss_creek data not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test ciss_creek"
        )

    _run_troute(config_path)

    # TODO: Add specific test


@pytest.mark.integration
def test_hot_brook() -> None:
    """Route a synthetic pulse through Hot Brook (two lakes, same flowpath).

    After routing, ``review.py`` is run to regenerate the diagnostic plot.
    That script has no assertions — failures surface as non-zero exit codes
    or exceptions raised during plot computation.
    """
    case_dir    = HERE / "hot_brook"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / "channel_forcing"

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "hot_brook forcing data not built. "
            "Run: python test/nhf/prep_tests.py --test hot_brook"
        )

    _run_troute(config_path)

    subprocess.run(
        [sys.executable, "review.py"],
        cwd=case_dir,
        check=True,
    )



@pytest.mark.integration
def test_great_lakes() -> None:
    """Force Great Lakes outflows via DA and verify they propagate downstream."""
    case_dir   = HERE / "great_lakes"
    domain_gpkg = case_dir / "domain" / "nhf.gpkg"
    config_path = case_dir / "no_inflow.yaml"

    if not domain_gpkg.exists():
        pytest.skip(
            "great_lakes domain not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test great_lakes"
        )

    if not config_path.exists():
        pytest.skip(
            f"great_lakes config not found: {config_path}. "
            "Ensure the case directory is complete."
        )

    # run_test.py writes its own forcing, runs t-route, and asserts results.
    subprocess.run(
        [sys.executable, "run_test.py"],
        cwd=case_dir,
        check=True,
    )

@pytest.mark.integration
def test_four_lakes_reservoir_da() -> None:
    """Verify all four reservoir DA types (USGS, USACE, RFC, USBR) route correctly."""
    rc = FourLakesRunContext()

    missing: list[str] = []
    if not rc.hf_path.exists():
        missing.append(f"domain gpkg ({rc.hf_path})")
    if not _has_files(rc.forcing_dir, "*.csv"):
        missing.append(f"channel forcing ({rc.forcing_dir})")
    if not _has_files(rc.da_dir, "**/*.ncdf"):
        missing.append(f"DA forcing ({rc.da_dir})")

    if missing:
        pytest.skip(
            "four_lakes data not built — missing: "
            + ", ".join(missing)
            + ".  Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test four_lakes"
        )

    run_troute(rc)
    review_results(rc)
