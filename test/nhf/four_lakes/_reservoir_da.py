"""Reservoir DA end-to-end test for the four_lakes case.

Setup (data generation, config writing) lives in ``four_lakes.setup``.
This file contains only the t-route execution and result assertions.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr

from four_lakes.setup import (
    RESERVOIR_DN_FP,
    RESERVOIR_FLOW_VALUES,
    RunContext,
)


def run_troute(run_context: RunContext) -> None:
    """Execute the generated T-Route run."""
    subprocess.run(
        [sys.executable, "-m", "nwm_routing", "-V5", "-f", str(run_context.config_path)],
        cwd=run_context.model_root,
        check=True,
    )


def review_results(run_context: RunContext) -> None:
    """Assert each reservoir's downstream reach carries the expected constant outflow."""
    output_path  = run_context.model_root / run_context.output_dir
    output_files = sorted(output_path.glob("*.nc"))
    assert output_files, f"No output files found in {output_path}"

    output_ds   = xr.concat(
        [xr.open_dataset(p, engine="netcdf4") for p in output_files],
        dim="time",
    )
    feature_ids = set(output_ds["feature_id"].values.tolist())

    for lake_id, dn_fp_id in RESERVOIR_DN_FP.items():
        if dn_fp_id not in feature_ids:
            continue
        expected = RESERVOIR_FLOW_VALUES[lake_id]
        actual   = output_ds["flow"].sel(feature_id=dn_fp_id).values
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.05,
            atol=1.0,
            err_msg=(
                f"Flow downstream of reservoir {lake_id} (fp {dn_fp_id}) "
                f"should be ~{expected} m³/s"
            ),
        )


def test_reservoir_da() -> None:
    """Run t-route and assert DA outflows are correct (data must be pre-built by prep_tests.py)."""
    run_context = RunContext()
    run_troute(run_context)
    review_results(run_context)


if __name__ == "__main__":
    test_reservoir_da()
