import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from test.nhf.utils.make_configs import Config


def run_troute(config_path: Path) -> None:
    """Run t-route from config_path's parent directory, raising on failure."""
    subprocess.run(
        [sys.executable, "-m", "nwm_routing", "-V5", "-f", config_path.name],
        cwd=config_path.parent,
        check=True,
    )


def has_files(directory: Path, pattern: str = "*") -> bool:
    """Return True if *directory* exists and contains at least one matching file."""
    return directory.is_dir() and any(directory.glob(pattern))


def delete_outputs(output_dir: Path) -> None:
    """Remove all .nc files from *output_dir* so stale results cannot mask failures."""
    if output_dir.is_dir():
        for nc_file in output_dir.glob("*.nc"):
            nc_file.unlink()


def skip_if_not_built(cfg: Config) -> None:
    """Skip the calling test if the case data has not been prepped yet."""
    passing = True
    if not cfg.config_path.exists():
        message = f"{cfg.config_path} does not exist"
        passing = False
    elif not has_files(cfg.channel_forcing_dir, cfg.qlat_file_pattern):
        message = f"missing forcing files in {cfg.channel_forcing_dir}"
        passing = False
    elif not cfg.domain_path.exists():
        message = f"{cfg.domain_path} does not exist"
        passing = False
    if not passing:
        pytest.skip(
            message
            + "run: python -m test.nhf._prep_tests --nhf-gpkg <path> --test `case_name`"
        )


def load_output(output_dir: Path) -> xr.Dataset:
    """Load and concatenate all .nc files in output_dir into a single Dataset."""
    nc_files = sorted(output_dir.glob("*.nc"))
    if not nc_files:
        pytest.fail(f"no output .nc files found in {output_dir}")
    return xr.concat(
        [xr.open_dataset(p, engine="netcdf4") for p in nc_files], dim="time"
    )


def assert_peak_bounds(
    output_dir: Path, peak_bounds: dict[int, tuple[float, float]]
) -> None:
    """Assert that peak flow for each feature_id falls within (lo, hi)."""
    ds = load_output(output_dir)

    violations = []
    for feature_id, (lo, hi) in peak_bounds.items():
        flows = ds["flow"].sel(feature_id=feature_id).values
        peak = float(np.nanmax(flows))
        if not (lo <= peak <= hi):
            violations.append(
                f"feature_id {feature_id}: observed peak {peak:.3f} outside [{lo}, {hi}]"
            )
    if violations:
        pytest.fail("Peak flow bounds violated:\n" + "\n".join(violations))
