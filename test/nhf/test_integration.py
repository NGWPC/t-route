"""End-to-end NHF integration tests.

Each test runs t-route against pre-built forcing/domain data and checks the
result.  Tests are marked ``integration`` so they can be run selectively:

    pytest -m integration          # run only these tests
    pytest -m "not integration"    # skip these tests

All tests perform a pre-flight data check and skip with an informational
message if the required files have not been built yet.  Run ``prep_tests.py``
first to generate the necessary inputs.

Test configurations (reach IDs, acceptable peak bounds) are centralised in
the ``TEST CONFIGURATIONS`` section below — that is the only place you need
to edit when updating IDs or thresholds.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

matplotlib.use("Agg")

from four_lakes.setup import (
    RESERVOIR_DN_FP,
    RESERVOIR_FLOW_VALUES,
    RESERVOIR_TYPE_MOD,
    RunContext as FourLakesRunContext,
)
from four_lakes._reservoir_da import review_results, run_troute

HERE = Path(__file__).resolve().parent
RUN_ID = "test_run"
CONFIG = f"{RUN_ID}.yaml"
FORCING_DIR = f"channel_forcing_{RUN_ID}"


### CLASSES ###


@dataclass
class ReachTrace:
    """One plotted line within a diagnostic panel.

    If ``peak_min`` and ``peak_max`` are both set this trace is:
      - asserted to have a peak within ``[peak_min, peak_max]`` after routing
      - shown with a highlighted acceptance band on the diagnostic plot

    Parameters
    ----------
    label:
        Legend entry for this line.
    feature_id:
        NHF fp_id (or waterbody id) whose ``flow`` variable is extracted.
    peak_min / peak_max:
        Inclusive acceptable peak-flow bounds in m³/s.  ``None`` disables
        both the assertion and the band.
    color:
        Matplotlib colour string.  ``None`` uses the auto-cycle palette.

    """

    label: str
    feature_id: int
    peak_min: Optional[float] = None
    peak_max: Optional[float] = None
    color: Optional[str] = None


@dataclass
class PlotSpec:
    """One subplot panel: a title, an output directory, and one or more traces.

    All traces in a panel share the same output directory (they come from the
    same t-route run).
    """

    title: str
    output_dir: Path
    traces: list[ReachTrace] = field(default_factory=list)


@dataclass
class TraceData:
    """Loaded time-series data for one ``ReachTrace``."""

    trace: ReachTrace
    times: list
    flows: list


@dataclass
class PlotData:
    """All loaded data for one ``PlotSpec`` panel, ready for rendering."""

    spec: PlotSpec
    traces: list[TraceData]


_PLOT_QUEUE: list[PlotData] = []


### TEST CONFIGS ###

CONECUH_PLOTS: list[PlotSpec] = [
    PlotSpec(
        title="Conecuh River (Dec 2009)",
        output_dir=HERE / "conecuh_case" / f"output_{RUN_ID}",
        traces=[
            ReachTrace(
                label="Outlet",
                feature_id=1270581653591645,
                peak_min=1600,
                peak_max=2300,
            ),
        ],
    ),
]

PATUXENT_PLOTS: list[PlotSpec] = [
    PlotSpec(
        title="Patuxent Reservoir (Sep 2011)",
        output_dir=HERE / "patuxent" / f"output_{RUN_ID}",
        traces=[
            ReachTrace(
                label="Reservoir inflow",
                feature_id=1284687464436834,
                peak_min=15,
                peak_max=20,
                color="steelblue",
            ),
            ReachTrace(
                label="Reservoir outflow",
                feature_id=1284687521058505,
                peak_min=0.2,
                peak_max=0.7,
                color="darkorange",
            ),
        ],
    ),
]

# ciss_creek: outlet reach not yet determined; add feature_id + bounds once known.
CISS_CREEK_PLOTS: list[PlotSpec] = [
    PlotSpec(
        title="Ciss Creek (synthetic pulse)",
        output_dir=HERE / "ciss_creek" / f"output_{RUN_ID}",
        traces=[
            ReachTrace(
                label="Outlet",
                feature_id=1288454913281725,
                peak_min=175,
                peak_max=190,
            ),
        ],
    ),
]

GREAT_LAKES_PLOTS: list[PlotSpec] = [
    PlotSpec(
        title="Great Lakes (DA-forced, Superior outlet)",
        output_dir=HERE / "great_lakes" / "output",
        traces=[
            ReachTrace(
                label="Superior downstream",
                feature_id=1278348162056612,
                color="steelblue",
            ),
            ReachTrace(
                label="Huron-Michigan downstream",
                feature_id=1276364270499315,
                color="seagreen",
            ),
            ReachTrace(
                label="Erie downstream",
                feature_id=1286192735893685,
                color="darkorange",
            ),
            ReachTrace(
                label="Ontario downstream",
                feature_id=1287248237297035,
                color="mediumpurple",
            ),
        ],
    ),
]

# Four-lakes plots are built at test time because the output directory comes
# from FourLakesRunContext.  See _build_four_lakes_plots().
_FOUR_LAKES_DA_TYPE_NAMES: dict[int, str] = {
    2: "USGS",
    3: "USACE",
    4: "RFC",
    7: "USBR",
}


def _build_four_lakes_plots(output_dir: Path) -> list[PlotSpec]:
    """Return one PlotSpec per reservoir DA type, derived from setup constants."""
    plots: list[PlotSpec] = []
    for lake_id, da_type in RESERVOIR_TYPE_MOD.items():
        dn_fp   = RESERVOIR_DN_FP[lake_id]
        da_flow = RESERVOIR_FLOW_VALUES[lake_id]
        da_name = _FOUR_LAKES_DA_TYPE_NAMES.get(da_type, f"type-{da_type}")
        plots.append(
            PlotSpec(
                title=f"Four Lakes — {da_name} (DA type {da_type})",
                output_dir=output_dir,
                traces=[
                    ReachTrace(
                        label=f"DA-forced reach (target {da_flow:.4g} m\u00b3/s)",
                        feature_id=dn_fp,
                        peak_min=da_flow * 0.95,
                        peak_max=da_flow * 1.05,
                    ),
                ],
            )
        )
    return plots


### HELPERS ###


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


def _delete_outputs(output_dir: Path) -> None:
    """Remove all .nc files from *output_dir* so stale results cannot mask failures."""
    if output_dir.is_dir():
        for nc_file in output_dir.glob("*.nc"):
            nc_file.unlink()


def _load_plot_data(spec: PlotSpec) -> Optional[PlotData]:
    """Load flow time-series for every trace in *spec* and return a PlotData.

    Returns ``None`` when the output directory is missing or unreadable.  Any
    individual trace whose ``feature_id`` is absent from the dataset is skipped
    silently (the run still produced output for the other traces).
    """
    nc_files = sorted(spec.output_dir.glob("*.nc"))
    if not nc_files:
        return None

    try:
        ds = xr.concat(
            [xr.open_dataset(p, engine="netcdf4") for p in nc_files],
            dim="time",
        )
    except Exception:
        return None

    trace_data: list[TraceData] = []
    for trace in spec.traces:
        try:
            flow_da = ds["flow"].sel(feature_id=trace.feature_id)
            trace_data.append(
                TraceData(
                    trace=trace,
                    times=flow_da["time"].values.tolist(),
                    flows=flow_da.values.tolist(),
                )
            )
        except KeyError:
            pass  # feature_id absent from this run's output — skip

    return PlotData(spec=spec, traces=trace_data) if trace_data else None


def _assert_peak_bounds(specs: list[PlotSpec], output_dir: Path) -> None:
    """Fail the calling test if any bounded trace's peak is outside its range.

    All violations are collected before raising so the full picture is reported
    in one assertion error rather than stopping at the first failure.
    """
    bounded = [
        (spec, trace)
        for spec in specs
        for trace in spec.traces
        if trace.peak_min is not None and trace.peak_max is not None
    ]
    if not bounded:
        return

    nc_files = sorted(output_dir.glob("*.nc"))
    if not nc_files:
        pytest.fail(f"No output .nc files found in {output_dir}")

    ds = xr.concat(
        [xr.open_dataset(p, engine="netcdf4") for p in nc_files],
        dim="time",
    )

    violations: list[str] = []
    for spec, trace in bounded:
        try:
            flows = ds["flow"].sel(feature_id=trace.feature_id).values
        except KeyError:
            violations.append(
                f"  [{spec.title} / {trace.label}] "
                f"feature_id {trace.feature_id} not found in output"
            )
            continue

        peak = float(np.nanmax(flows))
        if not (trace.peak_min <= peak <= trace.peak_max):
            violations.append(
                f"  [{spec.title} / {trace.label}] "
                f"peak {peak:.3f} m\u00b3/s outside acceptable range "
                f"[{trace.peak_min:.3f}, {trace.peak_max:.3f}]"
            )

    if violations:
        pytest.fail("Peak flow bounds violated:\n" + "\n".join(violations))


### COMBO PLOT FOR VISUAL REASSURANCE THAT RUNS ARE GOOD ###

_TRACE_COLORS = [
    "steelblue", "darkorange", "seagreen", "crimson",
    "mediumpurple", "saddlebrown", "teal", "deeppink",
]


@pytest.fixture(scope="session", autouse=True)
def _session_diagnostic_plot():
    """Generate ``diagnostic_hydrographs.png`` after all integration tests complete."""
    yield  # let all tests run first

    panels = [p for p in _PLOT_QUEUE if p is not None and p.traces]
    if not panels:
        return

    n     = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        "NHF Integration Test — Diagnostic Hydrographs",
        fontsize=14, fontweight="bold",
    )

    for idx, panel in enumerate(panels):
        ax   = axes[idx // ncols][idx % ncols]
        spec = panel.spec

        for tidx, td in enumerate(panel.traces):
            color = td.trace.color or _TRACE_COLORS[tidx % len(_TRACE_COLORS)]
            times = pd.to_datetime(td.times)
            flows = np.asarray(td.flows, dtype=float)
            peak  = float(flows.max()) if len(flows) else float("nan")

            ax.plot(
                times, flows,
                color=color, linewidth=1.5,
                label=f"{td.trace.label} (peak {peak:.1f} m\u00b3/s)",
            )

            if td.trace.peak_min is not None and td.trace.peak_max is not None:
                ax.axhspan(
                    td.trace.peak_min, td.trace.peak_max,
                    alpha=0.18, color=color,
                )
                ax.axhline(td.trace.peak_min, color=color, linewidth=0.8, linestyle="--")
                ax.axhline(td.trace.peak_max, color=color, linewidth=0.8, linestyle="--")

        ax.set_title(spec.title, fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flow (m\u00b3/s)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30, ha="right")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    out_path = HERE / "diagnostic_hydrographs.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDiagnostic plot saved \u2192 {out_path}")


### TESTS ###

@pytest.mark.integration
def test_conecuh() -> None:
    """Route the December 2009 Conecuh River flood event."""
    case_dir    = HERE / "conecuh_case"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / FORCING_DIR
    output_dir  = CONECUH_PLOTS[0].output_dir

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "conecuh_case data not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test conecuh"
        )

    _delete_outputs(output_dir)
    _run_troute(config_path)
    _assert_peak_bounds(CONECUH_PLOTS, output_dir)
    for spec in CONECUH_PLOTS:
        _PLOT_QUEUE.append(_load_plot_data(spec))


@pytest.mark.integration
def test_patuxent() -> None:
    """Route the September 2011 Patuxent Reservoir event."""
    case_dir    = HERE / "patuxent"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / FORCING_DIR
    output_dir  = PATUXENT_PLOTS[0].output_dir

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "patuxent data not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test patuxent"
        )

    _delete_outputs(output_dir)
    _run_troute(config_path)
    _assert_peak_bounds(PATUXENT_PLOTS, output_dir)
    for spec in PATUXENT_PLOTS:
        _PLOT_QUEUE.append(_load_plot_data(spec))

@pytest.mark.integration
def test_ciss_creek() -> None:
    """Route a synthetic pulse through Ciss Creek."""
    case_dir    = HERE / "ciss_creek"
    config_path = case_dir / CONFIG
    forcing_dir = case_dir / FORCING_DIR

    if not config_path.exists() or not _has_files(forcing_dir, "*.csv"):
        pytest.skip(
            "ciss_creek data not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test ciss_creek"
        )

    # Derive output_dir from config rather than CISS_CREEK_PLOTS so the test
    # can run even when no plot spec has been configured yet.
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    output_dir = case_dir / cfg["output_parameters"]["stream_output"]["stream_output_directory"]

    _delete_outputs(output_dir)
    _run_troute(config_path)
    _assert_peak_bounds(CISS_CREEK_PLOTS, output_dir)
    for spec in CISS_CREEK_PLOTS:
        _PLOT_QUEUE.append(_load_plot_data(spec))

    # TODO: add outlet reach ID + bounds to CISS_CREEK_PLOTS above


@pytest.mark.integration
def test_great_lakes() -> None:
    """Force Great Lakes outflows via DA and verify they propagate downstream."""
    case_dir    = HERE / "great_lakes"
    domain_gpkg = case_dir / "domain" / "nhf.gpkg"
    output_dir  = GREAT_LAKES_PLOTS[0].output_dir

    if not domain_gpkg.exists():
        pytest.skip(
            "great_lakes domain not built. "
            "Run: python test/nhf/prep_tests.py --nhf-gpkg <path> --test great_lakes"
        )

    _delete_outputs(output_dir)

    # run_test.py writes its own forcing, runs t-route, and asserts results.
    subprocess.run(
        [sys.executable, "run_test.py"],
        cwd=case_dir,
        check=True,
    )

    for spec in GREAT_LAKES_PLOTS:
        _PLOT_QUEUE.append(_load_plot_data(spec))


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

    output_dir       = rc.model_root / rc.output_dir
    four_lakes_plots = _build_four_lakes_plots(output_dir)

    _delete_outputs(output_dir)
    run_troute(rc)
    review_results(rc)

    _assert_peak_bounds(four_lakes_plots, output_dir)
    for spec in four_lakes_plots:
        _PLOT_QUEUE.append(_load_plot_data(spec))
