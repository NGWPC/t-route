import importlib
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from matplotlib import pyplot as plt

FORTRAN_SRC = (Path(__file__).resolve().parents[1] / "src/kernel/muskingum/MCsingleSegStime_f2py_NOLOOP.f90")
FORTRAN_MODULE_NAME = "mc_fortran_port"

SAMPLE_RANDOM_SEED = 11
random.seed(SAMPLE_RANDOM_SEED)
CONFIG_FILES = [
    # "conecuh_case/test_case.yaml",
    "conus_working/test_case.yaml",
    ]

### DEFINE FIXTURES ###

class RunContext:
    """Relevant data for tests of a t-route run.

    This context bundles model outputs, forcings, and network mappings needed
    to compute reach-scale and network-integrated routing performance diagnostics.
    """

    def __init__(self, config_key: str):
        self.cdir = Path(__file__).parent
        self.idx = config_key.split("/")[0]
        self.config_path = self.cdir / config_key

        # Load yaml
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    @cached_property
    def max_stream_order(self) -> int:
        """Maximum stream order in the domain."""
        return self.flowpaths_gdf["stream_order"].max()

    @cached_property
    def dt(self) -> float:
        """Timestep of t-route run in seconds."""
        return self.config["compute_parameters"]["forcing_parameters"]["dt"]

    @cached_property
    def nts(self) -> float:
        """Number of timesteps in the run."""
        return self.config["compute_parameters"]["forcing_parameters"]["nts"]

    @cached_property
    def qts_subdivisions(self) -> int:
        """Number of times each timestep is subdivided for routing."""
        # Note that this follows the logic in AbstractNetwork.py (line 909).
        # If that is ever updated to read YAML data, switch to
        # return self.config["compute_parameters"]["forcing_parameters"]["qts_subdivisions"]
        dt_qlat_timedelta = self.forcing_data.index[1] - self.forcing_data.index[0]
        dt_qlat = dt_qlat_timedelta.seconds
        return dt_qlat / self.dt


    @cached_property
    def routed_results(self) -> xr.Dataset:
        """Load results from t-route run."""
        output_path = self.config_path.parent / self.config["output_parameters"]["stream_output"]["stream_output_directory"]
        paths = sorted(output_path.glob("*.nc"))
        flows = [xr.open_dataset(p, engine="netcdf4") for p in paths]
        return xr.concat(flows, dim="time")

    @cached_property
    def forcing_data(self) -> pd.DataFrame:
        """Load channel forcing data that was used to run t-route."""
        forcing_dir = self.config_path.parent / self.config["compute_parameters"]["forcing_parameters"]["qlat_input_folder"]
        forcing_files = sorted(forcing_dir.glob("*.csv"))
        t0 = datetime.strptime(forcing_files[0].stem.split(".")[0], "%Y%m%d%H%M")
        tmax = t0 + pd.to_timedelta(self.nts * self.dt, unit="s")
        forcing_files = [f for f in forcing_files if datetime.strptime(f.stem.split(".")[0], "%Y%m%d%H%M") < tmax]
        forcing = pd.concat([pd.read_csv(f).sort_values("feature_id") for f in forcing_files], axis=1)
        columns = forcing["feature_id"].values[:, 0]
        forcing = forcing.drop(columns=["feature_id"]).T
        forcing.columns = columns
        forcing.index = pd.to_datetime(forcing.index)
        # Final trim
        forcing = forcing.loc[(forcing.index >= t0) & (forcing.index < tmax)]
        return forcing

    @cached_property
    def hydrofabric_path(self) -> Path:
        """Path to hydrofabric gpkg."""
        return self.config_path.parent / self.config["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"]

    @cached_property
    def virtual_flowpaths_gdf(self) -> gpd.GeoDataFrame:
        """Load virtual flowpaths layer."""
        return gpd.read_file(self.hydrofabric_path, layer="virtual_flowpaths", ignore_geometry=True)

    @cached_property
    def reference_flowpaths_gdf(self) -> gpd.GeoDataFrame:
        """Load reference flowpaths layer."""
        return gpd.read_file(self.hydrofabric_path, layer="reference_flowpaths", ignore_geometry=True,)

    @cached_property
    def flowpaths_gdf(self) -> gpd.GeoDataFrame:
        """Load flowpaths layer."""
        return gpd.read_file(self.hydrofabric_path, layer="flowpaths", ignore_geometry=True,)

    @cached_property
    def vfp_length_mapping(self) -> dict[int, float]:
        """Make a mapping from virtual flowpath ID to its length."""
        return dict(zip(self.virtual_flowpaths_gdf["virtual_fp_id"].to_numpy(), self.virtual_flowpaths_gdf["length_km"].to_numpy()))

    @cached_property
    def virtual_name_remap(self) -> dict[int, int]:
        """Make a mapping from virtual nexuses to their downstream virtual flowpath ID."""
        # Drop rows where upstream ID is NaN
        mask = self.virtual_flowpaths_gdf["up_virtual_nex_id"].notna()

        up = self.virtual_flowpaths_gdf.loc[mask, "up_virtual_nex_id"].astype("int64").to_numpy()
        down = self.virtual_flowpaths_gdf.loc[mask, "virtual_fp_id"].to_numpy()

        return dict(zip(up, down))

    @cached_property
    def virtual_us_mapping(self) -> dict[int, list[int]]:
        """Make a mapping from virtual flowpath ID to the list of upstream virtual flowpath IDs."""
        mapping = defaultdict(list)
        for row in self.virtual_flowpaths_gdf.itertuples(index=False):
            key = self.virtual_name_remap.get(row[1])
            if key is not None:
                mapping[key].append(row[0])
        return dict(mapping)

    @cached_property
    def vfp_fp_mapping(self) -> dict[int, int]:
        """Make a mapping from virtual flowpath ID to flowpath ID."""
        reference_flowpaths_gdf = pd.merge(self.reference_flowpaths_gdf[["virtual_fp_id", "div_id"]], self.flowpaths_gdf[["div_id", "fp_id"]], on="div_id", how="left")
        return {row[0]: row[2] for row in reference_flowpaths_gdf.itertuples(index=False)}

    @cached_property
    def da_pct_mapping(self) -> dict[int, float]:
        """Make a mapping for percent area of each div corresponding to each virtual flowpath ID."""
        return dict(zip(self.virtual_flowpaths_gdf["virtual_fp_id"].to_numpy(), self.virtual_flowpaths_gdf["percentage_area_contribution"].to_numpy()))

    @cached_property
    def stream_order_mapping(self) -> dict[int, int]:
        """Make a mapping from flowpath ID to stream order."""
        return dict(zip(self.flowpaths_gdf["fp_id"].to_numpy(), self.flowpaths_gdf["stream_order"].to_numpy()))

@pytest.fixture
def run_context(request) -> RunContext:
    """Return a RunContext for a given run directory."""
    return RunContext(config_key=request.param)

@pytest.fixture(scope="session", autouse=True)
def build_fortran_extensions():
    """Ensure MC kernel is compiled."""
    ensure_f2py_module()


### BUILD TESTING ANALYSIS ###
# This stretches the definition of a fixture.  We're going to use it to build a big test dataset that can be used for both diagnostics and PyTest assertions.

@pytest.fixture
def sampled_run_dataset(run_context: RunContext) -> pd.DataFrame:
    """Build dataset of run diagnostics at the reach and network level."""
    # Sample from reaches
    reaches = sample_reaches(run_context)

    for k, v in reaches.items():
        diagnostics = generate_reach_diagnostics(run_context, k, v, plot=True)
        reaches[k].update(diagnostics)

    df = pd.DataFrame.from_dict(reaches, orient="index")
    df.to_parquet(f"{run_context.idx}.parquet")
    return df


### HELPER FUNCTIONS ###

def sample_reaches(run_context: RunContext, n_samples: int = 500, pct_length: float = 0.1, pct_slope: float = 0.1) -> dict[int, dict[str, Any]]:
    """Create a balanced random sample of reaches."""
    # Reserve pct_length% of samples for shortest and longest reaches
    # Reserve pct_slope% of samples for flatest and steepest reaches
    # Balance remaining % of samples across stream orders
    n_length = int(n_samples * pct_length)
    n_slope = int(n_samples * pct_slope)
    n_stream_order_reaches = n_samples - n_length - n_slope
    samples_per_stream_order = n_stream_order_reaches // run_context.max_stream_order
    working_reach_list = set()

    # Sample from stream orders
    for i in range(1, run_context.max_stream_order + 1):
        stream_order_reaches = run_context.flowpaths_gdf.loc[run_context.flowpaths_gdf["stream_order"] == i, "fp_id"]
        stream_order_sample = random.sample(list(stream_order_reaches), min(samples_per_stream_order, len(stream_order_reaches)))
        tmp_vfps = run_context.reference_flowpaths_gdf[run_context.reference_flowpaths_gdf["fp_id"].isin(stream_order_sample)].groupby("fp_id").agg("first")["virtual_fp_id"].values
        working_reach_list.update(tmp_vfps)

    # Sample from reach lengths
    routing_reaches = run_context.virtual_flowpaths_gdf[run_context.virtual_flowpaths_gdf["routing_segment"]]
    short_reaches = routing_reaches.nsmallest(n_length // 2, "length_km")["virtual_fp_id"]
    long_reaches = routing_reaches.nlargest(n_length - len(short_reaches), "length_km")["virtual_fp_id"]
    working_reach_list.update(short_reaches)
    working_reach_list.update(long_reaches)

    # Sample from reach slopes
    flat_reaches = run_context.flowpaths_gdf.nsmallest(n_slope // 2, "slope")["fp_id"]
    tmp_vfps = run_context.reference_flowpaths_gdf[run_context.reference_flowpaths_gdf["fp_id"].isin(flat_reaches)].groupby("fp_id").agg("first")["virtual_fp_id"].values
    working_reach_list.update(tmp_vfps)
    steep_reaches = run_context.flowpaths_gdf.nlargest(n_slope - len(flat_reaches), "slope")["fp_id"]
    tmp_vfps = run_context.reference_flowpaths_gdf[run_context.reference_flowpaths_gdf["fp_id"].isin(steep_reaches)].groupby("fp_id").agg("first")["virtual_fp_id"].values
    working_reach_list.update(tmp_vfps)

    # Pad out list with true randoms
    remaining_reaches = n_samples - len(working_reach_list)
    if remaining_reaches > 0:
        options = set(routing_reaches["virtual_fp_id"]) - working_reach_list
        random_reaches = random.sample(list(options), min(remaining_reaches, len(options)))
        working_reach_list.update(random_reaches)

    return attribute_reaches(run_context, working_reach_list)

def attribute_reaches(run_context: RunContext, reach_list: set[int]) -> dict[int, dict[str, Any]]:
    """Make a dictionary mapping reach ID to its attributes."""
    reach_attributes = {}
    for reach in reach_list:
        fp = run_context.vfp_fp_mapping[reach]
        stream_order = run_context.flowpaths_gdf.loc[run_context.flowpaths_gdf["fp_id"] == fp, "stream_order"].values[0]
        fp_gdf_row = run_context.flowpaths_gdf.loc[run_context.flowpaths_gdf["fp_id"] == fp]
        vfp_fp_gdf_row = run_context.virtual_flowpaths_gdf.loc[run_context.virtual_flowpaths_gdf["virtual_fp_id"] == reach]
        So = fp_gdf_row["slope"].values[0] #/ 100  # TODO: Check if this is what happens in NHF code and check what legacy code does.
        dx = vfp_fp_gdf_row["length_km"].values[0] * 1000
        n = fp_gdf_row["n"].values[0]
        Cs = fp_gdf_row["chslp"].values[0]
        Bw = fp_gdf_row["btmwdth"].values[0]
        Tw = fp_gdf_row["topwdth"].values[0]
        TwCC = fp_gdf_row["topwdthcc"].values[0]
        nCC = fp_gdf_row["ncc"].values[0]
        reach_attributes[reach] = {
            "So": So,
            "stream_order": stream_order,
            "dx": dx,
            "n": n,
            "Cs": Cs,
            "Bw": Bw,
            "Tw": Tw,
            "TwCC": TwCC,
            "nCC": nCC
        }
    return reach_attributes

def reroute(qlat: np.ndarray, qus: np.ndarray, dx: float, Bw: float, Tw: float, TwCC: float, n: float, nCC: float, Cs: float, So: float, dt: float, qts_subdivisions: int):
    import mc_fortran_port_2

    qus = np.repeat(qus, qts_subdivisions)
    qlat = np.repeat(qlat, qts_subdivisions)
    qdp = qus[0]
    velp = 0
    depthp = 0
    outflows = []
    depths = []
    courants = []
    celerities = []
    xs = []
    tws = []
    for i in range(len(qus)):
        quc = qus[i]
        qup = qus[i]
        ql = qlat[i]
        out = mc_fortran_port_2.muskingcunge_module.muskingcungenwm(
            dt,
            qup,
            quc,
            qdp,
            ql,
            dx,
            Bw,
            Tw,
            TwCC,
            n,
            nCC,
            Cs,
            So,
            velp,  # velp
            depthp,  # depthp
        )
        depthp = out[2]
        qdp = out[0]
        velp = out[1]
        geom = hydraulic_geometry(depthp, Bw, Tw, TwCC, Cs)
        if (i % qts_subdivisions) == 0:
            outflows.append(out[0])
            depths.append(out[2])
            courants.append(out[4])
            celerities.append(out[3])
            xs.append(out[5])
            tws.append(geom["twl"])

    courant_recalc = (np.array(celerities) * dt) / dx
    # x_recalc = 0.5 * (1 - ((np.array(qlat) + np.array(qus))[::int(qts_subdivisions)] / (np.array(tws) * np.array(celerities) * dx)))
    return {
        "qout": np.array(outflows),
        "depth": np.array(depths),
        "courant": np.array(courants),
        "celerity": np.array(celerities),
        "x": np.array(xs),
        "twl": np.array(tws),
        "courant_recalc": courant_recalc,
        # "x_recalc": x_recalc
    }

def hydraulic_geometry(depth: float, bw: float, tw: float, twcc: float, cs: float):
    if cs == 0:
        z = 1
    else:
        z = 1 / cs

    if bw > tw:
        bfd = bw / 0.00001
    elif bw == tw:
        bfd = bw / (2.0 * z)
    else:
        bfd = (tw - bw) / (2.0 * z)

    import mc_fortran_port_2
    twl, R, AREA, AREAC, WP, WPC, h_gt_bf, h_lt_bf = mc_fortran_port_2.muskingcunge_module.hydraulic_geometry(
        depth,
        bfd,
        bw,
        twcc,
        z,
    )

    return {
        "twl": twl,
        "R": R,
        "AREA": AREA,
        "AREAC": AREAC,
        "WP": WP,
        "WPC": WPC,
        "h_gt_bf": h_gt_bf,
        "h_lt_bf": h_lt_bf,
    }

def ensure_f2py_module():
    try:
        importlib.import_module(FORTRAN_MODULE_NAME)
        return
    except ImportError:
        pass

    cmd = [sys.executable, "-m", "numpy.f2py", "-c", str(FORTRAN_SRC), "-m", FORTRAN_MODULE_NAME]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError("Failed to build Fortran test dependency\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    # Make sure Python can now import it
    importlib.invalidate_caches()
    importlib.import_module(FORTRAN_MODULE_NAME)

def generate_reach_diagnostics(run_context: RunContext, reach_id: int, reach_attributes: dict[str, float], max_walk: float = 50.0, plot: bool = True) -> dict[str, float]:
    """Test mass conservation for a given reach."""
    # Initialize results dict
    results = {}

    # Load data
    local_qin, _ = virtual_flowpath_mass_conservation_network(reach_id, run_context, max_walk=0)
    network_qin, max_walk_actual = virtual_flowpath_mass_conservation_network(reach_id, run_context, max_walk=max_walk)
    qout_hydrograph = run_context.routed_results.sel(feature_id=reach_id)["flow"].values
    outflow = qout_hydrograph.sum()
    qlat, qus = get_inflows(run_context, reach_id)
    qin_hydrograph = qlat + qus
    rerouted_results = reroute(qlat, qus, reach_attributes["dx"], reach_attributes["Bw"], reach_attributes["Tw"], reach_attributes["TwCC"], reach_attributes["n"], reach_attributes["nCC"], reach_attributes["Cs"], reach_attributes["So"], dt=run_context.dt, qts_subdivisions=run_context.qts_subdivisions)

    # Check mass conservation at-reach
    results["local_mass_conservation_error"] = abs(local_qin - outflow) / outflow if outflow > 0 else np.nan

    # Check mass conservation across network
    results["network_mass_conservation_error"] = abs(network_qin - outflow) / outflow if outflow > 0 else np.nan
    results["network_mass_conservation_walk_dist"] = max_walk_actual

    # Calculate hydrograph lag
    timesteps = run_context.routed_results["time"].values
    t0 = timesteps[0]
    dt_seconds = (timesteps - t0) / np.timedelta64(1, "s")
    qin_sum = np.sum(qin_hydrograph)
    qout_sum = np.sum(qout_hydrograph)
    inflow_centroid = np.sum(qin_hydrograph * dt_seconds) / qin_sum if qin_sum > 0 else 0
    outflow_centroid = np.sum(qout_hydrograph * dt_seconds) / qout_sum if qout_sum > 0 else 0
    results["hydrograph_lag"] = (outflow_centroid - inflow_centroid)

    # Check courant conditions
    for i in ["courant", "celerity", "x", "courant_recalc"]:
        results[f"min_{i}"] = np.min(rerouted_results[i])
        results[f"max_{i}"] = np.max(rerouted_results[i])
        results[f"mean_{i}"] = np.mean(rerouted_results[i])
        results[f"median_{i}"] = np.median(rerouted_results[i])
        results[f"std_{i}"] = np.std(rerouted_results[i])
    results["negative_qout"] = np.min(rerouted_results["qout"]) < 0
    results["pct_attenuation"] = 1 - (np.max(qout_hydrograph) / np.max(qin_hydrograph)) if np.max(qin_hydrograph) > 0 else np.nan
    results["acceleration"] = np.max(qout_hydrograph) > np.max(qin_hydrograph)
    results["reroute_mass_error"] = abs(np.sum(rerouted_results["qout"]) - outflow) / outflow if outflow > 0 else np.nan
    reroute_sum = np.sum(rerouted_results["qout"])
    rerouted_centroid = np.sum(rerouted_results["qout"] * dt_seconds) / reroute_sum if reroute_sum > 0 else 0
    results["reroute_time_error"] = (outflow_centroid - rerouted_centroid)

    # Get ratio of reach length to courant ideal
    # Uses method of Ponce and Theurer (1982) Accuracy Criteria in Diffusion Routing
    qref = (qin_hydrograph.max() + qin_hydrograph.min()) / 2
    ref_ind = np.argmin(np.abs(qin_hydrograph - qref))
    qref_actual = rerouted_results["qout"][ref_ind]
    cref = rerouted_results["celerity"][ref_ind]
    twref = rerouted_results["twl"][ref_ind]
    dxc = run_context.dt * cref
    dxd = (qref_actual / twref) / (reach_attributes["So"] * cref)
    dxmax = 0.5 * (dxc + dxd)
    cmax = rerouted_results["celerity"].max()
    dxmin = cmax * run_context.dt
    ideal_dx = max([dxmin, dxmax])
    results["dx_ratio"] = reach_attributes["dx"] / ideal_dx

    if plot and results["acceleration"]:
        plot_hydrograps(qin_hydrograph, qout_hydrograph, rerouted_results["qout"], reach_id, run_context)
    return results

def plot_hydrograps(qin: np.ndarray, qout: np.ndarray, qreroute: np.ndarray, reach_id: int, run_context: RunContext):
    fig, ax = plt.subplots()
    timesteps = run_context.routed_results["time"].values
    ax.plot(timesteps, qin, label="qin", color="blue")
    ax.plot(timesteps, qout, label="qout", color="orange")
    ax.plot(timesteps, qreroute, label="reroute", color="green", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(f"Reach {reach_id}")
    ax.legend()
    fig.tight_layout()
    out_dir = Path(__file__).parent / "test_results"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{run_context.idx}_hydrograph_reach_{reach_id}.png")
    plt.close(fig)

def virtual_flowpath_mass_conservation_network(reach: int, run_context: RunContext, cur_dist: float = 0, max_walk: float = 10):
    cur_dist += run_context.vfp_length_mapping[reach]
    fp = run_context.vfp_fp_mapping[reach]
    out_calc = run_context.forcing_data[fp].values.sum() * run_context.da_pct_mapping[reach]
    if reach not in run_context.virtual_us_mapping:
        return out_calc, cur_dist
    elif cur_dist > max_walk:
        us = run_context.virtual_us_mapping[reach]
        us_streamflow = run_context.routed_results["flow"].sel(feature_id=us).sum(dim="feature_id").values
        out_calc += us_streamflow.sum()
        return out_calc, cur_dist
    elif cur_dist < max_walk:
        us = run_context.virtual_us_mapping[reach]
        us_q = [virtual_flowpath_mass_conservation_network(u, run_context, cur_dist, max_walk) for u in us]
        out_calc += sum([q[0] for q in us_q])
        return out_calc, max([q[1] for q in us_q])



def get_inflows(run_context: RunContext, reach_id: int) -> np.ndarray:
    """Get inflows to a reach by summing forcing and upstream flow."""
    local_qin = run_context.forcing_data[run_context.vfp_fp_mapping[reach_id]].values  * run_context.da_pct_mapping[reach_id]
    if reach_id not in run_context.virtual_us_mapping:
        return local_qin, np.zeros_like(local_qin)
    else:
        us = run_context.virtual_us_mapping[reach_id]
        us_q = run_context.routed_results["flow"].sel(feature_id=us).sum(dim="feature_id").values
        return local_qin, us_q

### TESTS ###

@pytest.mark.parametrize("run_context", CONFIG_FILES, indirect=True)
def test_test(sampled_run_dataset: pd.DataFrame):
    """Test that testing framework is working."""
    pass
