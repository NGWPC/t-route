import random
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

RESULTS_DIR = Path(__file__).parent / "conecuh_case" / "output"
FORCING_DIR = Path(__file__).parent / "conecuh_case" / "channel_forcing"
DOMAIN_PATH = (Path(__file__).parent / "conecuh_case" / "domain").glob("*.gpkg").__next__()  # assumes only one gpkg in domain directory

@dataclass(frozen=True)
class RunContext:
    """
    Relevant data for tests of a t-route run.

    This context bundles model outputs, forcings, and network mappings needed
    to compute reach-scale and network-integrated routing performance diagnostics.
    """

    routed_results: xr.Dataset
    """Outputs of t-route."""

    forcing_data: pd.DataFrame
    """Channel forcing data that was used to run t-route."""

    virtual_us_mapping: dict[int, list[int]]
    """Mapping from virtual flowpath ID to list of upstream virtual flowpath IDs."""

    vfp_fp_mapping: dict[int, int]
    """Mapping of virtual flowpath ID to flowpath ID (joined by div-id)."""

    da_pct_mapping: dict[int, float]
    """Percent of each flowpath's contributing area that is represented by its corresponding virtual flowpath."""

    vfp_length_mapping: dict[int, float]
    """Length of each virtual flowpath in km."""


@pytest.fixture(scope="module")
def mass_conservation_context(
    routed_results,
    forcing_data,
    virtual_us_mapping,
    vfp_fp_mapping,
    da_pct_mapping,
    vfp_length_mapping,
) -> RunContext:
    return RunContext(
        routed_results=routed_results,
        forcing_data=forcing_data,
        virtual_us_mapping=virtual_us_mapping,
        vfp_fp_mapping=vfp_fp_mapping,
        da_pct_mapping=da_pct_mapping,
        vfp_length_mapping=vfp_length_mapping,
    )

@pytest.fixture(scope="session")
def routed_results() -> xr.Dataset:
    """Load results from t-route run."""
    paths = sorted(RESULTS_DIR.glob("*.nc"))
    flows = [xr.open_dataset(p, engine="netcdf4") for p in paths]
    return xr.concat(flows, dim="time")

@pytest.fixture(scope="session")
def forcing_data() -> pd.DataFrame:
    """Load inputs to t-route run."""
    forcing_files = sorted(list(FORCING_DIR.glob("*.csv")))
    forcing = pd.concat([pd.read_csv(f).sort_values("feature_id") for f in forcing_files], axis=1)
    columns = forcing["feature_id"].values[:, 0]
    forcing = forcing.drop(columns=["feature_id"]).T
    forcing.columns = columns
    forcing.index = pd.to_datetime(forcing.index)
    return forcing

@pytest.fixture(scope="session")
def virtual_flowpaths_gdf() -> gpd.GeoDataFrame:
    """Load virtual flowpaths layer."""
    return gpd.read_file(DOMAIN_PATH, layer="virtual_flowpaths", ignore_geometry=True)

@pytest.fixture(scope="session")
def reference_flowpaths_gdf() -> gpd.GeoDataFrame:
    """Load reference flowpaths layer."""
    return gpd.read_file(DOMAIN_PATH, layer="reference_flowpaths", ignore_geometry=True,)

@pytest.fixture(scope="session")
def flowpaths_gdf() -> gpd.GeoDataFrame:
    """Load flowpaths layer."""
    return gpd.read_file(DOMAIN_PATH, layer="flowpaths", ignore_geometry=True,)

@pytest.fixture(scope="session")
def vfp_length_mapping(virtual_flowpaths_gdf: gpd.GeoDataFrame) -> dict[int, float]:
    """Make a mapping from virtual flowpath ID to its length."""
    return dict(zip(virtual_flowpaths_gdf["virtual_fp_id"], virtual_flowpaths_gdf["length_km"]))

@pytest.fixture(scope="session")
def virtual_name_remap(virtual_flowpaths_gdf: gpd.GeoDataFrame) -> dict[int, int]:
    """Make a mapping from virtual nexuses to their downstream virtual flowpath ID."""
    return dict(zip(virtual_flowpaths_gdf["up_virtual_nex_id"], virtual_flowpaths_gdf["virtual_fp_id"]))

@pytest.fixture(scope="session")
def virtual_us_mapping(virtual_flowpaths_gdf: gpd.GeoDataFrame, virtual_name_remap: dict[int, int]) -> dict[int, list[int]]:
    """Make a mapping from virtual flowpath ID to the list of upstream virtual flowpath IDs."""
    mapping = defaultdict(list)
    for row in virtual_flowpaths_gdf.itertuples(index=False):
        key = virtual_name_remap.get(row[1])
        if key is not None:
            mapping[key].append(row[0])
    return dict(mapping)

@pytest.fixture(scope="session")
def vfp_fp_mapping(reference_flowpaths_gdf: gpd.GeoDataFrame, flowpaths_gdf: gpd.GeoDataFrame) -> dict[int, int]:
    """Make a mapping from virtual flowpath ID to flowpath ID."""
    reference_flowpaths_gdf = pd.merge(reference_flowpaths_gdf[["virtual_fp_id", "div_id"]], flowpaths_gdf[["div_id", "fp_id"]], on="div_id", how="left")
    return {row[0]: row[2] for row in reference_flowpaths_gdf.itertuples(index=False)}

@pytest.fixture(scope="session")
def da_pct_mapping(virtual_flowpaths_gdf: gpd.GeoDataFrame) -> dict[int, float]:
    """Make a mapping for percent area of each div corresponding to each virtual flowpath ID."""
    return dict(zip(virtual_flowpaths_gdf["virtual_fp_id"], virtual_flowpaths_gdf["percentage_area_contribution"]))



def test_virtual_flowpath_mass_conservation(routed_results: xr.Dataset, forcing_data: pd.DataFrame, virtual_us_mapping: dict[int, list[int]], vfp_fp_mapping: dict[int, int], da_pct_mapping: dict[int, float]):
    """Test that mass is conserved for a random sample of reaches."""
    # Derive errors
    conservation_errors = {}
    for i in virtual_us_mapping:  # Only non-headwaters
        in_sum, out_sum = _test_virtual_flowpath_mass_conservation_single(i, routed_results, forcing_data, virtual_us_mapping, vfp_fp_mapping, da_pct_mapping)
        conservation_errors[i] = abs(in_sum - out_sum) / out_sum
    df = pd.DataFrame.from_dict(conservation_errors, orient="index", columns=["pct_error"])

    # Export diagnostics
    df.to_parquet(Path(__file__).parent / "test_results" /  "vfp_mass_Conservation.parquet")
    fig, ax = plt.subplots()
    ax.hist(df["pct_error"] * 100, fc="darkgray", ec="k", bins=35)
    ax.set_xlabel("Mass Error (abs(Qin-Qout)) as Percent of Total Outflow")
    ax.set_ylabel("Frequency")
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "test_results" /  "vfp_mass_Conservation.png")
    plt.close(fig)

    # PyTest
    df = df[df["pct_error"] > 0.1]
    if len(df) > 0:
        raise RuntimeError(f"Mass conservation errors found\n{df}")

def _test_virtual_flowpath_mass_conservation_single(reach: int, routed_results: xr.Dataset, forcing_data: pd.DataFrame, virtual_us_mapping: dict[int, list[int]], vfp_fp_mapping: dict[int, int], da_pct_mapping: dict[int, float]):
    us = virtual_us_mapping[reach]
    fp = vfp_fp_mapping[reach]
    da_pct = da_pct_mapping[reach]
    us_streamflow = routed_results.sel(feature_id=us)["flow"].sum(dim="feature_id").values
    outflow = routed_results.sel(feature_id=reach)["flow"].values
    qlat_forcing = forcing_data[fp].values.sum() * da_pct
    inflow = us_streamflow.sum() + qlat_forcing

    return inflow, outflow.sum()


def test_network_mass_conservation(routed_results: xr.Dataset, forcing_data: pd.DataFrame, virtual_us_mapping: dict[int, list[int]], vfp_fp_mapping: dict[int, int], da_pct_mapping: dict[int, float], vfp_length_mapping: dict[int, float]):
    max_walk = 10  # km
    random_keys = random.sample(list(virtual_us_mapping.keys()), 300)
    conservation_errors = {}

    for i in random_keys:
        qin, tmp_max_walk = _test_virtual_flowpath_mass_conservation_network(i, routed_results, forcing_data, virtual_us_mapping, vfp_fp_mapping, da_pct_mapping, vfp_length_mapping, cur_dist=0, max_walk=max_walk)
        outflow = routed_results.sel(feature_id=i)["flow"].values.sum()
        conservation_errors[i] = abs(qin - outflow) / outflow if outflow > 0 else 0
    df = pd.DataFrame.from_dict(conservation_errors, orient="index", columns=["pct_error"])

    # Export diagnostics
    df.to_parquet(Path(__file__).parent / "test_results" /  f"vfp_mass_conservation_network_{max_walk}.parquet")
    fig, ax = plt.subplots()
    ax.hist(df["pct_error"] * 100, fc="darkgray", ec="k", bins=35)
    ax.set_xlabel("Mass Error (abs(Qin-Qout)) as Percent of Total Outflow")
    ax.set_ylabel("Frequency")
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "test_results" /  f"vfp_mass_conservation_network_{max_walk}.png")
    plt.close(fig)

    # PyTest
    df = df[df["pct_error"] > 0.1]
    if len(df) > 0:
        raise RuntimeError(f"Mass conservation errors found\n{df}")


def _test_virtual_flowpath_mass_conservation_network(reach: int, routed_results: xr.Dataset, forcing_data: pd.DataFrame, virtual_us_mapping: dict[int, list[int]], vfp_fp_mapping: dict[int, int], da_pct_mapping: dict[int, float], vfp_length_mapping: dict[int, float], cur_dist: float = 0, max_walk: float = 10):
    cur_dist += vfp_length_mapping[reach]
    fp = vfp_fp_mapping[reach]
    out_calc = forcing_data[fp].values.sum() * da_pct_mapping[reach]
    if reach not in virtual_us_mapping:
        return out_calc, cur_dist
    elif cur_dist > max_walk:
        us = virtual_us_mapping[reach]
        us_streamflow = routed_results.sel(feature_id=us)["flow"].sum(dim="feature_id").values
        out_calc += us_streamflow.sum()
        return out_calc, cur_dist
    elif cur_dist < max_walk:
        us = virtual_us_mapping[reach]
        us_q = [_test_virtual_flowpath_mass_conservation_network(u, routed_results, forcing_data, virtual_us_mapping, vfp_fp_mapping, da_pct_mapping, vfp_length_mapping, cur_dist, max_walk) for u in us]
        out_calc += sum([q[0] for q in us_q])
        return out_calc, max([q[1] for q in us_q])


def
