
from dataclasses import dataclass
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely import LineString
from shapely.geometry import Point
import numpy as np
import shapely
from pathlib import Path
from shapely.geometry.base import BaseGeometry


### DATA CLASSES ###

@dataclass
class FlowpathData:
    geoms: np.ndarray
    lengths: np.ndarray
    ids: np.ndarray
    dn_nex_ids: np.ndarray

@dataclass
class VirtualNexusData:
    geoms: np.ndarray
    ids: np.ndarray
    fp_ids: np.ndarray

    def filter(self, mask: np.ndarray) -> None:
        self.geoms = self.geoms[mask]
        self.ids = self.ids[mask]
        self.fp_ids = self.fp_ids[mask]

@dataclass
class FlowpathNode:
    ids: np.ndarray
    distances: np.ndarray
    fp_inds: np.ndarray

@dataclass
class LinkArrays:
    fp_ind: np.ndarray
    ds_node_id: np.ndarray
    us_node_id: np.ndarray
    start_frac: np.ndarray
    end_frac: np.ndarray
    lengths: np.ndarray

    def filter(self, mask: np.ndarray) -> None:
        self.fp_ind = self.fp_ind[mask]
        self.ds_node_id = self.ds_node_id[mask]
        self.us_node_id = self.us_node_id[mask]
        self.start_frac = self.start_frac[mask]
        self.end_frac = self.end_frac[mask]
        self.lengths = self.lengths[mask]

### MAIN FUNCTION ###

def discretize_flowpaths(
    flowpaths: gpd.GeoDataFrame,
    virtual_flowpaths: pd.DataFrame,
    virtual_nexus: gpd.GeoDataFrame,
    reference_flowpaths: pd.DataFrame,
    nexus: gpd.GeoDataFrame,
    discretization_len_m: float = 300.0,
    aggregate_short_reaches: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # # Perform joins to prep data
    # vfp_to_div = reference_flowpaths.dropna(subset="virtual_fp_id").set_index("virtual_fp_id")["div_id"].to_dict()
    # div_to_fp = reference_flowpaths.dropna(subset=["fp_id"]).set_index("div_id")["fp_id"].astype(int).to_dict()
    # vnex_to_fp_id = virtual_flowpaths[["dn_virtual_nex_id", "virtual_fp_id"]].copy()
    # vnex_to_fp_id["div_id"] = vnex_to_fp_id["virtual_fp_id"].map(vfp_to_div)
    # vnex_to_fp_id["fp_id"] = vnex_to_fp_id["div_id"].map(div_to_fp)
    # virtual_nexus = virtual_nexus.merge(vnex_to_fp_id[["dn_virtual_nex_id", "fp_id"]], left_on="virtual_nex_id", right_on="dn_virtual_nex_id", how="left")
    # vnex_ids = virtual_nexus["virtual_nex_id"].to_numpy()

    # graph_base = flowpaths[["fp_id", "dn_nex_id", "up_nex_id"]].set_index("fp_id")
    # fp_dn_nex_dict = graph_base["dn_nex_id"].to_dict()
    # # Add new nodes for u/s ends of flowpaths
    # headwater_mask = graph_base["up_nex_id"].isnull()
    # max_id = max([flowpaths["dn_nex_id"].max(), virtual_flowpaths["dn_virtual_nex_id"].max()]) + 1
    # graph_base.loc[headwater_mask, "up_nex_id"] = np.arange(max_id, max_id + headwater_mask.sum())
    # fp_up_nex_dict = graph_base["up_nex_id"].astype(int).to_dict()

    # Extract necessary data, pulling out into arrays for efficiency/performance
    fp_dict = _get_fp_dict(flowpaths, nexus)

    vnex_dict = _get_vnex_dict(reference_flowpaths, virtual_flowpaths, virtual_nexus) 

    # Create links
    links_dict = _create_links(fp_dict, vnex_dict)
    if aggregate_short_reaches:
        links_dict = _aggregate_links(links_dict)
    links_dict = _discretize_links(links_dict)

    # Format and return
    return _format_link_df(links_dict)

### HELPER FUNCTIONS ###

def export_segments_and_nodes(
    segments: pd.DataFrame,
    flowpaths: gpd.GeoDataFrame,
    output_gpkg: str,
):
    flowpaths["geometry"] = shapely.line_merge(flowpaths.geometry.values)
    flow_lookup = flowpaths.set_index("fp_id").geometry
    seg_geom = []
    us_nodes = segments.index.to_numpy()
    ds_nodes = segments["downstream"].to_numpy()

    start_frac = segments["start_frac"].to_numpy()
    end_frac = segments["end_frac"].to_numpy()

    fp_ids = segments["fp_id"].to_numpy()

    # Build segment geometries
    for fp, s, e in zip(fp_ids, start_frac, end_frac):
        line = flow_lookup.loc[fp]
        seg_geom.append(shapely.ops.substring(line, s, e, normalized=True))

    seg_gdf = gpd.GeoDataFrame(
        segments.copy(),
        geometry=seg_geom,
        crs=flowpaths.crs,
    )

    # ---- Build nodes ----

    node_geom = {}

    for fp, s, e, us, ds in zip(fp_ids, start_frac, end_frac, us_nodes, ds_nodes):
        line = flow_lookup.loc[fp]

        if us not in node_geom:
            node_geom[us] = shapely.line_interpolate_point(line, s, normalized=True)

        if ds not in node_geom:
            node_geom[ds] = shapely.line_interpolate_point(line, e, normalized=True)

    node_gdf = gpd.GeoDataFrame(
        {"node_id": list(node_geom.keys())},
        geometry=list(node_geom.values()),
        crs=flowpaths.crs,
    ).set_index("node_id")

    # ---- Export ----

    output_gpkg = Path(output_gpkg)

    seg_gdf.to_file(output_gpkg, layer="segments", driver="GPKG")
    node_gdf.to_file(output_gpkg, layer="nodes", driver="GPKG")


def _get_fp_dict(flowpaths: gpd.GeoDataFrame, nexus: gpd.GeoDataFrame) -> FlowpathData:
    # Correct flowpath orientations
    nex_lookup = nexus.set_index("nex_id").geometry
    dn_nex_geom = nex_lookup.loc[flowpaths["dn_nex_id"].to_numpy()].values
    corrected_geoms = _correct_fp_orientations(flowpaths.geometry.values, dn_nex_geom)

    return FlowpathData(
        corrected_geoms,
        flowpaths["length_km"].to_numpy() * 1000,
        flowpaths["fp_id"].to_numpy(),
        flowpaths["dn_nex_id"].to_numpy()
    )

def _get_vnex_dict(reference_flowpaths: pd.DataFrame, virtual_flowpaths: gpd.GeoDataFrame, virtual_nexus: gpd.GeoDataFrame) -> VirtualNexusData:
    vnex_to_fp = pd.merge(virtual_nexus[["geometry", "virtual_nex_id"]], virtual_flowpaths[["up_virtual_nex_id", "virtual_fp_id"]], how="left", left_on="virtual_nex_id", right_on="up_virtual_nex_id")
    vnex_to_fp = pd.merge(vnex_to_fp, reference_flowpaths[["virtual_fp_id", "div_id"]].drop_duplicates(), how="left", on="virtual_fp_id")[["geometry", "virtual_nex_id", "div_id"]]
    vnex_to_fp = pd.merge(vnex_to_fp, reference_flowpaths[["div_id", "fp_id"]].dropna().drop_duplicates(), how="left", on="div_id")[["geometry", "virtual_nex_id", "fp_id"]]

    # TODO: Remove when NHF patched
    vnex_to_fp = vnex_to_fp.dropna()

    return VirtualNexusData(
        vnex_to_fp["geometry"].to_numpy(),
        vnex_to_fp["virtual_nex_id"].to_numpy(),
        vnex_to_fp["fp_id"].to_numpy().astype(int)
    )


def _correct_fp_orientations(flow_geom: np.ndarray[BaseGeometry], dn_nex_geom: np.ndarray[BaseGeometry]) -> np.ndarray[BaseGeometry]:
    # Correct any backwards flowpaths
    start_pts = shapely.get_point(flow_geom, 0)
    end_pts = shapely.get_point(flow_geom, -1)

    dist_start = shapely.distance(start_pts, dn_nex_geom)
    dist_end = shapely.distance(end_pts, dn_nex_geom)

    reverse_mask = dist_start < dist_end
    flow_geom = np.where(reverse_mask, shapely.reverse(flow_geom), flow_geom)

    return flow_geom

def _create_links(fp: FlowpathData, vnex: VirtualNexusData) -> LinkArrays:
    vnex_sorted = _attribute_and_sort_vnex(fp, vnex)

    ## Construct segment lengths array

    # Count number of virtual nexus per flowpath
    counts = np.bincount(vnex_sorted.fp_inds, minlength=len(fp.geoms))
    seg_counts = counts + 1  # N+2 breakpoints (including ends); N+1 segments

    # Offsets for flattened arrays
    offsets = np.zeros_like(seg_counts)
    offsets[1:] = np.cumsum(seg_counts[:-1])
    total_segments = offsets[-1] + seg_counts[-1]

    # Prepare arrays
    start_frac = np.zeros(total_segments, dtype=np.float64)
    end_frac = np.zeros(total_segments, dtype=np.float64)
    seg_fp_idx = np.zeros(total_segments, dtype=np.int64)
    ds_node_id = np.empty(total_segments, dtype=np.int64)
    us_node_id = np.empty(total_segments, dtype=np.int64)

    # Initialize dict to store link to flowpath outlet mapping
    fp_outlet_crosswalk = {}

    # Attribute links
    cur = 0

    for i in range(len(fp.geoms)):
        # get fractions for this flowpath
        mask = vnex_sorted.fp_inds == i
        tmp_vnex = np.append(vnex_sorted.ids[mask], )
        dists_full = np.append(vnex_sorted.distances[mask], )

        n = len(dists_full)
        tmp_fp_id = fp.ids[i]
        ds_node_id[cur:cur + n] = tmp_vnex[1:]  # TODO: validate
        us_node_id[cur:cur + n] = tmp_vnex
        start_frac[cur:cur + n] = dists_full[:-1]
        end_frac[cur:cur + n] = dists_full[1:]
        seg_fp_idx[cur:cur + n] = i

        # Store last seg
        fp_outlet_crosswalk[us_node_id[cur + n - 1]] = tmp_fp_id

        cur += n

    return LinkArrays(
        seg_fp_idx,
        ds_node_id,
        us_node_id,
        start_frac,
        end_frac,
        (end_frac - start_frac) * fp.lengths[seg_fp_idx]
    )

def _attribute_and_sort_vnex(fp: FlowpathData, vnex: VirtualNexusData) -> FlowpathNode:
    fp_index_map = {fp: i for i, fp in enumerate(fp.ids)}  # map fp_id to index in flowpath layers
    vnex_flow_ind = np.array([fp_index_map[x] for x in vnex.fp_ids])  # fp_id indices corresponding to each virtual nexus

    # TODO: Need to filter out non-intersecting before vnex_flow_ind
    intersecting_mask = shapely.intersects(vnex.geoms, fp.geoms[vnex_flow_ind])
    vnex.filter(intersecting_mask)
    vnex_flow_ind = vnex_flow_ind[intersecting_mask]

    ## Get distance along each flowpath of each line

    # Indexing to get flowpath line for each virtual nexus
    
    # Compute fraction along flowpath for each virtual nexus
    frac = shapely.line_locate_point(fp.geoms[vnex_flow_ind], vnex.geoms, normalized=True)
    # frac = dist / fp.lengths[vnex_flow_ind]

    # Sort fractions per flowpath. frac is potentially unsorted
    vnex_sorted_idx = np.argsort(vnex_flow_ind + frac * 1e-9)  # tiny factor to break ties
    return FlowpathNode(
        vnex.ids[vnex_sorted_idx],
        frac[vnex_sorted_idx],  # virtual nexus fractions ordered by ocurrence along flowpath
        vnex_flow_ind[vnex_sorted_idx]  # flowpath index for each virtual nexus
    )

def _aggregate_links(links_dict: LinkArrays, discretization_len_m: float) -> LinkArrays:
    ## Combine links that are below discretization length with the next upstream reach
    short_mask = links_dict.lengths < discretization_len_m

    if not np.any(short_mask):
        return links_dict
    
    short_idx = np.where(short_mask)[0]

    for idx in short_idx:
        us_node = links_dict.us_node_id[idx]
        ds_node = links_dict.ds_node_id[idx]
        length = links_dict.lengths[idx]

        us_filter = links_dict.ds_node_id == us_node
        links_dict.ds_node_id[us_filter] = ds_node
        links_dict.lengths[us_filter] += length

    links_dict.filter(~short_idx)

    return links_dict

def _discretize_links(links_dict: LinkArrays, discretization_len_m: float, cur_node_id: int = 0) -> LinkArrays:

    ## Subdivide to target length
    long_mask = links_dict.lengths > discretization_len_m

    if not np.any(long_mask):
        return

    # indices of long segments
    long_idx = np.where(long_mask)[0]

    subdiv_fp_ind = []
    subdiv_start = []
    subdiv_end = []
    subdiv_ds_node = []
    subdiv_us_node = []
    subdiv_id = []


    for idx in long_idx:
        n = int(np.ceil(links_dict.lengths[idx] / discretization_len_m))
        fracs = np.linspace(links_dict.start_frac[idx], links_dict.end_frac[idx], n + 1)
        new_node_ids = np.arange(cur_node_id, cur_node_id + n - 1)
        node_ids = np.concatenate(([links_dict.us_node_id[idx]], new_node_ids, [links_dict.ds_node_id[idx]]))

        cur_node_id += n - 1

        subdiv_us_node.append(node_ids[:-1])
        subdiv_ds_node.append(node_ids[1:])
        subdiv_fp_ind.append(np.full(n, links_dict.fp_ind[idx]))
        subdiv_start.append(fracs[:-1])
        subdiv_end.append(fracs[1:])

        if node_ids[0] in fp_outlet_crosswalk:
            tmp_fp_id = fp_outlet_crosswalk.pop(node_ids[0])
            fp_outlet_crosswalk[node_ids[-2]] = tmp_fp_id

    subdiv_fp_ind = np.concatenate(subdiv_fp_ind)
    subdiv_start = np.concatenate(subdiv_start)
    subdiv_end = np.concatenate(subdiv_end)
    subdiv_id = np.concatenate(subdiv_id)
    subdiv_ds_node = np.concatenate(subdiv_ds_node)
    subdiv_us_node = np.concatenate(subdiv_us_node)

    subdiv_lengths = (subdiv_end - subdiv_start) * links_dict.lengths[long_mask]

    # mask out original long segments
    keep_mask = ~long_mask
    start_frac = np.concatenate([start_frac[keep_mask], subdiv_start])
    end_frac = np.concatenate([end_frac[keep_mask], subdiv_end])
    seg_fp_idx = np.concatenate([seg_fp_idx[keep_mask], subdiv_fp])
    lengths = np.concatenate([lengths[keep_mask], subdiv_lengths])
    link_id = np.concatenate([link_id[keep_mask], subdiv_id])
    us_node_id = np.concatenate([us_node_id[keep_mask], subdiv_us_node])
    ds_node_id = np.concatenate([ds_node_id[keep_mask], subdiv_ds_node])



def _format_link_df(links: dict[str, np.ndarray]) -> pd.DataFrame:

    # Fetch channel geometry
    channel_params = ["n", "mainstem_lp", "topwdth", "slope", "ncc", "btmwdth", "musx", "chslp", "topwdthcc", "musk"]
    channel_dict = {i: flowpaths[i].to_numpy()[seg_fp_idx] for i in channel_params}

    ## Export
    segments = pd.DataFrame(
        {
            "fp_id": fp_ids[seg_fp_idx],
            "ds_node_id": ds_node_id,
            "us_node_id": us_node_id,
            "link_id": link_id,
            "length": lengths,
            "alt": np.zeros_like(lengths),
            "start_frac": start_frac,
            "end_frac": end_frac,
            **channel_dict
        }
    )
    # Conform to abstractnetwork _dataframe
    renames = {
        "ds_node_id": "downstream",
        "length": "dx",
        "mainstem_lp": "mainstem",
        "topwdth": "tw",
        "slope": "s0",
        "btmwdth": "bw",
        "length_km": "dx",
        "chslp": "cs",
        "topwdthcc": "twcc"
    }
    segments = segments.set_index("us_node_id").rename(columns=renames)

    # export_segments_and_nodes(segments, flowpaths, "discretized_network.gpkg")
    return segments, fp_outlet_crosswalk
