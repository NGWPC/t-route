
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely import LineString, MultiLineString, line_interpolate_point
from shapely.geometry import Point
import numpy as np
import shapely
from pathlib import Path
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring
from zmq import has


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
    fp_id: np.ndarray
    ds_node_id: np.ndarray
    us_node_id: np.ndarray
    start_frac: np.ndarray
    end_frac: np.ndarray
    lengths: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        n = len(df)
        return cls(
            df["fp_id"].to_numpy().astype(int),
            df["dn_virtual_nex_id"].to_numpy().astype(int),
            df["up_virtual_nex_id"].to_numpy().astype(int),
            np.zeros(n),
            np.zeros(n),
            df["length_km"].to_numpy() * 1000
        )
    
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({f.name: getattr(self, f.name) for f in fields(self)})

    def filter(self, mask: np.ndarray) -> None:
        self.fp_id = self.fp_id[mask]
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

    virtual_flowpaths = _tmp_find_routing_vfps(virtual_flowpaths, flowpaths, reference_flowpaths)
    max_node_ind = virtual_flowpaths["up_virtual_nex_id"].max() + 1
    
    links = LinkArrays.from_df(
        pd.merge(virtual_flowpaths, reference_flowpaths[["fp_id", "virtual_fp_id"]].drop_duplicates(), on="virtual_fp_id")
    )
    links, node_remapping = _aggregate_links(links, discretization_len_m)
    links = _discretize_links(links, discretization_len_m, max_node_ind)

    # export_segments_and_nodes(links, virtual_flowpaths, "links_nodes_discretization.gpkg")
    return *_format_link_df(links, flowpaths), node_remapping


    # # Extract necessary data, pulling out into arrays for efficiency/performance
    # fp_dict = _get_fp_dict(flowpaths, nexus)

    # vnex_dict = _get_vnex_dict(reference_flowpaths, virtual_flowpaths, virtual_nexus) 

    # fp_ds_vnex_dict = _get_fp_ds_vnex_dict(reference_flowpaths, virtual_flowpaths)

    # # Create links
    # links_dict = _create_links(fp_dict, vnex_dict, fp_ds_vnex_dict)
    # if aggregate_short_reaches:
    #     links_dict = _aggregate_links(links_dict, discretization_len_m)
    # links_dict = _discretize_links(links_dict, discretization_len_m, max_node_ind)

    # # Format and return
    # export_segments_and_nodes(links_dict, virtual_flowpaths, "links_nodes_discretization.gpkg")
    # return _format_link_df(links_dict)

### HELPER FUNCTIONS ###

def export_segments_and_nodes(
    segments: LinkArrays,
    src_vfp: gpd.GeoDataFrame,
    output_gpkg: str,
):
    # Initialize data stores
    us_node_ids = []
    ds_node_ids = []
    link_geometries = []
    node_geometries = []

    # Make network graphs
    ds_network = dict(zip(segments.us_node_id, segments.ds_node_id))
    ds_network[set(segments.ds_node_id).difference(segments.us_node_id).pop()] = -9999

    len_lookup = dict(zip(segments.us_node_id, segments.lengths))

    # Processing loop
    src_vfp = src_vfp.set_index("up_virtual_nex_id")
    non_eclipsed = src_vfp[src_vfp.index.isin(segments.us_node_id)]
    for i in non_eclipsed.itertuples():
        us_nex = int(i.Index)
        dn_nex = i.dn_virtual_nex_id

        # Merge eclipsed
        geoms = [i.geometry]
        while dn_nex not in segments.ds_node_id:
            tmp = src_vfp.loc[dn_nex]
            geoms.append(tmp.geometry)
            dn_nex = tmp.dn_virtual_nex_id
        geom = shapely.line_merge(MultiLineString([j.geoms[0] for j in geoms[::-1]]))

        # Get segments
        seg_us = us_nex
        seg_ds = ds_network[seg_us]
        segs = [us_nex]
        lengths = [len_lookup[seg_us]]
        while seg_ds != dn_nex:
            segs.append(seg_ds)
            lengths.append(len_lookup[seg_us])
            seg_ds = ds_network[seg_ds]
        
        # Subdivide geometry
        cumdist = np.cumsum([0.0] + lengths)
        cumdist[-1] = geom.length  # just in case
        seg_geoms = [
            substring(geom, start_dist, end_dist)
            for start_dist, end_dist in zip(cumdist[:-1], cumdist[1:])
        ]
        node_geoms = [line_interpolate_point(geom, start_dist) for start_dist in cumdist[:-1]]

        # Log
        us_node_ids.extend(segs)
        ds_node_ids.extend(segs[1:] + [dn_nex])
        link_geometries.extend(seg_geoms)
        node_geometries.extend(node_geoms)

    # Export geopackage
    gpd.GeoDataFrame({"us_virtual_nex_id": us_node_ids, "ds_virtual_nex_id": ds_node_ids}, geometry=link_geometries, crs=src_vfp.crs).to_file(output_gpkg, layer="links")
    gpd.GeoDataFrame({"virtual_nex_id": us_node_ids}, geometry=node_geometries, crs=src_vfp.crs).to_file(output_gpkg, layer="nodes")

def _tmp_find_routing_vfps(virtual_flowpaths: gpd.GeoDataFrame, flowpaths: gpd.GeoDataFrame, reference_flowpaths: pd.DataFrame) -> gpd.GeoDataFrame:
    ### TODO: REMOVE THIS AFTER NHF UPDATE
    vfp = virtual_flowpaths.rename(columns={"geometry": "vfp_geom"})
    fp = flowpaths.rename(columns={"geometry": "fp_geom"})

    vfp_to_fp = pd.merge(
        vfp,
        reference_flowpaths[["virtual_fp_id", "div_id"]].drop_duplicates(),
        how="left",
        on="virtual_fp_id",
    )[["virtual_fp_id", "vfp_geom", "div_id"]]

    vfp_to_fp = pd.merge(
        vfp_to_fp,
        reference_flowpaths[["div_id", "fp_id"]].dropna().drop_duplicates(),
        how="left",
        on="div_id",
    )[["virtual_fp_id", "vfp_geom", "fp_id"]]

    vfp_to_fp = pd.merge(
        vfp_to_fp,
        fp[["fp_id", "fp_geom"]],
        how="left",
        on="fp_id",
    )

    # Check if first point of vfp matches first point of fp
    vfp_to_fp["vfp_geom"] = shapely.get_point(shapely.get_geometry(vfp_to_fp["vfp_geom"], 0).values, 0)
    vfp_to_fp["fp_geom"]  = shapely.get_point(shapely.get_geometry(vfp_to_fp["fp_geom"], 0).values, 0)

    matches = shapely.equals(vfp_to_fp["vfp_geom"], vfp_to_fp["fp_geom"])

    # Force new virtual nexus at top
    matches = matches & virtual_flowpaths["up_virtual_nex_id"].isna()
    start_id = virtual_flowpaths["up_virtual_nex_id"].max() + 1
    new_ids = np.arange(start_id, start_id + matches.sum())
    virtual_flowpaths.loc[matches, "up_virtual_nex_id"] = new_ids

    # gpd.GeoDataFrame(vfp_to_fp, geometry="vfp_geom", crs=flowpaths.crs).to_file("ends.gpkg", layer="vfp")
    # gpd.GeoDataFrame(vfp_to_fp, geometry="fp_geom", crs=flowpaths.crs).to_file("ends.gpkg", layer="fp")

    virtual_flowpaths = virtual_flowpaths.dropna(subset="up_virtual_nex_id")
    virtual_flowpaths["up_virtual_nex_id"] = virtual_flowpaths["up_virtual_nex_id"].astype(int)
    return virtual_flowpaths

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

def _get_fp_ds_vnex_dict(reference_flowpaths: pd.DataFrame, virtual_flowpaths: gpd.GeoDataFrame) -> dict[int, int]:
    tmp_ref = pd.merge(reference_flowpaths[["fp_id", "virtual_fp_id"]], virtual_flowpaths[["virtual_fp_id", "dn_virtual_nex_id"]], how="left", on="virtual_fp_id")
    # Assumes that virtual flowpath id increases in d/s direction
    return tmp_ref[["fp_id", "dn_virtual_nex_id"]].dropna().astype(int).groupby("fp_id")["dn_virtual_nex_id"].max().to_dict()

def _correct_fp_orientations(flow_geom: np.ndarray[BaseGeometry], dn_nex_geom: np.ndarray[BaseGeometry]) -> np.ndarray[BaseGeometry]:
    # Correct any backwards flowpaths
    start_pts = shapely.get_point(flow_geom, 0)
    end_pts = shapely.get_point(flow_geom, -1)

    dist_start = shapely.distance(start_pts, dn_nex_geom)
    dist_end = shapely.distance(end_pts, dn_nex_geom)

    reverse_mask = dist_start < dist_end
    flow_geom = np.where(reverse_mask, shapely.reverse(flow_geom), flow_geom)

    return flow_geom

def _create_links(fp: FlowpathData, vnex: VirtualNexusData, fp_ds_vnex_dict: dict[int, int]) -> LinkArrays:
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
        tmp_fp_id = fp.ids[i]

        # get fractions for this flowpath
        mask = vnex_sorted.fp_inds == i
        tmp_vnex = np.append(vnex_sorted.ids[mask], fp_ds_vnex_dict[tmp_fp_id])
        dists_full = np.append(vnex_sorted.distances[mask], 1)

        n = len(dists_full) - 1
        ds_node_id[cur:cur + n] = tmp_vnex[1:]  # TODO: validate
        us_node_id[cur:cur + n] = tmp_vnex[:-1]
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

    # Don't apply to headwaters
    has_us = np.isin(links_dict.us_node_id, links_dict.ds_node_id)
    short_mask = short_mask & has_us

    # Make a store for rename mapping
    node_remapping = {}

    while short_mask.sum() > 0:
        short_idx = np.where(short_mask)[0]

        for idx in short_idx:
            us_node = links_dict.us_node_id[idx]
            ds_node = links_dict.ds_node_id[idx]
            length = links_dict.lengths[idx]

            us_filter = links_dict.ds_node_id == us_node
            if us_filter.sum() > 0:  # Merge with upstream
                links_dict.ds_node_id[us_filter] = ds_node
                links_dict.lengths[us_filter] += length

                node_remapping[us_node] = ds_node
            else:  # Retain
                short_mask[idx]

        links_dict.filter(~short_mask)
        short_mask = links_dict.lengths < discretization_len_m
        has_us = np.isin(links_dict.us_node_id, links_dict.ds_node_id)
        short_mask = short_mask & has_us
    
    # Clean remapping
    for k in list(node_remapping.keys()):
        v = node_remapping[k]
        if node_remapping[k] not in node_remapping:
            continue
        while v in node_remapping:
            v = node_remapping[v]
        node_remapping[k] = v

    return links_dict, node_remapping

def _discretize_links(links_dict: LinkArrays, discretization_len_m: float, cur_node_id: int = 0) -> LinkArrays:

    ## Subdivide to target length
    long_mask = links_dict.lengths > discretization_len_m

    if not np.any(long_mask):
        return

    # indices of long segments
    long_idx = np.where(long_mask)[0]

    subdiv_fp_id = []
    subdiv_start = []
    subdiv_end = []
    subdiv_ds_node = []
    subdiv_us_node = []
    subdiv_lengths = []
    # subdiv_id = []


    for idx in long_idx:
        n = int(np.ceil(links_dict.lengths[idx] / discretization_len_m))
        new_len = links_dict.lengths[idx] / n
        fracs = np.linspace(links_dict.start_frac[idx], links_dict.end_frac[idx], n + 1)
        new_node_ids = np.arange(cur_node_id, cur_node_id + n - 1, dtype=int)
        node_ids = np.concatenate(([links_dict.us_node_id[idx]], new_node_ids, [links_dict.ds_node_id[idx]]))

        cur_node_id += n - 1

        subdiv_us_node.append(node_ids[:-1])
        subdiv_ds_node.append(node_ids[1:])
        subdiv_fp_id.append(np.full(n, links_dict.fp_id[idx]))
        subdiv_start.append(fracs[:-1])
        subdiv_end.append(fracs[1:])
        subdiv_lengths.append(np.repeat(new_len, n))

    subdiv_fp_id = np.concatenate(subdiv_fp_id)
    subdiv_start = np.concatenate(subdiv_start)
    subdiv_end = np.concatenate(subdiv_end)
    # subdiv_id = np.concatenate(subdiv_id)
    subdiv_ds_node = np.concatenate(subdiv_ds_node)
    subdiv_us_node = np.concatenate(subdiv_us_node)
    subdiv_lengths = np.concatenate(subdiv_lengths)

    # mask out original long segments
    keep_mask = ~long_mask
    start_frac = np.concatenate([links_dict.start_frac[keep_mask], subdiv_start])
    end_frac = np.concatenate([links_dict.end_frac[keep_mask], subdiv_end])
    seg_fp_idx = np.concatenate([links_dict.fp_id[keep_mask], subdiv_fp_id])
    lengths = np.concatenate([links_dict.lengths[keep_mask], subdiv_lengths])
    # link_id = np.concatenate([link_id[keep_mask], subdiv_id])
    us_node_id = np.concatenate([links_dict.us_node_id[keep_mask], subdiv_us_node])
    ds_node_id = np.concatenate([links_dict.ds_node_id[keep_mask], subdiv_ds_node])

    return LinkArrays(
        seg_fp_idx,
        ds_node_id,
        us_node_id,
        start_frac,
        end_frac,
        lengths
    )


def _format_link_df(links: LinkArrays, flowpaths: pd.DataFrame) -> pd.DataFrame:
    channel_params = ["n", "mainstem_lp", "topwdth", "slope", "ncc", "btmwdth", "musx", "chslp", "topwdthcc", "musk"]
    segments = pd.merge(links.to_df(), flowpaths[channel_params + ["fp_id"]], on="fp_id", how="left")
    segments["alt"] = 0

    # Conform to abstractnetwork _dataframe
    renames = {
        "ds_node_id": "downstream",
        "length": "dx",
        "mainstem_lp": "mainstem",
        "topwdth": "tw",
        "slope": "s0",
        "btmwdth": "bw",
        "lengths": "dx",
        "chslp": "cs",
        "topwdthcc": "twcc"
    }
    # Summarize outlets for remapping t-route results to input flowpaths
    fp_outlet_crosswalk = segments.groupby("fp_id")["us_node_id"].max().rename_axis("fp_id").reset_index().set_index("us_node_id")["fp_id"].to_dict()


    segments = segments.set_index("us_node_id").rename(columns=renames)
    
    return segments, fp_outlet_crosswalk
