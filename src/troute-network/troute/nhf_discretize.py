
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely import LineString
from shapely.geometry import Point
import numpy as np
import shapely
from pathlib import Path

def discretize_flowpaths(
    flowpaths: gpd.GeoDataFrame,
    virtual_flowpaths: pd.DataFrame,
    virtual_nexus: gpd.GeoDataFrame,
    reference_flowpaths: pd.DataFrame,
    nexus: gpd.GeoDataFrame,
    discretization_len_m: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Perform joins to prep data
    vfp_to_div = reference_flowpaths.dropna(subset="virtual_fp_id").set_index("virtual_fp_id")["div_id"].to_dict()
    div_to_fp = reference_flowpaths.dropna(subset=["fp_id"]).set_index("div_id")["fp_id"].astype(int).to_dict()
    vnex_to_fp_id = virtual_flowpaths[["dn_virtual_nex_id", "virtual_fp_id"]].copy()
    vnex_to_fp_id["div_id"] = vnex_to_fp_id["virtual_fp_id"].map(vfp_to_div)
    vnex_to_fp_id["fp_id"] = vnex_to_fp_id["div_id"].map(div_to_fp)
    virtual_nexus = virtual_nexus.merge(vnex_to_fp_id[["dn_virtual_nex_id", "fp_id"]], left_on="virtual_nex_id", right_on="dn_virtual_nex_id", how="left")
    vnex_ids = virtual_nexus["virtual_nex_id"].to_numpy()

    graph_base = flowpaths[["fp_id", "dn_nex_id", "up_nex_id"]].set_index("fp_id")
    fp_dn_nex_dict = graph_base["dn_nex_id"].to_dict()
    # Add new nodes for u/s ends of flowpaths
    headwater_mask = graph_base["up_nex_id"].isnull()
    max_id = max([flowpaths["dn_nex_id"].max(), virtual_flowpaths["dn_virtual_nex_id"].max()]) + 1
    graph_base.loc[headwater_mask, "up_nex_id"] = np.arange(max_id, max_id + headwater_mask.sum())
    fp_up_nex_dict = graph_base["up_nex_id"].astype(int).to_dict()

    # Extract necessary data, pulling out into arrays for efficiency
    flow_geom = flowpaths.geometry.values
    flow_lengths = flowpaths["length_km"].to_numpy() * 1000
    fp_ids = flowpaths["fp_id"].to_numpy()
    dn_nex_ids = flowpaths["dn_nex_id"].to_numpy()

    vnex_geom = virtual_nexus.geometry.values
    vnex_fp_ids = virtual_nexus["fp_id"].to_numpy()

    nex_lookup = nexus.set_index("nex_id").geometry
    dn_nex_geom = nex_lookup.loc[dn_nex_ids].values

    # Correct any backwards flowpaths
    start_pts = shapely.get_point(flow_geom, 0)
    end_pts = shapely.get_point(flow_geom, -1)

    dist_start = shapely.distance(start_pts, dn_nex_geom)
    dist_end = shapely.distance(end_pts, dn_nex_geom)

    reverse_mask = dist_start < dist_end
    flow_geom = np.where(reverse_mask, shapely.reverse(flow_geom), flow_geom)

    ## Get distance along each flowpath of each line

    # Indexing to get flowpath line for each virtual nexus
    fp_index_map = {fp: i for i, fp in enumerate(fp_ids)}  # map fp_id to index in flowpath layers
    vnex_flow_ind = np.array([fp_index_map[x] for x in vnex_fp_ids])  # fp_id indices corresponding to each virtual nexus

    # Compute fraction along flowpath for each virtual nexus
    dist = shapely.line_locate_point(flow_geom[vnex_flow_ind], vnex_geom)
    frac = dist / flow_lengths[vnex_flow_ind]

    ## Construct segment lengths array

    # Count number of virtual nexus per flowpath
    counts = np.bincount(vnex_flow_ind, minlength=len(flow_geom))
    seg_counts = counts + 1  # N+2 breakpoints (including ends); N+1 segments

    # Offsets for flattened arrays
    offsets = np.zeros_like(seg_counts)
    offsets[1:] = np.cumsum(seg_counts[:-1])
    total_segments = offsets[-1] + seg_counts[-1]

    # Prepare arrays
    link_id = np.arange(total_segments)
    start_frac = np.zeros(total_segments, dtype=np.float64)
    end_frac = np.zeros(total_segments, dtype=np.float64)
    seg_fp_idx = np.zeros(total_segments, dtype=np.int64)
    ds_node_id = np.empty(total_segments, dtype=np.int64)
    us_node_id = np.empty(total_segments, dtype=np.int64)

    # Sort fractions per flowpath. frac is potentially unsorted
    vnex_sorted_idx = np.argsort(vnex_flow_ind + frac * 1e-9)  # tiny factor to break ties
    vnex_id_sorted = vnex_ids[vnex_sorted_idx]
    vnex_dist_sorted = frac[vnex_sorted_idx]  # virtual nexus fractions ordered by ocurrence along flowpath
    vnex_fp_sorted = vnex_flow_ind[vnex_sorted_idx]  # flowpath index for each virtual nexus

    # Initialize dict to store link to flowpath outlet mapping
    fp_outlet_crosswalk = {}

    # Attribute links
    cur = 0

    for i in range(len(flow_geom)):
        # get fractions for this flowpath
        mask = vnex_fp_sorted == i
        tmp_vnex = vnex_id_sorted[mask]
        internal_dists = vnex_dist_sorted[mask]
        dists_full = np.concatenate(([0.0], internal_dists, [1.0]))

        n = len(dists_full) - 1
        tmp_fp_id = fp_ids[i]
        ds_node_id[cur:cur + n] = np.concatenate((tmp_vnex, [fp_dn_nex_dict[tmp_fp_id]]))
        us_node_id[cur:cur + n] = np.concatenate(([fp_up_nex_dict[tmp_fp_id]], tmp_vnex))
        start_frac[cur:cur + n] = dists_full[:-1]
        end_frac[cur:cur + n] = dists_full[1:]
        seg_fp_idx[cur:cur + n] = i

        fp_outlet_crosswalk[us_node_id[cur + n - 1]] = tmp_fp_id

        cur += n

    ## Subdivide to target length
    lengths = (end_frac - start_frac) * flow_lengths[seg_fp_idx]
    long_mask = lengths > discretization_len_m

    if np.any(long_mask):
        # indices of long segments
        long_idx = np.where(long_mask)[0]

        subdiv_fp = []
        subdiv_start = []
        subdiv_end = []
        subdiv_ds_node = []
        subdiv_us_node = []
        subdiv_id = []

        cur_node_id = us_node_id.max() + 1  # Create indices above this
        cur_link_id = total_segments + 1

        for idx in long_idx:
            n = int(np.ceil(lengths[idx] / discretization_len_m))
            fracs = np.linspace(start_frac[idx], end_frac[idx], n + 1)
            new_node_ids = np.arange(cur_node_id, cur_node_id + n - 1)
            node_ids = np.concatenate(([us_node_id[idx]], new_node_ids, [ds_node_id[idx]]))
            link_ids = np.arange(cur_link_id, cur_link_id + n)

            cur_node_id += n - 1
            cur_link_id += n

            subdiv_id.append(link_ids)
            subdiv_us_node.append(node_ids[:-1])
            subdiv_ds_node.append(node_ids[1:])
            subdiv_fp.append(np.full(n, seg_fp_idx[idx]))
            subdiv_start.append(fracs[:-1])
            subdiv_end.append(fracs[1:])

            if node_ids[0] in fp_outlet_crosswalk:
                tmp_fp_id = fp_outlet_crosswalk.pop(node_ids[0])
                fp_outlet_crosswalk[node_ids[-2]] = tmp_fp_id

        subdiv_fp = np.concatenate(subdiv_fp)
        subdiv_start = np.concatenate(subdiv_start)
        subdiv_end = np.concatenate(subdiv_end)
        subdiv_id = np.concatenate(subdiv_id)
        subdiv_ds_node = np.concatenate(subdiv_ds_node)
        subdiv_us_node = np.concatenate(subdiv_us_node)

        subdiv_lengths = (subdiv_end - subdiv_start) * flow_lengths[subdiv_fp]

        # mask out original long segments
        keep_mask = ~long_mask
        start_frac = np.concatenate([start_frac[keep_mask], subdiv_start])
        end_frac = np.concatenate([end_frac[keep_mask], subdiv_end])
        seg_fp_idx = np.concatenate([seg_fp_idx[keep_mask], subdiv_fp])
        lengths = np.concatenate([lengths[keep_mask], subdiv_lengths])
        link_id = np.concatenate([link_id[keep_mask], subdiv_id])
        us_node_id = np.concatenate([us_node_id[keep_mask], subdiv_us_node])
        ds_node_id = np.concatenate([ds_node_id[keep_mask], subdiv_ds_node])

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
