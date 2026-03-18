"""Discretize flowpath network into uniform link segments.

This module
    1. Identifies on-flowpath virtual flowpaths (to serve as basis for routing links)
    2. (optionally) Aggregates short links into upstream neighbors
    3. Subdivides long links into near-uniform lengths

Notable Assumptions
---------------
- Node IDs increase in the downstream direction
- Flowpaths form a directed acyclic graph (no loops)

"""

from dataclasses import dataclass, fields
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely import MultiLineString, line_interpolate_point
import numpy as np
import shapely
from shapely.ops import substring

### NHF SCHEMA ###
# Pulling this to the top for easy updates later
FIELD_FP_ID = "fp_id"
FIELD_VIRTUAL_FP_ID = "virtual_fp_id"
FIELD_DN_VIRTUAL_NEX_ID = "dn_virtual_nex_id"
FIELD_UP_VIRTUAL_NEX_ID = "up_virtual_nex_id"
FIELD_LENGTH = "length_km"
FIELD_LENGTH_CONVERSION = 1000
CHANNEL_PARAMS = [
    "n",
    "mainstem_lp",
    "topwdth",
    "slope",
    "ncc",
    "btmwdth",
    "musx",
    "chslp",
    "topwdthcc",
    "musk",
]

### DATA CLASSES ###


@dataclass
class LinkArrays:
    """Routing link data structure.

    Notes
    -----
     - Arrays are used because they are more performant than Pandas dfs and scale better.

    """

    fp_id: np.ndarray
    dn_node_id: np.ndarray
    up_node_id: np.ndarray
    length: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """Load LinksArrays from a virtual flowpaths layer."""
        return cls(
            df[FIELD_FP_ID].to_numpy().astype(int),
            df[FIELD_DN_VIRTUAL_NEX_ID].to_numpy().astype(int),
            df[FIELD_UP_VIRTUAL_NEX_ID].to_numpy().astype(int),
            df[FIELD_LENGTH].to_numpy() * FIELD_LENGTH_CONVERSION,
        )

    def get_short_mask(self, discretization_len_m: float) -> np.ndarray:
        """Identify links shorter than discretization threshold and eligible for merging."""
        short_mask = self.length < discretization_len_m

        # Don't apply to headwaters, because they cannot be merged with upstream
        has_us = np.isin(self.up_node_id, self.dn_node_id)

        return short_mask & has_us

    def to_df(self) -> pd.DataFrame:
        """Convert LinkArrays to a Pandas DataFrame."""
        return pd.DataFrame({f.name: getattr(self, f.name) for f in fields(self)})

    def filter(self, mask: np.ndarray) -> None:
        """Keep masked links."""
        self.fp_id = self.fp_id[mask]
        self.dn_node_id = self.dn_node_id[mask]
        self.up_node_id = self.up_node_id[mask]
        self.length = self.length[mask]


### MAIN FUNCTION ###


def discretize_flowpaths(
    flowpaths: gpd.GeoDataFrame,
    virtual_flowpaths: gpd.GeoDataFrame,
    reference_flowpaths: pd.DataFrame,
    discretization_len_m: float = 300.0,
    aggregate_short_reaches: bool = True,
    export_links_nodes_gpkg_path: Union[None, str] = None,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Discretize flowpaths into uniform-length links and resolve short reaches.

    Parameters
    ----------
    flowpaths : gpd.GeoDataFrame
        Must contain:
        - FIELD_FP_ID
        - CHANNEL_PARAMS columns
        - geometry
    virtual_flowpaths : gpd.GeoDataFrame
        Must contain:
        - FIELD_VIRTUAL_FP_ID
        - FIELD_UP_VIRTUAL_NEX_ID
        - FIELD_DN_VIRTUAL_NEX_ID
        - geometry
    reference_flowpaths : pd.DataFrame
        Must contain:
        - FIELD_FP_ID
        - FIELD_VIRTUAL_FP_ID
        - div_id
    discretization_len_m : float, default 300.0
        Target link length in meters.
    aggregate_short_reaches : bool, default True
        Whether to merge short links upstream before discretization.
    export_links_nodes_gpkg_path : str or None
        If provided, exports links and nodes to GeoPackage.

    Returns
    -------
    tuple
        (
            pd.DataFrame,  # formatted link table that matches _dataframe format from AbstractNetwork
            dict[int, int] # mapping from link id (up_node_id) back to fp_id used to remap flowpath outflows to original fp_id
            dict[int, int] # mapping of virtual_nexus_id to a new node when the original node was merged
        )

    """
    virtual_flowpaths = _tmp_find_routing_vfps(
        virtual_flowpaths, flowpaths, reference_flowpaths
    )
    cur_node_id = virtual_flowpaths[FIELD_UP_VIRTUAL_NEX_ID].max() + 1

    links = LinkArrays.from_df(
        pd.merge(
            virtual_flowpaths,
            reference_flowpaths[[FIELD_FP_ID, FIELD_VIRTUAL_FP_ID]].drop_duplicates(),
            on=FIELD_VIRTUAL_FP_ID,
        )
    )
    if aggregate_short_reaches:
        links, merged_node_crosswalk = _aggregate_links(links, discretization_len_m)
    else:
        merged_node_crosswalk = {}
    links = _discretize_links(links, discretization_len_m, cur_node_id)

    if export_links_nodes_gpkg_path is not None:
        export_links_and_nodes(links, virtual_flowpaths, export_links_nodes_gpkg_path)
    return *_format_link_df(links, flowpaths), merged_node_crosswalk


### HELPER FUNCTIONS ###


def export_links_and_nodes(
    links: LinkArrays,
    virtual_flowpaths: gpd.GeoDataFrame,
    export_links_nodes_gpkg_path: str,
) -> None:
    """Export discretized links and nodes as GeoPackage layers (for debugging)."""
    # Initialize data stores
    up_node_ids = []
    dn_node_ids = []
    link_geometries = []
    node_geometries = []

    # Make network graphs
    dn_network = dict(zip(links.up_node_id, links.dn_node_id))
    dn_network[set(links.dn_node_id).difference(links.up_node_id).pop()] = -9999

    len_lookup = dict(zip(links.up_node_id, links.length))

    # Processing loop
    virtual_flowpaths = virtual_flowpaths.set_index(FIELD_UP_VIRTUAL_NEX_ID)
    non_eclipsed = virtual_flowpaths[virtual_flowpaths.index.isin(links.up_node_id)]
    for i in non_eclipsed.itertuples():
        up_node = int(i.Index)
        dn_node = i.dn_virtual_nex_id

        # Walk downstream through missing (eclipsed) nodes, merging geometries until a retained node is found
        geoms = [i.geometry]
        while dn_node not in links.dn_node_id:
            tmp = virtual_flowpaths.loc[dn_node]
            geoms.append(tmp.geometry)
            dn_node = tmp.dn_virtual_nex_id
        geom = shapely.line_merge(MultiLineString([j.geoms[0] for j in geoms[::-1]]))

        # Reconstruct link chain between retained nodes using dn_network lookup
        link_up = up_node
        link_dn = dn_network[link_up]
        tmp_links = [up_node]
        length = [len_lookup[link_up]]
        while link_dn != dn_node:
            tmp_links.append(link_dn)
            length.append(len_lookup[link_up])
            link_dn = dn_network[link_dn]

        # Build cumulative distances along merged geometry to segment into individual links
        cumdist = np.cumsum([0.0] + length)
        cumdist[-1] = geom.length  # just in case
        link_geoms = [
            substring(geom, start_dist, end_dist)
            for start_dist, end_dist in zip(cumdist[:-1], cumdist[1:])
        ]
        node_geoms = [
            line_interpolate_point(geom, start_dist) for start_dist in cumdist[:-1]
        ]

        # Log
        up_node_ids.extend(tmp_links)
        dn_node_ids.extend(tmp_links[1:] + [dn_node])
        link_geometries.extend(link_geoms)
        node_geometries.extend(node_geoms)

    # Export geopackage
    gpd.GeoDataFrame(
        {"up_node_id": up_node_ids, "dn_node_id": dn_node_ids},
        geometry=link_geometries,
        crs=virtual_flowpaths.crs,
    ).to_file(export_links_nodes_gpkg_path, layer="links")
    gpd.GeoDataFrame(
        {"node_id": up_node_ids}, geometry=node_geometries, crs=virtual_flowpaths.crs
    ).to_file(export_links_nodes_gpkg_path, layer="nodes")


def _tmp_find_routing_vfps(
    virtual_flowpaths: gpd.GeoDataFrame,
    flowpaths: gpd.GeoDataFrame,
    reference_flowpaths: pd.DataFrame,
) -> gpd.GeoDataFrame:
    ### TODO: REMOVE THIS AFTER NHF UPDATE
    vfp = virtual_flowpaths.rename(columns={"geometry": "vfp_geom"})
    fp = flowpaths.rename(columns={"geometry": "fp_geom"})

    vfp_to_fp = pd.merge(
        vfp,
        reference_flowpaths[[FIELD_VIRTUAL_FP_ID, "div_id"]].drop_duplicates(),
        how="left",
        on=FIELD_VIRTUAL_FP_ID,
    )[[FIELD_VIRTUAL_FP_ID, "vfp_geom", "div_id"]]

    vfp_to_fp = pd.merge(
        vfp_to_fp,
        reference_flowpaths[["div_id", FIELD_FP_ID]].dropna().drop_duplicates(),
        how="left",
        on="div_id",
    )[[FIELD_VIRTUAL_FP_ID, "vfp_geom", FIELD_FP_ID]]

    vfp_to_fp = pd.merge(
        vfp_to_fp,
        fp[[FIELD_FP_ID, "fp_geom"]],
        how="left",
        on=FIELD_FP_ID,
    )

    # Check if first point of vfp matches first point of fp
    vfp_to_fp["vfp_geom"] = shapely.get_point(
        shapely.get_geometry(vfp_to_fp["vfp_geom"], 0).values, 0
    )
    vfp_to_fp["fp_geom"] = shapely.get_point(
        shapely.get_geometry(vfp_to_fp["fp_geom"], 0).values, 0
    )

    matches = shapely.equals(vfp_to_fp["vfp_geom"], vfp_to_fp["fp_geom"])

    # Force new virtual nexus at top
    matches = matches & virtual_flowpaths[FIELD_UP_VIRTUAL_NEX_ID].isna()
    start_id = virtual_flowpaths[FIELD_UP_VIRTUAL_NEX_ID].max() + 1
    new_ids = np.arange(start_id, start_id + matches.sum())
    virtual_flowpaths.loc[matches, FIELD_UP_VIRTUAL_NEX_ID] = new_ids

    virtual_flowpaths = virtual_flowpaths.dropna(subset=FIELD_UP_VIRTUAL_NEX_ID)
    virtual_flowpaths[FIELD_UP_VIRTUAL_NEX_ID] = virtual_flowpaths[
        FIELD_UP_VIRTUAL_NEX_ID
    ].astype(int)
    return virtual_flowpaths


def _aggregate_links(
    links: LinkArrays, discretization_len_m: float
) -> tuple[LinkArrays, dict[int, int]]:
    """Merge links shorter than threshold into upstream neighbors.

    Notes
    -----
    - Iteratively merges short links until all satisfy threshold or are headwaters.
    - Node remapping provides lookup for removed nodes to their new terminal node.

    """
    ## Combine links that are below discretization length with the next upstream reach
    short_mask = links.get_short_mask(discretization_len_m)

    # Make a store for rename mapping.  This is used so Qlats can bew mapped to the new dn node
    merged_node_crosswalk: dict[int, int] = {}

    while short_mask.sum() > 0:
        short_idx = np.where(short_mask)[0]

        for idx in short_idx:
            up_node = links.up_node_id[idx]
            dn_node = links.dn_node_id[idx]
            length = links.length[idx]

            # Merge short links into immediate upstream link if upstream exists
            us_filter = links.dn_node_id == up_node
            if us_filter.sum() > 0:  # Merge with upstream
                links.dn_node_id[us_filter] = dn_node
                links.length[us_filter] += length

                merged_node_crosswalk[up_node] = dn_node
            else:  # Retain
                short_mask[idx]

        links.filter(~short_mask)
        short_mask = links.get_short_mask(discretization_len_m)

    # Clean remapping so each node maps directly to its final downstream node
    for k in list(merged_node_crosswalk.keys()):
        v = merged_node_crosswalk[k]
        if merged_node_crosswalk[k] not in merged_node_crosswalk:
            continue
        while v in merged_node_crosswalk:
            v = merged_node_crosswalk[v]
        merged_node_crosswalk[k] = v

    return links, merged_node_crosswalk


def _discretize_links(
    links: LinkArrays, discretization_len_m: float, cur_node_id: int = 0
) -> LinkArrays:
    """Subdivide links longer than threshold into uniform segments (uses ceil such that lengths will undershoot target)."""
    ## Subdivide to target length
    long_mask = links.length > discretization_len_m

    if not np.any(long_mask):
        return links

    # indices of long links
    long_idx = np.where(long_mask)[0]

    subdiv_fp_id = []
    subdiv_dn_node = []
    subdiv_up_node = []
    subdiv_length = []

    # Split long links into equal-length segments and insert new intermediate node IDs
    for idx in long_idx:
        n = int(np.ceil(links.length[idx] / discretization_len_m))
        new_len = links.length[idx] / n
        new_node_ids = np.arange(cur_node_id, cur_node_id + n - 1, dtype=int)
        node_ids = np.concatenate(
            ([links.up_node_id[idx]], new_node_ids, [links.dn_node_id[idx]])
        )

        cur_node_id += n - 1

        subdiv_up_node.append(node_ids[:-1])
        subdiv_dn_node.append(node_ids[1:])
        subdiv_fp_id.append(np.full(n, links.fp_id[idx]))
        subdiv_length.append(np.full(n, new_len))

    subdiv_fp_id = np.concatenate(subdiv_fp_id)
    subdiv_dn_node = np.concatenate(subdiv_dn_node)
    subdiv_up_node = np.concatenate(subdiv_up_node)
    subdiv_length = np.concatenate(subdiv_length)

    # mask out original long links
    keep_mask = ~long_mask
    link_fp_id = np.concatenate([links.fp_id[keep_mask], subdiv_fp_id])
    length = np.concatenate([links.length[keep_mask], subdiv_length])
    up_node_id = np.concatenate([links.up_node_id[keep_mask], subdiv_up_node])
    dn_node_id = np.concatenate([links.dn_node_id[keep_mask], subdiv_dn_node])

    return LinkArrays(link_fp_id, dn_node_id, up_node_id, length)


def _format_link_df(
    links: LinkArrays, flowpaths: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Conform to AbstractNetwork format and build mapping from link id to fp_id."""
    _dataframe = pd.merge(
        links.to_df(),
        flowpaths[CHANNEL_PARAMS + [FIELD_FP_ID]],
        on=FIELD_FP_ID,
        how="left",
    )

    # Summarize outlets for remapping t-route results to input flowpaths
    # Assumes that nexus and node IDs increase in downstream direction
    fp_outlet_crosswalk = (
        _dataframe.groupby(FIELD_FP_ID)["up_node_id"]
        .max()
        .rename_axis(FIELD_FP_ID)
        .reset_index()
        .set_index("up_node_id")[FIELD_FP_ID]
        .to_dict()
    )

    # Conform to abstractnetwork _dataframe
    _dataframe["alt"] = 0
    renames = {
        "dn_node_id": "downstream",
        "length": "dx",
        "mainstem_lp": "mainstem",
        "topwdth": "tw",
        "slope": "s0",
        "btmwdth": "bw",
        "chslp": "cs",
        "topwdthcc": "twcc",
    }
    _dataframe = _dataframe.set_index("up_node_id").rename(columns=renames)

    return _dataframe, fp_outlet_crosswalk
