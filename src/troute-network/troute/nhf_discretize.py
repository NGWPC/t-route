import numpy as np
import pandas as pd
from shapely.geometry import Point


def discretize_flowpaths(
    flowpaths: pd.DataFrame,
    virtual_flowpaths: pd.DataFrame,
    virtual_nexus: pd.DataFrame,
    reference_flowpaths: pd.DataFrame,
    nexus: pd.DataFrame,
    discretization_len_m: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discretize flowpaths into links and nodes for MC routing.

    Terminal virtual nexuses (where VFPs meet the main channel) are
    preserved as special nodes in the link chain. These mark the points
    where area-scaled lateral flow enters the channel.

    Parameters
    ----------
    flowpaths : pd.DataFrame
        Physical flowpath data with columns: fp_id, div_id, dn_nex_id,
        length_km, and channel params (n, slope, btmwdth, topwdth, etc.)
    virtual_flowpaths : pd.DataFrame
        Virtual flowpath data with columns: virtual_fp_id, dn_virtual_nex_id
    virtual_nexus : pd.DataFrame
        Virtual nexus data with columns: virtual_nex_id, dn_virtual_fp_id
    reference_flowpaths : pd.DataFrame
        Crosswalk table with columns: fp_id, virtual_fp_id, div_id
    nexus : pd.DataFrame
        Regular nexus data with columns: nex_id, dn_fp_id
    discretization_len_m : float
        Target link length in meters (default 300m)

    Returns
    -------
    links_df : pd.DataFrame
        Index: link_id
        Columns: fp_id, div_id, dn_node_id, up_node_id,
                 length_km, and all channel params from flowpath
    nodes_df : pd.DataFrame
        Columns: node_id, dn_link_id, fp_id, is_terminal_nexus
    """
    target_length_km = discretization_len_m / 1000.0

    # Compute next_id above all existing IDs to avoid collisions
    id_pools = []
    if not flowpaths.empty:
        id_pools.append(flowpaths['fp_id'].max())
    if not nexus.empty:
        id_pools.append(nexus['nex_id'].max())
    if not virtual_flowpaths.empty:
        id_pools.append(virtual_flowpaths['virtual_fp_id'].max())
    if not virtual_nexus.empty:
        id_pools.append(virtual_nexus['virtual_nex_id'].max())
    next_id = int(max(id_pools)) + 1 if id_pools else 1

    # Build lookup: for each div_id, find terminal virtual nexus IDs
    # Terminal virtual nexuses are those with dn_virtual_fp_id == NaN
    if 'dn_virtual_fp_id' in virtual_nexus.columns:
        terminal_vnex_ids = set(
            virtual_nexus.loc[virtual_nexus['dn_virtual_fp_id'].isna(), 'virtual_nex_id']
        )
    else:
        # NHF 1.1.2+: all virtual nexuses are terminal (no VFP chains)
        terminal_vnex_ids = set(virtual_nexus['virtual_nex_id'])

    # Map div_id -> list of terminal virtual nexus IDs via reference_flowpaths + virtual_flowpaths
    vfp_refs = reference_flowpaths[reference_flowpaths['virtual_fp_id'].notna()].copy()
    if not vfp_refs.empty:
        vfp_refs['virtual_fp_id'] = vfp_refs['virtual_fp_id'].astype(int)
        # Merge to get dn_virtual_nex_id for each VFP
        vfp_with_nex = vfp_refs.merge(
            virtual_flowpaths[['virtual_fp_id', 'dn_virtual_nex_id']],
            on='virtual_fp_id',
            how='left',
        )
        # Filter to only terminal virtual nexuses
        vfp_terminal = vfp_with_nex[
            vfp_with_nex['dn_virtual_nex_id'].isin(terminal_vnex_ids)
        ]
        div_to_terminal_vnex = (
            vfp_terminal.groupby('div_id')['dn_virtual_nex_id']
            .apply(lambda x: sorted(x.dropna().astype(int).tolist()))
            .to_dict()
        )
    else:
        div_to_terminal_vnex = {}

    # Channel parameter columns to inherit from flowpath to links
    channel_params = [
        'n', 'slope', 'btmwdth', 'topwdth', 'ncc', 'topwdthcc',
        'musx', 'chslp', 'musk', 'mainstem_lp',
    ]
    # Only include columns that actually exist in flowpaths
    channel_params = [c for c in channel_params if c in flowpaths.columns]

    # Build geometry lookups for positioning terminal nexuses
    fp_geom_lookup = {}
    if 'geometry' in flowpaths.columns:
        for _, row in flowpaths.iterrows():
            geom = row['geometry']
            if geom is not None:
                fp_geom_lookup[int(row['fp_id'])] = (
                    geom.geoms[0] if geom.geom_type == 'MultiLineString' else geom
                )

    vnex_geom_lookup = {}
    if 'geometry' in virtual_nexus.columns:
        for _, row in virtual_nexus.iterrows():
            geom = row['geometry']
            if geom is not None:
                vnex_geom_lookup[int(row['virtual_nex_id'])] = geom

    nexus_geom_lookup = {}
    if 'geometry' in nexus.columns:
        for _, row in nexus.iterrows():
            geom = row['geometry']
            if geom is not None:
                nexus_geom_lookup[int(row['nex_id'])] = geom

    link_records = []
    node_records = []

    for _, fp in flowpaths.iterrows():
        fp_id = int(fp['fp_id'])
        div_id = int(fp['div_id'])
        length_km = fp['length_km']
        dn_nex_id = int(fp['dn_nex_id'])

        # Handle missing/zero length
        if pd.isna(length_km) or length_km <= 0:
            length_km = 0.0

        # Get terminal virtual nexus IDs for this divide
        tnex_ids = div_to_terminal_vnex.get(div_id, [])

        # Build channel param dict for this flowpath
        fp_params = {col: fp[col] for col in channel_params if col in fp.index}

        # Try geometry-aware placement
        fp_geom = fp_geom_lookup.get(fp_id)
        has_geometry = fp_geom is not None and tnex_ids

        if has_geometry:
            fp_geom_length_m = fp_geom.length

            # Project each terminal vnex onto the flowpath LineString
            vnex_positions = []
            for vnex_id in tnex_ids:
                vnex_point = vnex_geom_lookup.get(vnex_id)
                if vnex_point is not None:
                    dist_m = fp_geom.project(vnex_point)
                    vnex_positions.append((vnex_id, dist_m))

            # Determine direction: is the downstream nexus near the start or end?
            dn_nex_point = nexus_geom_lookup.get(dn_nex_id)
            if dn_nex_point is not None:
                dist_to_start = dn_nex_point.distance(Point(fp_geom.coords[0]))
                dist_to_end = dn_nex_point.distance(Point(fp_geom.coords[-1]))
                downstream_at_start = dist_to_start < dist_to_end
            else:
                downstream_at_start = True

            # Normalize projected distances to km
            for i, (vnex_id, dist_m) in enumerate(vnex_positions):
                fraction = dist_m / fp_geom_length_m if fp_geom_length_m > 0 else 0
                if downstream_at_start:
                    pos_km = fraction * length_km
                else:
                    pos_km = (1.0 - fraction) * length_km
                vnex_positions[i] = (vnex_id, pos_km)

            # Sort by position (downstream -> upstream)
            vnex_positions.sort(key=lambda x: x[1])

            # Build fixed-point list: downstream end + vnex positions + upstream end
            fixed_points = [(None, 0.0)]
            for vnex_id, pos_km in vnex_positions:
                pos_km = max(0.0, min(pos_km, length_km))
                fixed_points.append((vnex_id, pos_km))
            fixed_points.append((None, length_km))

            # Subdivide each segment into links
            dn_node_for_next_seg = dn_nex_id
            for seg_idx in range(len(fixed_points) - 1):
                seg_start_id, seg_start_km = fixed_points[seg_idx]
                seg_end_id, seg_end_km = fixed_points[seg_idx + 1]
                segment_length_km = seg_end_km - seg_start_km

                if segment_length_km <= 0:
                    continue

                n_sub = max(1, round(segment_length_km / target_length_km))
                sub_link_length = segment_length_km / n_sub

                for sub_idx in range(n_sub):
                    link_id = next_id; next_id += 1

                    # Determine up_node_id for this sub-link
                    if sub_idx == n_sub - 1:
                        if seg_end_id is not None:
                            # Terminal vnex node
                            up_node_id = seg_end_id
                            node_records.append({
                                'node_id': seg_end_id,
                                'fp_id': fp_id,
                                'is_terminal_nexus': True,
                                '_node_index_in_fp': len(node_records),
                            })
                        elif seg_idx == len(fixed_points) - 2:
                            # Upstream end of flowpath
                            up_node_id = None
                        else:
                            # Should not happen, but handle gracefully
                            up_node_id = None
                    else:
                        # Internal sub-link: create a new internal node
                        up_node_id = next_id; next_id += 1
                        node_records.append({
                            'node_id': up_node_id,
                            'fp_id': fp_id,
                            'is_terminal_nexus': False,
                            '_node_index_in_fp': len(node_records),
                        })

                    link_records.append({
                        'link_id': link_id,
                        'fp_id': fp_id,
                        'div_id': div_id,
                        'dn_node_id': dn_node_for_next_seg,
                        'up_node_id': up_node_id,
                        'length_km': sub_link_length,
                        **fp_params,
                    })
                    dn_node_for_next_seg = up_node_id

        else:
            # Fallback: equal-spacing (no geometry or no terminal nexuses)
            n_terminal_nexuses = len(tnex_ids)
            if length_km <= 0 or target_length_km <= 0:
                n_links = 1
            else:
                n_by_length = max(1, int(length_km / target_length_km))
                n_links = max(n_by_length, n_terminal_nexuses + 1) if n_terminal_nexuses > 0 else n_by_length

            link_length_km = length_km / n_links if n_links > 0 else length_km

            # Create internal nodes
            n_nodes = n_links - 1
            node_ids = []
            for i in range(n_nodes):
                if i < n_terminal_nexuses:
                    node_ids.append(int(tnex_ids[i]))
                else:
                    node_ids.append(next_id)
                    next_id += 1

            # Create link records (ordered downstream to upstream)
            for i in range(n_links):
                link_id = next_id; next_id += 1

                dn_node = dn_nex_id if i == 0 else node_ids[i - 1]
                up_node = node_ids[i] if i < n_nodes else None

                link_records.append({
                    'link_id': link_id,
                    'fp_id': fp_id,
                    'div_id': div_id,
                    'dn_node_id': dn_node,
                    'up_node_id': up_node,
                    'length_km': link_length_km,
                    **fp_params,
                })

            # Create node records
            for i, nid in enumerate(node_ids):
                node_records.append({
                    'node_id': nid,
                    'fp_id': fp_id,
                    'is_terminal_nexus': i < n_terminal_nexuses,
                    '_node_index_in_fp': i,
                })

    # Build DataFrames
    if link_records:
        links_df = pd.DataFrame(link_records).set_index('link_id')
    else:
        cols = ['fp_id', 'div_id', 'dn_node_id', 'up_node_id', 'length_km'] + channel_params
        links_df = pd.DataFrame(columns=cols)
        links_df.index.name = 'link_id'

    if node_records:
        nodes_df = pd.DataFrame(node_records)
        # Compute dn_link_id: for each node, find the link whose up_node_id == node_id
        up_node_to_link = links_df.reset_index().dropna(subset=['up_node_id'])
        up_node_to_link = dict(zip(
            up_node_to_link['up_node_id'].astype(int),
            up_node_to_link['link_id'],
        ))
        nodes_df['dn_link_id'] = nodes_df['node_id'].map(up_node_to_link)
        nodes_df = nodes_df.drop(columns=['_node_index_in_fp'])
    else:
        nodes_df = pd.DataFrame(columns=['node_id', 'dn_link_id', 'fp_id', 'is_terminal_nexus'])

    return links_df, nodes_df


def distribute_catchment_discharge(
    div_lateralflows_df: pd.DataFrame,
    vfp_dataframe: pd.DataFrame,
    links_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    reference_flowpaths: pd.DataFrame,
    fp_to_dn_nex: dict[int, int],
    nex_to_dn_fp: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Distribute catchment discharge to links for routing and to virtual
    flowpaths for flow-scaling output.

    VFP discharge and the un-VFP'd remainder are routed as upstream
    boundary flow (q_up) via the offnetwork_upstreams mechanism, not as
    distributed lateral inflow along the channel.

    - VFP discharge enters at the terminal virtual nexus node (targeting
      the downstream link within the same flowpath).
    - Remainder discharge enters at the downstream physical nexus
      (targeting the first link of the downstream flowpath).
    - Fallback: if no target link can be found (e.g. terminal flowpaths
      with no downstream, or VFPs without a mapped terminal nexus), the
      flow is distributed as lateral inflow (qlat) across all links.

    Parameters
    ----------
    div_lateralflows_df : pd.DataFrame
        Lateral flows indexed by div_id, columns are timestamps
    vfp_dataframe : pd.DataFrame
        Virtual flowpath DataFrame indexed by virtual_fp_id with columns:
        div_id, percentage_area_contribution, dn_virtual_nex_id
    links_df : pd.DataFrame
        Link DataFrame indexed by link_id with columns: fp_id, div_id
    nodes_df : pd.DataFrame
        Node DataFrame with columns: node_id, dn_link_id, fp_id, is_terminal_nexus
    reference_flowpaths : pd.DataFrame
        Crosswalk table with columns: fp_id, virtual_fp_id, div_id
    fp_to_dn_nex : dict[int, int]
        Mapping of flowpath ID -> downstream physical nexus ID
    nex_to_dn_fp : dict[int, int]
        Mapping of physical nexus ID -> downstream flowpath ID

    Returns
    -------
    routing_qlats : pd.DataFrame
        Qlats indexed by link_id, columns are timestamps (fallback only)
    flow_scaling_df : pd.DataFrame
        Qlats for VFPs (for non-routing flow-scaling output), indexed by virtual_fp_id
    upstream_inflow_df : pd.DataFrame
        Upstream inflow indexed by target link_id, columns are timestamps.
        Contains VFP and remainder discharge to be injected as q_up.
    """
    timestamps = div_lateralflows_df.columns

    # Initialize routing qlats with zeros for all links
    routing_qlats = pd.DataFrame(
        0.0,
        index=links_df.index,
        columns=timestamps,
    )

    # Initialize upstream inflow with zeros for all links
    upstream_inflow_df = pd.DataFrame(
        0.0,
        index=links_df.index,
        columns=timestamps,
    )

    # Build VFP info: div_id, percentage, and terminal nexus node
    vfp_info = vfp_dataframe[['div_id', 'percentage_area_contribution', 'dn_virtual_nex_id']].copy()
    vfp_info['div_id'] = vfp_info['div_id'].astype(int)

    # Map terminal nexus node_id -> dn_link_id (the link downstream of that node)
    tnex_to_link = {}
    if not nodes_df.empty:
        terminal_nodes = nodes_df[nodes_df['is_terminal_nexus']]
        if not terminal_nodes.empty:
            tnex_to_link = dict(zip(
                terminal_nodes['node_id'].astype(int),
                terminal_nodes['dn_link_id'].astype(int),
            ))

    # Map div_id -> fp_id
    # Use reference_flowpaths where fp_id IS NOT NULL
    fp_refs = reference_flowpaths[reference_flowpaths['fp_id'].notna()].copy()
    div_to_fp = dict(zip(fp_refs['div_id'].astype(int), fp_refs['fp_id'].astype(int)))

    # Map fp_id -> list of link_ids
    fp_to_links = links_df.reset_index().groupby('fp_id')['link_id'].apply(list).to_dict()

    # Map fp_id -> first (most-upstream) link_id
    fp_to_first_link = {}
    for lid, row in links_df[links_df['up_node_id'].isna()].iterrows():
        fp_to_first_link[int(row['fp_id'])] = lid

    # Compute flow_scaling_df (VFP-level qlats for non-routing output)
    flow_scaling_records = []

    # Process each divide
    for div_id in div_lateralflows_df.index:
        div_id_int = int(div_id)
        div_qlat = div_lateralflows_df.loc[div_id]  # Series: timestamps -> values

        fp_id = div_to_fp.get(div_id_int)
        if fp_id is None:
            continue
        link_ids = fp_to_links.get(fp_id, [])
        if not link_ids:
            continue

        n_links = len(link_ids)

        # Get VFPs for this divide
        div_vfps = vfp_info[vfp_info['div_id'] == div_id_int]
        sum_pct = div_vfps['percentage_area_contribution'].sum()

        # VFP-covered area: upstream inflow at terminal nexus
        for vfp_id, vfp_row in div_vfps.iterrows():
            pct = vfp_row['percentage_area_contribution']
            vfp_qlat = div_qlat * pct

            # Store for flow scaling output
            flow_scaling_records.append(
                pd.Series(vfp_qlat.values, index=timestamps, name=vfp_id)
            )

            # Route as upstream inflow at terminal nexus node
            dn_vnex = vfp_row['dn_virtual_nex_id']
            if pd.notna(dn_vnex) and int(dn_vnex) in tnex_to_link:
                target_link = tnex_to_link[int(dn_vnex)]
                upstream_inflow_df.loc[target_link] += vfp_qlat.values
            else:
                # Fallback: no mapped terminal nexus -> distribute as qlat
                for lid in link_ids:
                    routing_qlats.loc[lid] += vfp_qlat.values / n_links

        # Remainder: upstream inflow at downstream physical nexus
        remainder_pct = 1.0 - sum_pct
        if remainder_pct > 1e-10:
            remainder_qlat = div_qlat * remainder_pct
            dn_nex = fp_to_dn_nex.get(fp_id)
            dn_fp = nex_to_dn_fp.get(dn_nex) if dn_nex else None
            target_link = fp_to_first_link.get(dn_fp) if dn_fp else None
            if target_link is not None:
                upstream_inflow_df.loc[target_link] += remainder_qlat.values
            else:
                # Terminal flowpath (no downstream): fall back to qlat
                per_link = remainder_qlat.values / n_links
                for lid in link_ids:
                    routing_qlats.loc[lid] += per_link

    # Build flow_scaling_df
    if flow_scaling_records:
        flow_scaling_df = pd.DataFrame(flow_scaling_records)
        flow_scaling_df.index.name = 'virtual_fp_id'
    else:
        flow_scaling_df = pd.DataFrame(columns=timestamps)
        flow_scaling_df.index.name = 'virtual_fp_id'

    return routing_qlats, flow_scaling_df, upstream_inflow_df
