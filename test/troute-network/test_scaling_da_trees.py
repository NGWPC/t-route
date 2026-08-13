"""Unit tests for the simple-scaling upstream-tree builder on synthetic topologies.

Ported from the standalone proof-of-concept. The Ohio-gpkg integration tests
are deferred to the stage that wires :func:`build_gage_trees` onto t-route's
real network topology; here we cover :func:`build_one_gage_tree` against
synthetic reverse-connectivity graphs.
"""

from __future__ import annotations

import pytest

from troute.scaling_da import build_one_gage_tree

# Per-segment areas (in km^2) used across the synthetic graphs.
AREAS_CHAIN = {1: 100.0, 2: 80.0, 3: 60.0, 4: 40.0, 5: 20.0}
AREAS_JUNC = {1: 100.0, 2: 100.0, 3: 60.0, 4: 40.0}


def test_single_segment_tree() -> None:
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={},
        area_sqkm={1: 100.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    assert tree.n_segments == 1
    assert tree.gage_fp == 1
    assert tree.gage_area_sqkm == 100.0
    assert tree.seg_order.tolist() == [1]
    assert tree.seg_areas_sqkm.tolist() == [100.0]
    assert tree.theta == 0.77


def test_linear_chain_tree() -> None:
    # 5 -> 4 -> 3 -> 2 -> 1 (gage)
    rconn = {1: [2], 2: [3], 3: [4], 4: [5]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm=AREAS_CHAIN,
        stop_segs=frozenset(),
        theta=0.77,
    )
    assert tree.n_segments == 5
    assert tree.seg_order.tolist() == [1, 2, 3, 4, 5]
    # Per-segment areas follow seg_order.
    assert tree.seg_areas_sqkm.tolist() == [100.0, 80.0, 60.0, 40.0, 20.0]


def test_single_junction_tree() -> None:
    # 3 and 4 merge at 2, then 2 -> 1 (gage)
    rconn = {1: [2], 2: [3, 4]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm=AREAS_JUNC,
        stop_segs=frozenset(),
        theta=0.77,
    )
    assert tree.n_segments == 4
    # BFS: gage 1, then 2, then its upstream pair 3 and 4.
    assert tree.seg_order.tolist() == [1, 2, 3, 4]


def test_two_junctions_in_series() -> None:
    # 5,6 -> 4 -> 3 (gage), and 7 -> 3
    rconn = {3: [4, 7], 4: [5, 6]}
    areas = {3: 100.0, 4: 60.0, 5: 30.0, 6: 30.0, 7: 40.0}
    tree = build_one_gage_tree(
        gage_fp=3,
        rconn=rconn,
        area_sqkm=areas,
        stop_segs=frozenset(),
        theta=0.5,
    )
    assert tree.n_segments == 5
    # BFS order: 3 first, then 4 and 7 (children of 3), then 5 and 6 (children of 4).
    assert tree.seg_order.tolist() == [3, 4, 7, 5, 6]


def test_junction_structure_on_chain() -> None:
    """A linear chain has no confluences: every step is an area-scaling step."""
    rconn = {1: [2], 2: [3], 3: [4], 4: [5]}
    tree = build_one_gage_tree(
        gage_fp=1, rconn=rconn, area_sqkm=AREAS_CHAIN, stop_segs=frozenset(), theta=0.77
    )
    assert tree.seg_parent_idx.tolist() == [-1, 0, 1, 2, 3]
    assert tree.step_is_junction.tolist() == [False, False, False, False, False]


def test_junction_structure_two_junctions() -> None:
    """Parent index and confluence flags on a two-junction topology."""
    # 5,6 -> 4 -> 3 (gage), and 7 -> 3.
    rconn = {3: [4, 7], 4: [5, 6]}
    areas = {3: 100.0, 4: 60.0, 5: 30.0, 6: 30.0, 7: 40.0}
    tree = build_one_gage_tree(
        gage_fp=3, rconn=rconn, area_sqkm=areas, stop_segs=frozenset(), theta=0.5
    )
    # seg_order = [3, 4, 7, 5, 6]; each parent points to the downstream index.
    assert tree.seg_parent_idx.tolist() == [-1, 0, 0, 1, 1]
    # Gage 3 (children 4, 7) and node 4 (children 5, 6) are both confluences, so
    # their children take the flow-ratio split; the gage's own entry is False.
    assert tree.step_is_junction.tolist() == [False, True, True, True, True]


def test_junction_with_stopped_branch_still_flagged() -> None:
    """A physical confluence is flagged even when one branch is pruned by the stop rule."""
    # 2 and 3 merge at the gage 1; branch 3 is stopped (e.g. an upstream gage).
    rconn = {1: [2, 3]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm={1: 100.0, 2: 60.0, 3: 40.0},
        stop_segs=frozenset({3}),
        theta=0.77,
    )
    assert tree.seg_order.tolist() == [1, 2]  # 3 pruned
    # Node 1 is still a physical confluence (rconn[1] = [2, 3]), so the surviving
    # branch 2 takes the flow-ratio split, not area-scaling.
    assert tree.step_is_junction.tolist() == [False, True]


def test_stop_at_upstream_gage() -> None:
    # 5 -> 4 -> 3 (gage1), with gage2 at fp 5
    rconn = {3: [4], 4: [5]}
    tree = build_one_gage_tree(
        gage_fp=3,
        rconn=rconn,
        area_sqkm={3: 100.0, 4: 80.0, 5: 50.0},
        stop_segs=frozenset({5}),
        theta=0.77,
    )
    assert tree.n_segments == 2
    assert tree.seg_order.tolist() == [3, 4]
    # 5 is the stop -> not in the tree.
    assert 5 not in tree.seg_order.tolist()


def test_stop_at_waterbody() -> None:
    # 4 -> 3 (gage), with 4 inside a waterbody
    rconn = {3: [4]}
    tree = build_one_gage_tree(
        gage_fp=3,
        rconn=rconn,
        area_sqkm={3: 100.0, 4: 80.0},
        stop_segs=frozenset({4}),
        theta=0.77,
    )
    assert tree.n_segments == 1
    assert tree.seg_order.tolist() == [3]


def test_gage_in_stop_segs_raises() -> None:
    with pytest.raises(ValueError, match="cannot be in stop_segs"):
        build_one_gage_tree(
            gage_fp=1,
            rconn={},
            area_sqkm={1: 100.0},
            stop_segs=frozenset({1}),
            theta=0.77,
        )


def test_bfs_visits_each_segment_once() -> None:
    """BFS in a tree must not double-visit segments even if the rconn entry duplicates them."""
    # Intentional duplicate upstream, defensive check.
    rconn = {1: [2, 2], 2: [3]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm={1: 1.0, 2: 1.0, 3: 1.0},
        stop_segs=frozenset(),
        theta=0.5,
    )
    # Should produce a normal chain, not two copies of 2.
    assert tree.seg_order.tolist() == [1, 2, 3]


def test_with_positions_maps_seg_order_correctly() -> None:
    tree = build_one_gage_tree(
        gage_fp=10,
        rconn={10: [20, 30]},
        area_sqkm={10: 5.0, 20: 3.0, 30: 2.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    fp_to_position = {10: 100, 20: 200, 30: 300, 99: 999}
    out = tree.with_positions(fp_to_position)
    assert out.seg_positions.tolist() == [100, 200, 300]
    # with_positions must preserve the junction structure (10 is a confluence).
    assert out.seg_parent_idx.tolist() == [-1, 0, 0]
    assert out.step_is_junction.tolist() == [False, True, True]


def test_tree_with_unmapped_reservoir_segment_is_dropped() -> None:
    """A tree that walks onto a segment with no drainage area must not be kept.

    On NHF the synthetic reservoir ids live in ``connections``/``reverse_network`` but
    never get a ``dataframe`` row, so their area lookup misses and ``build_one_gage_tree``
    fills NaN. The area-scaling step then multiplies NaN through the whole subtree, and
    once the spread seeds state, that NaN reaches the warmstate and routes.

    The Ohio benchmark subset has zero rows in its ``lakes`` layer, so no integration
    run can exercise this path; this is the regression test that can.
    """
    from troute.scaling_da import build_gage_trees_from_mappings

    # head(30) -> lake(20) -> gage(10). The lake carries no drainage area.
    rconn = {10: [20], 20: [30], 30: []}
    area = {10: 100.0, 30: 50.0}  # note: no key for the lake id 20

    # Without a waterbody stop set (what NHF used to supply), the tree walks onto the
    # lake and the guard must reject the whole tree rather than emit NaN.
    trees = build_gage_trees_from_mappings(rconn, {"G": 10}, area, waterbody_segs=frozenset())
    assert "G" not in trees

    # With the lake correctly in the stop set, the tree is kept and simply stops there.
    trees = build_gage_trees_from_mappings(rconn, {"G": 10}, area, waterbody_segs=frozenset({20}))
    assert trees["G"].seg_order.tolist() == [10]
