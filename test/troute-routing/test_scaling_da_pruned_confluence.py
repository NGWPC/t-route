"""Edge Case 1 and Edge Case 2 interacting at a partially pruned confluence.

At a confluence where one branch carries its own gage, Edge Case 1 removes that branch from
this gage's tree. Edge Case 2 still splits the parent increment across ALL branches in
proportion to their modeled flow, so the surviving branch is owed ``Q_b / sum(Q)`` and the
stopped branch's share must go UNALLOCATED -- the gage that owns it applies its own
correction there, and reassigning that share double-counts against it.

The defect this pins: ``branch_sum`` was summed over surviving branches only, and the
denominator is ``max(Q_parent, branch_sum)``. The routed parent flow is the only remaining
trace of the stopped branch, and on a rising limb it lags BELOW the surviving branch alone.
The clamp then selected the surviving branch's own flow as the denominator and handed it a
factor of 1.0 -- the entire parent increment, stopped branch's share included.

Steady state hid it: there ``Q_parent >= sum(Q_branch)``, so the denominator was the parent
flow and the split was already correct. Only the lagged regime exposes it, which is the same
regime the non-amplification clamp exists to handle (measured at 38% of active confluence
timesteps on the spun-up continental flood).

Both the NumPy reference and the compiled kernel are covered, since the two must agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.routing.fast_reach.scaling_da import (
    _pruned_branch_flow,
    _tree_dq_nodes,
    apply_scaling_da,
)
from troute.scaling_da import build_gage_trees_from_mappings

THETA = 0.77
Q_SURVIVING, Q_STOPPED = 12.0, 8.0
EXPECTED_SHARE = Q_SURVIVING / (Q_SURVIVING + Q_STOPPED)   # Edge Case 2: 0.6


def _trees():
    """Confluence 100 fed by 101 (survives) and 102 (has gage H, so it is pruned)."""
    rconn = {100: [101, 102], 101: [], 102: []}
    area = {100: 1000.0, 101: 600.0, 102: 400.0}
    built = build_gage_trees_from_mappings(rconn, {"G": 100, "H": 102}, area,
                                           theta_default=THETA)
    return {"G": built["G"]}


def test_the_stopped_branch_is_recorded_against_its_confluence():
    """The tree must remember what the stop rule removed, or the split cannot be right."""
    t = _trees()["G"]
    assert list(np.asarray(t.seg_order).ravel()) == [100, 101]     # 102 pruned
    assert list(np.asarray(t.pruned_segs).ravel()) == [102]
    assert list(np.asarray(t.pruned_ptr).ravel()) == [0, 1, 1]     # attached to node 0


def test_positions_resolve_for_the_pruned_branch():
    t = _trees()["G"].with_positions({100: 0, 101: 1, 102: 2})
    assert list(np.asarray(t.pruned_positions).ravel()) == [2]
    assert list(np.asarray(t.pruned_pos_ptr).ravel()) == [0, 1, 1]


@pytest.mark.parametrize(
    ("q_parent", "label"),
    [(20.0, "steady: parent flow equals the branch sum"),
     (4.0, "rising limb: parent flow lags below the surviving branch alone")],
)
def test_reference_split_is_proportional_in_both_regimes(q_parent, label):
    t = _trees()["G"].with_positions({100: 0, 101: 1, 102: 2})
    q_full = np.array([[q_parent, Q_SURVIVING, Q_STOPPED]], dtype=float)
    node = _tree_dq_nodes(
        t, q_full[:, np.asarray(t.seg_positions)], np.array([10.0]),
        1e-6, "flow_ratio", "G",
        pruned_q=_pruned_branch_flow(t, q_full),
    )
    assert node[0, 1] / node[0, 0] == pytest.approx(EXPECTED_SHARE), label


@pytest.mark.parametrize("q_parent", [20.0, 4.0])
def test_compiled_kernel_agrees_end_to_end(q_parent):
    """The compiled path must not keep the defect after the reference is fixed."""
    idx = pd.date_range("2020-01-01", periods=2, freq="1h")
    q_model = pd.DataFrame([[q_parent, Q_SURVIVING, Q_STOPPED]] * 2,
                           index=idx, columns=[100, 101, 102], dtype=float)
    q_obs = pd.DataFrame([[q_parent + 10.0]] * 2, index=idx, columns=["G"], dtype=float)
    _, dq = apply_scaling_da(q_model, q_obs, {"G": 100}, _trees(),
                                    method="flow_ratio", min_flow_cms=1e-6)
    assert dq.iloc[0][101] / dq.iloc[0][100] == pytest.approx(EXPECTED_SHARE)


def test_the_stopped_branch_share_is_not_reassigned():
    """The surviving branch must never receive the whole parent increment.

    This is the failure directly: before the fix the lagged case returned a share of 1.0.
    """
    t = _trees()["G"].with_positions({100: 0, 101: 1, 102: 2})
    q_full = np.array([[4.0, Q_SURVIVING, Q_STOPPED]], dtype=float)
    node = _tree_dq_nodes(
        t, q_full[:, np.asarray(t.seg_positions)], np.array([10.0]),
        1e-6, "flow_ratio", "G",
        pruned_q=_pruned_branch_flow(t, q_full),
    )
    share = node[0, 1] / node[0, 0]
    assert share < 1.0
    assert share == pytest.approx(EXPECTED_SHARE)


def test_an_unpruned_confluence_is_unchanged():
    """No pruning means no pruned flow, so the arithmetic must be bit-identical."""
    rconn = {100: [101, 102], 101: [], 102: []}
    area = {100: 1000.0, 101: 600.0, 102: 400.0}
    t = build_gage_trees_from_mappings(rconn, {"G": 100}, area, theta_default=THETA)["G"]
    t = t.with_positions({100: 0, 101: 1, 102: 2})
    assert np.asarray(t.pruned_positions).size == 0
    q_full = np.array([[20.0, Q_SURVIVING, Q_STOPPED]], dtype=float)
    assert _pruned_branch_flow(t, q_full) is None
    node = _tree_dq_nodes(t, q_full[:, np.asarray(t.seg_positions)], np.array([10.0]),
                          1e-6, "flow_ratio", "G", pruned_q=None)
    # Both branches present: shares are proportional and sum to the parent.
    assert node[0, 1] + node[0, 2] == pytest.approx(node[0, 0])


# --------------------------------------------------- travel-time lag contracts


def test_lag_edge_falls_back_to_the_latest_increment():
    """The LAST timestep must keep a correction on every lagged segment.

    That timestep is what seeds q0 for the forecast. An earlier revision applied
    nothing past the window end, which handed off a state whose upstream spread was
    zero everywhere except the gage segments -- silently nulling the whole point of
    the prognostic hand-off.
    """
    from troute.routing.fast_reach.scaling_da import _lagged_dq

    nt = 8
    dq = np.full(nt, 5.0)
    for shift in (0, 1, 3):
        out = _lagged_dq(dq, np.ones((nt, 1)), None, np.array([shift], dtype=np.int64))
        assert out[-1, 0] == pytest.approx(5.0), f"shift {shift} lost the seeded step"


def test_lag_is_not_sliceable_across_chunks():
    """Pins WHY chunking must be disabled under the lag rather than trusted.

    A chunked call slices dq_o but keeps whole-window shifts, so t+shift crossing a
    boundary reads past its slice. The spread claims bit-identity with the unchunked
    call; with a lag that claim is false, so the driver turns chunking off.
    """
    from troute.routing.fast_reach.scaling_da import _lagged_dq

    dq = np.arange(8, dtype=float)
    shift = np.array([2], dtype=np.int64)
    full = _lagged_dq(dq, np.ones((8, 1)), None, shift).ravel()
    chunked = np.concatenate(
        [_lagged_dq(dq[c:c + 4], np.ones((4, 1)), None, shift).ravel() for c in (0, 4)]
    )
    assert not np.array_equal(full, chunked)


def test_tau_accumulates_the_parent_reach_and_syncs_junction_siblings():
    """tau[j] must sum the PARENT chain, not the segment's own reach.

    Two consequences, both load-bearing. (1) The correction sits on segment j's
    OUTPUT, so the water it describes still has to traverse the parent to reach the
    gage; charging j's own length is off by one reach. (2) Because every child of a
    junction is charged the same parent reach, siblings share tau exactly -- so the
    flow-ratio split at that junction is evaluated at the same instant as both
    branch corrections. Charging each child its own length desynchronises them.

    tau is returned in TIMESTEPS.
    """
    import pandas as pd
    from nwm_routing.scaling_da_apply import ScalingDA

    from troute.scaling_da import build_gage_trees_from_mappings

    # gage 100; junction at 101 with two children 102, 103 of DIFFERENT lengths.
    rconn = {100: [101], 101: [102, 103], 102: [], 103: []}
    area = {100: 40.0, 101: 30.0, 102: 10.0, 103: 20.0}
    trees = build_gage_trees_from_mappings(rconn, {"G": 100}, area, theta_default=0.77)
    da = ScalingDA.__new__(ScalingDA)
    da.celerity_mps = 1.0
    da._dx = pd.Series({100: 3600.0, 101: 7200.0, 102: 1800.0, 103: 900.0}, dtype=float)
    tau, _ = da._tree_tau(trees["G"], dt=1.0)  # dt=1 s -> steps == seconds

    order = list(map(int, trees["G"].seg_order))
    tau_of = dict(zip(order, tau))
    assert tau_of[100] == pytest.approx(0.0)          # the gage itself
    assert tau_of[101] == pytest.approx(3600.0)       # traverses 100, not 101
    # Both children traverse 101 (7200 s) on top of 101's own tau -> identical.
    assert tau_of[102] == pytest.approx(10800.0)
    assert tau_of[103] == pytest.approx(10800.0)


def test_missing_reach_length_fails_closed_past_the_horizon():
    """A reach absent from a network that HAS lengths must not get zero transit.

    Zero transit put the reach at the gage's own instant and kept its whole
    subtree inside the propagation horizon, applying corrections at
    demonstrably wrong timesteps. Past the horizon withholds them instead. A
    network with NO lengths at all is a different case (an inert lag) and is
    covered by the partial-instance tests elsewhere.
    """
    import numpy as np
    import pandas as pd
    from nwm_routing.scaling_da_apply import ScalingDA

    from troute.scaling_da import build_gage_trees_from_mappings

    trees = build_gage_trees_from_mappings(
        {100: [101], 101: []}, {"G": 100}, {100: 30.0, 101: 20.0}, theta_default=0.77
    )
    da = ScalingDA.__new__(ScalingDA)
    da.celerity_mps = 1.0
    da.max_travel_time_h = 1.0
    da._dx = pd.Series({101: 3600.0}, dtype=float)  # 100 (the crossed reach) missing
    dt = 3600.0
    tau, fb = da._tree_tau(trees["G"], dt=dt)
    assert fb == 1
    horizon_steps = da.max_travel_time_h * 3600.0 / dt
    order = list(map(int, trees["G"].seg_order))
    tau_of = dict(zip(order, tau))
    assert float(tau_of[101]) > horizon_steps  # withheld, not applied at t


def test_edge_persistence_decays():
    """Past the last observation the increment persists, but DECAYS.

    Everything else in this DA decays a stale observation (da_decay_coefficient);
    holding the edge increment constant for up to max_travel_time_h would make the
    lag's closure the one place a stale innovation never ages.
    """
    from troute.routing.fast_reach.scaling_da import _lagged_dq

    nt = 6
    dq = np.full(nt, 10.0)
    shift = np.array([3], dtype=np.int64)
    out = _lagged_dq(dq, np.ones((nt, 1)), None, shift, 0.5).ravel()
    # steps 0..2 read within the window; 3..5 are 1,2,3 steps past its end.
    assert out[:3] == pytest.approx([10.0, 10.0, 10.0])
    assert out[3:] == pytest.approx([5.0, 2.5, 1.25])


def test_chunked_spread_matches_unchunked_under_a_lag():
    """Chunks overlap by max(shift), so chunking stays bit-identical WITH a lag.

    Slicing dq_o to the chunk alone made t+shift cross the boundary and read the
    wrong end of the series, while the docstring promised bit-identity. Ohio never
    auto-chunks; CONUS does, so this path was unexercised where it mattered.
    """
    from troute.routing.fast_reach.scaling_da import _lagged_dq

    nt, chunk, sh = 8, 4, 2
    dq = np.arange(nt, dtype=float)
    shift = np.array([sh], dtype=np.int64)
    full = _lagged_dq(dq, np.ones((nt, 1)), None, shift).ravel()
    parts = []
    for c0 in range(0, nt, chunk):
        c1 = min(c0 + chunk, nt)
        # exactly what the driver now passes: the chunk plus the shift's reach
        parts.append(_lagged_dq(dq[c0:min(c1 + sh, nt)], np.ones((c1 - c0, 1)),
                                None, shift).ravel())
    assert np.array_equal(full, np.concatenate(parts))
