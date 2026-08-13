"""Unit tests for the Stage A in-kernel downstream DA path.

Covers the two pieces that differ from the output-only path:
  * ``apply_scaling_da(dq_o_by_site=...)`` -- drive the upstream spread
    from a supplied per-gage delta (the kernel-recorded ``nudge``) instead of
    ``obs - Q_model(gage)``; must equal the obs-driven result when the delta
    matches.
  * ``ScalingDA.apply_in_kernel`` -- map the kernel ``nudge`` (``r[9]``)
    to gage segments, reconstruct the gage background (``Q_analyzed - nudge``),
    spread upstream, and scatter back leaving the gage segment a no-op (it was
    already overridden in the MC kernel).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from nwm_routing.scaling_da_apply import ScalingDA

from troute.routing.fast_reach.scaling_da import apply_scaling_da
from troute.scaling_da import build_gage_trees_from_mappings


def _bare(**attrs) -> ScalingDA:
    o = ScalingDA.__new__(ScalingDA)
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


def _linear_tree():
    """Gage at seg 100, linear upstream chain 100<-101<-102 (areas 30/20/10)."""
    rconn = {100: [101], 101: [102], 102: []}
    area = {100: 30.0, 101: 20.0, 102: 10.0}
    return build_gage_trees_from_mappings(rconn, {"G": 100}, area, theta_default=0.77)


def test_kernel_dq_o_override_matches_obs_path():
    """dq_o_by_site reproduces the obs-driven spread when delta == obs - Q_gage."""
    trees = _linear_tree()
    idx = pd.date_range("2000-01-01", periods=2, freq="h")
    q_model = pd.DataFrame({100: [10.0, 10.0], 101: [6.0, 6.0], 102: [4.0, 4.0]}, index=idx)
    gmap = {"G": 100}

    # obs-driven: obs=12 -> dq_o = 12-10 = 2 at the gage.
    q_obs = pd.DataFrame({"G": [12.0, 12.0]}, index=idx)
    q_a, _ = apply_scaling_da(q_model, q_obs, gmap, trees, method="flow_ratio")

    # override-driven: supply dq_o = 2 directly, with background at the gage.
    q_bg = q_model.copy()
    q_b, _ = apply_scaling_da(
        q_bg,
        None,
        gmap,
        trees,
        dq_o_by_site={"G": np.array([2.0, 2.0])},
    )
    pd.testing.assert_frame_equal(q_a, q_b)
    # sanity: gage recovered the obs; upstream got an area-scaled FRACTION of dq_o
    # (101 = 6 + 2*(20/30)^0.77), i.e. above background but below background+dq_o.
    assert np.allclose(q_b[100].to_numpy(), 12.0)
    assert (q_b[101].to_numpy() > 6.0).all()
    assert (q_b[101].to_numpy() < 8.0).all()


def _run_results(nudge_vals):
    """One subnetwork: segs [100,101,102], gage 100 overridden to 12, nts=len(vals).

    r[1] q-columns are the ANALYZED (post-override) flow; r[3][0] the gage seg;
    r[9] the nudge (gages, nts+1) with col 0 = IC (0) and cols 1: the applied delta.
    """
    nudge_vals = np.asarray(nudge_vals, dtype=float)
    nts = nudge_vals.size
    ids = np.array([100, 101, 102])
    q = np.array([12.0, 6.0, 4.0])  # analyzed: gage already = obs
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = np.tile(q[:, None], (1, nts))  # q at every timestep
    nudge = np.zeros((1, nts + 1), dtype=np.float32)
    nudge[0, 1:] = nudge_vals
    r = [
        ids,
        arr,
        0,
        (np.array([100]), np.zeros(1), np.zeros(1)),
        0,
        0,
        0,
        0,
        np.zeros((3, nts), dtype=np.float32),
        nudge,
    ]
    return [r]


def test_apply_in_kernel_spreads_upstream_gage_is_noop():
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
    )
    rr = _run_results([2.0, 2.0])
    before = rr[0][1][:, 0::4].copy()
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")
    after = rr[0][1][:, 0::4]
    # gage seg 100 (row 0) unchanged -- already corrected in-kernel.
    np.testing.assert_allclose(after[0], before[0])
    # upstream 101/102 (rows 1,2) got the area-scaled spread (increased).
    assert (after[1] > before[1]).all()
    assert (after[2] > before[2]).all()
    # background reconstruction: 101 delta == 2 * (20/30)^0.77.
    expected_101 = 6.0 + 2.0 * (20.0 / 30.0) ** 0.77
    np.testing.assert_allclose(after[1], expected_101, rtol=1e-5)


def test_apply_in_kernel_zero_nudge_is_noop():
    """obs == model -> nudge 0 -> strict no-op (idempotency)."""
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
    )
    rr = _run_results([0.0, 0.0])
    before = rr[0][1][:, 0::4].copy()
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")
    np.testing.assert_array_equal(rr[0][1][:, 0::4], before)


def _two_tree():
    """Two disjoint gages: G1 seg 100<-101, G2 seg 200<-201."""
    rconn = {100: [101], 101: [], 200: [201], 201: []}
    area = {100: 30.0, 101: 20.0, 200: 30.0, 201: 20.0}
    return build_gage_trees_from_mappings(rconn, {"G1": 100, "G2": 200}, area, theta_default=0.77)


def test_zero_own_innovation_still_spreads_the_halo():
    """A site with no innovation in its OWN window must still spread a nonzero halo.

    The lag reads the halo at timesteps inside this window, so inclusion has to be
    decided from the concatenated spread array. Gating on the window's own values
    (the np.any(nud) candidate gate, or the "trusted somewhere in this window"
    survivor set) dropped such sites, and what a backward shift applied then
    depended on max_loop_size: the 544-segment, 38.6 cms residual left after the
    static-celerity fix on the 48 h vs 96 h comparison.
    """
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        max_travel_time_h=48.0,
        celerity_mps=1.0,  # dx=3600 m at dt=3600 s -> tau(101) = exactly 1 step
        _loop_obs=None,
        _dx=pd.Series({100: 3600.0, 101: 3600.0, 102: 3600.0}, dtype=float),
    )
    rr = _run_results([0.0, 0.0])  # zero own innovation
    before = rr[0][1][:, 0::4].copy()
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01",
                      halo={100: np.array([5.0, 5.0])})
    after = rr[0][1][:, 0::4]
    # Gage itself: shift 0 reads only the (zero) own innovation -- unchanged.
    np.testing.assert_allclose(after[0], before[0])
    # 101 at t=0 reads dq[0+1] = own zero; at t=1 reads dq[2] = the halo.
    np.testing.assert_allclose(after[1][0], before[1][0])
    expected_101_t1 = 6.0 + 5.0 * (20.0 / 30.0) ** 0.77
    np.testing.assert_allclose(after[1][1], expected_101_t1, rtol=1e-5)


def test_spread_reaches_the_warmstate_columns():
    """State seeding depends on the corrected discharge landing in the exact
    columns ``new_q0`` reads: ``r[1][:, [-4, -4, -2, -1]]`` -> qu0/qd0/h0/ql0.

    Column ``-4`` of an ``(n_seg, 4*nts)`` array is ``4*(nts-1)``, which is in the
    ``0::4`` discharge stride, so the last timestep's corrected q is what carries
    over. This asserts that indexing claim rather than trusting it; depth (``-2``)
    is deliberately NOT corrected, which is the accepted transient.
    """
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
    )
    rr = _run_results([2.0, 2.0])
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")

    arr = rr[0][1]
    warmstate = arr[:, [-4, -4, -2, -1]]  # exactly what AbstractNetwork.new_q0 slices
    expected_101 = 6.0 + 2.0 * (20.0 / 30.0) ** 0.77
    # upstream reach 101 carries its CORRECTED discharge into qu0 and qd0. This is the
    # load-bearing assertion: it fails if -4 ever stops landing in the 0::4 stride.
    np.testing.assert_allclose(warmstate[1, 0], expected_101, rtol=1e-5)
    np.testing.assert_allclose(warmstate[1, 1], expected_101, rtol=1e-5)
    # depth (-2) is deliberately left uncorrected -- pin it so the accepted Q/h
    # inconsistency stays a known, deliberate gap rather than drifting silently.
    np.testing.assert_array_equal(warmstate[:, 2], 0.0)


def test_seed_state_only_on_the_final_window():
    """The warmstate is seeded once per RUN, not once per window.

    Muskingum-Cunge holds no storage state, so a dQ written into q0 is a one-shot mass
    pulse rather than a persistent analysis increment: it transits the source gage and
    inflates its background, which debits the NEXT window's innovation (obs - background)
    by the amount we injected. Re-seeding every window therefore starves every tree of
    correction, including the sibling branches held-out gages sit on. Measured on the
    Ohio LOO fold, 24 h windows: held-out bias -23.8% with one window boundary vs -29.1%
    with eleven.

    This survives the removal of the arm switch. "Always prognostic" means the spread is
    always the state-seeding kind, NOT that it seeds on every window -- collapsing the
    two would reintroduce exactly the debit above.
    """
    from nwm_routing.scaling_da_apply import should_seed_state

    n = 4
    da = object()  # should_seed_state only needs "not None" now
    assert [should_seed_state(da, i, n) for i in range(n)] == [False, False, False, True]

    # No DA configured at all.
    assert not any(should_seed_state(None, i, n) for i in range(n))

    # Single-window run: that window IS the hand-off, so it seeds.
    assert should_seed_state(da, 0, 1)


def test_spread_runs_exactly_once_per_window():
    """The spread is applied once per window -- never twice, never zero.

    The two drivers pick the position with `seed_state` and run the output pass under
    `not seed_state`, so the two branches partition every window. A second call in the
    same window would spread on top of already-corrected interior flow, because
    apply_in_kernel reconstructs the background only at the tree root.
    """
    from nwm_routing.scaling_da_apply import should_seed_state

    n = 5
    da = object()
    for i in range(n):
        before = should_seed_state(da, i, n)
        after = not before
        assert before + after == 1, f"window {i}: applied {before + after} times"


def test_chunked_spread_matches_unchunked_through_apply_in_kernel():
    """Chunking the spread must not change the result, WITH a lag in play.

    The equivalent check on the spread helper alone did not catch a driver-level
    defect: the chunk loop sliced the lag arrays by time, which is meaningless
    for a per-segment lag and raised only once a window was long enough to
    chunk. Ohio at 48 h never chunks, so this has to be forced.
    """
    o_kwargs = dict(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_travel_time_h=48.0,
        celerity_mps=1.0,
        _dx=pd.Series({100: 3600.0, 101: 3600.0, 102: 3600.0}, dtype=float),
    )
    n = 8
    nudge = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]

    rr_full = _run_results(nudge)
    _bare(spread_chunk_timesteps=0, **o_kwargs).apply_in_kernel(
        rr_full, nts=n, dt=3600, t0="2000-01-01"
    )
    rr_chunked = _run_results(nudge)
    _bare(spread_chunk_timesteps=3, **o_kwargs).apply_in_kernel(
        rr_chunked, nts=n, dt=3600, t0="2000-01-01"
    )
    np.testing.assert_array_equal(
        rr_chunked[0][1][:, 0::4], rr_full[0][1][:, 0::4]
    )
    # and the spread actually did something, or the comparison is vacuous
    assert (rr_full[0][1][1:, 0::4] != _run_results(nudge)[0][1][1:, 0::4]).any()


class TestResolveSpreadChunk:
    def _o(self, **attrs):
        return _bare(min_flow=1e-6, **attrs)

    def test_explicit_config_wins_including_zero(self, monkeypatch):
        monkeypatch.setenv("SCALING_DA_CHUNK", "7")
        assert self._o(spread_chunk_timesteps=12)._resolve_spread_chunk(288, 10**6) == 12
        assert self._o(spread_chunk_timesteps=0)._resolve_spread_chunk(288, 10**6) == 0

    def test_env_var_is_honored_when_config_unset(self, monkeypatch):
        monkeypatch.setenv("SCALING_DA_CHUNK", "7")
        assert self._o(spread_chunk_timesteps=None)._resolve_spread_chunk(288, 100) == 7

    def test_auto_chunks_only_a_large_window(self, monkeypatch):
        monkeypatch.delenv("SCALING_DA_CHUNK", raising=False)
        o = self._o(spread_chunk_timesteps=None)
        # Ohio scale: no chunking.
        assert o._resolve_spread_chunk(288, 11_327) == 0
        # CONUS scale: chunk to keep one transient under the element budget.
        chunk = o._resolve_spread_chunk(288, 1_100_000)
        assert 0 < chunk < 288
        assert chunk * 1_100_000 <= o._SPREAD_CHUNK_BUDGET_ELEMS

