"""Compiled-kernel vs NumPy-reference equivalence for the simple-scaling DA spread.

Drives the compiled :func:`troute.routing.fast_reach.scaling_da_kernel.spread_trees`
against the NumPy reference :func:`_tree_dq_nodes` (the in-file oracle) on random
gage trees -- chains, confluences, deep telescoping, dry junctions, and gaps -- and
asserts bit-level agreement (``allclose(rtol=0, atol=1e-9)``). This is the swap-under-
a-delivered-result gate from ``t-route_dev: scaling_da/next/cython-kernel.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.routing.fast_reach.scaling_da import (
    _tree_dq_nodes,
    apply_scaling_da,
)
from troute.routing.fast_reach.scaling_da_kernel import spread_trees
from troute.scaling_da.gage_tree import GageTree

MIN_FLOW = 1e-6
ATOL = 1e-9


def _rand_tree(rng, n, nt, kind):
    """A random BFS tree (parent[j] < j) plus its per-timestep gage correction."""
    parent = np.empty(n, dtype=np.int64)
    parent[0] = -1
    is_junc = np.zeros(n, dtype=np.uint8)
    for j in range(1, n):
        if kind == "chain":
            parent[j] = j - 1  # deep telescoping
        elif kind == "confluence":
            parent[j] = rng.integers(0, j)  # many children per parent
        else:  # mixed
            parent[j] = rng.integers(max(0, j - 3), j)
    for j in range(1, n):
        is_junc[j] = 1 if (kind != "chain" and rng.random() < 0.5) else 0
    areas = rng.uniform(1.0, 5000.0, size=n).astype(np.float64)
    theta = np.full(n, float(rng.uniform(0.4, 1.3)), dtype=np.float64)  # uniform per tree
    dq_o = rng.uniform(-50.0, 50.0, size=nt).astype(np.float64)
    if kind == "mixed" and rng.random() < 0.4:
        dq_o[rng.integers(0, nt)] = 0.0  # a gap (no correction at that step)
    # Travel-time lag, one value per segment. Half the trees stay un-lagged
    # (in-window 1, shift 0) so that path is fuzzed too; the rest get a random
    # horizon mask and a random whole-timestep shift, including shifts past the
    # window end so the kernel's clamp-to-last-column and its edge decay are
    # exercised against the reference.
    if rng.random() < 0.5:
        in_window = np.ones(n, dtype=np.float64)
        tshift = np.zeros(n, dtype=np.int64)
    else:
        in_window = (rng.random(n) >= 0.2).astype(np.float64)
        tshift = rng.integers(0, nt + 2, size=n).astype(np.int64)
    return parent, is_junc, areas, theta, dq_o, in_window, tshift


def _gagetree(cols, areas, theta, parent, is_junc):
    return GageTree(
        gage_fp=int(cols[0]),
        gage_area_sqkm=float(areas[0]),
        seg_order=cols.copy(),
        seg_areas_sqkm=areas.copy(),
        theta=float(theta[0]),
        seg_positions=cols.copy(),
        seg_parent_idx=parent.copy(),
        step_is_junction=is_junc.astype(bool),
    )


@pytest.mark.parametrize("seed", range(40))
def test_spread_trees_matches_numpy_reference(seed):  # noqa: PLR0915
    """Batched compiled kernel == per-tree NumPy reference, bit-for-bit.

    One long body on purpose: the fuzzed network, both implementations, and the
    comparison have to read in order for the equivalence claim to be checkable.
    """
    rng = np.random.default_rng(seed)
    ncol = 200
    # Half the seeds decay the persisted edge increment, half do not.
    edge_decay = 1.0 if seed % 2 == 0 else float(rng.uniform(0.5, 0.99))
    kinds = ("chain", "confluence", "mixed")
    max_err_q = 0.0
    max_err_dq = 0.0

    for _ in range(50):  # 40 seeds x 50 = 2000 batched cases
        ntrees = int(rng.integers(1, 5))
        nt = int(rng.integers(1, 6))
        q_model = rng.uniform(0.0, 300.0, size=(nt, ncol)).astype(np.float64)
        # Inject near-dry columns to exercise the min_flow guard and the clamp.
        if rng.random() < 0.5:
            dry = rng.choice(ncol, size=int(rng.integers(1, 20)), replace=False)
            q_model[:, dry] = rng.uniform(0.0, 5e-7, size=(nt, dry.size))

        used: set[int] = set()
        trees = []
        for _t in range(ntrees):
            n = int(rng.integers(1, 46))
            avail = [c for c in range(ncol) if c not in used]
            if len(avail) < n:
                continue
            cols = np.asarray(rng.choice(avail, size=n, replace=False), dtype=np.int64)
            used.update(int(c) for c in cols)
            parent, is_junc, areas, theta, dq_o, in_window, tshift = _rand_tree(rng, n, nt, kinds[rng.integers(0, 3)])
            trees.append((cols, parent, is_junc, areas, theta, dq_o, in_window, tshift))
        if not trees:
            continue

        # NumPy reference: the original per-tree accumulation.
        q_ref = q_model.copy()
        dq_ref = np.zeros_like(q_model)
        for cols, parent, is_junc, areas, theta, dq_o, in_window, tshift in trees:
            gt = _gagetree(cols, areas, theta, parent, is_junc)
            dq_nodes = _tree_dq_nodes(gt, q_model[:, cols], dq_o, MIN_FLOW, "flow_ratio", "s",
                                      in_window=in_window, tshift=tshift,
                                      edge_decay=edge_decay)
            q_ref[:, cols] += dq_nodes
            dq_ref[:, cols] += dq_nodes

        # Compiled kernel: flatten into CSR buffers exactly as the adapter does.
        offs = [0]
        pf, jf, af, tf, cf, df, wf, sf = [], [], [], [], [], [], [], []
        for cols, parent, is_junc, areas, theta, dq_o, in_window, tshift in trees:
            cf.append(cols)
            pf.append(parent)
            jf.append(is_junc)
            af.append(areas)
            tf.append(theta)
            df.append(dq_o)
            wf.append(in_window)
            sf.append(tshift)
            offs.append(offs[-1] + cols.size)
        tree_off = np.asarray(offs, dtype=np.int64)
        max_seg = int(np.max(np.diff(tree_off)))
        qbuf = np.empty((max_seg, nt))
        mult = np.empty((max_seg, nt))
        bsum = np.empty((max_seg, nt))
        q_ker = q_model.copy()
        dq_ker = np.zeros_like(q_model)
        spread_trees(
            q_model,
            tree_off,
            np.concatenate(pf),
            np.concatenate(jf),
            np.concatenate(af),
            np.concatenate(tf),
            np.concatenate(cf),
            np.ascontiguousarray(np.stack(df)),
            float(MIN_FLOW),
            edge_decay,
            np.concatenate(wf),
            np.concatenate(sf),
            np.zeros(int(tree_off[-1]) + 1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            q_ker,
            dq_ker,
            qbuf,
            mult,
            bsum,
        )

        max_err_q = max(max_err_q, float(np.max(np.abs(q_ker - q_ref))))
        max_err_dq = max(max_err_dq, float(np.max(np.abs(dq_ker - dq_ref))))
        np.testing.assert_allclose(q_ker, q_ref, rtol=0, atol=ATOL)
        np.testing.assert_allclose(dq_ker, dq_ref, rtol=0, atol=ATOL)

    assert max_err_q <= ATOL
    assert max_err_dq <= ATOL


def test_nan_junction_parent_matches_reference():
    """A NaN at a junction-parent column must zero the split, like np.maximum.

    Regression for the clamp's NaN handling: ``denom = qp if qp > bs else bs`` is
    np.fmax (NaN-suppressing) and would fall back to the finite branch sum,
    spuriously correcting children the reference never touches. The reference uses
    np.maximum (NaN-propagating) -> denom=NaN -> ``denom > min_flow`` False ->
    factor 0. The fuzz above only injects near-dry *finite* columns, so this pins
    the one case that diverges (q_parent=NaN, finite branch flows).
    """
    # cols: [gage, junction-parent, branch1, branch2]; parent col carries NaN.
    cols = np.array([10, 11, 12, 13], dtype=np.int64)
    parent = np.array([-1, 0, 1, 1], dtype=np.int64)
    is_junc = np.array([0, 0, 1, 1], dtype=np.uint8)
    areas = np.array([100.0, 80.0, 30.0, 50.0], dtype=np.float64)
    theta = np.full(4, 0.5, dtype=np.float64)
    dq_o = np.array([10.0], dtype=np.float64)
    ncol = 14
    q_model = np.zeros((1, ncol), dtype=np.float64)
    q_model[0, 10] = 20.0
    q_model[0, 11] = np.nan  # junction parent has NaN modeled flow
    q_model[0, 12] = 6.0
    q_model[0, 13] = 10.0

    gt = _gagetree(cols, areas, theta, parent, is_junc)
    dq_ref = _tree_dq_nodes(gt, q_model[:, cols], dq_o, MIN_FLOW, "flow_ratio", "s")

    tree_off = np.array([0, 4], dtype=np.int64)
    qbuf = np.empty((4, 1))
    mult = np.empty((4, 1))
    bsum = np.empty((4, 1))
    q_ker = q_model.copy()
    dq_ker = np.zeros_like(q_model)
    spread_trees(
        q_model,
        tree_off,
        parent,
        is_junc,
        areas,
        theta,
        cols,
        np.ascontiguousarray(dq_o[None, :]),
        float(MIN_FLOW),
        1.0,                            # edge_decay: none
        np.ones(4, dtype=np.float64),   # lag off: every segment in window
        np.zeros(4, dtype=np.int64),    # lag off: no shift
        np.zeros(5, dtype=np.int64),    # no pruned branches in this fixture
        np.empty(0, dtype=np.int64),
        q_ker,
        dq_ker,
        qbuf,
        mult,
        bsum,
    )
    # Children (cols 12, 13) get zero correction in both; NaN parent stops the split.
    np.testing.assert_allclose(dq_ker[:, cols], dq_ref, rtol=0, atol=ATOL, equal_nan=True)
    assert dq_ker[0, 12] == 0.0
    assert dq_ker[0, 13] == 0.0


def test_apply_end_to_end_multi_timestep():
    """apply_scaling_da drives the compiled path on a multi-timestep frame.

    Guards the F-contiguity trap: to_numpy() on a homogeneous-float frame yields an
    F-ordered block for >1 timestep, which the kernel's ``double[:, ::1]`` binding
    rejects unless the adapter forces C-contiguity.
    """

    from troute.routing.fast_reach.scaling_da import apply_scaling_da
    from troute.scaling_da.gage_tree import build_one_gage_tree

    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    q_model = pd.DataFrame(
        {1: [20.0, 22.0, 19.0], 2: [16.0, 17.0, 15.0], 3: [6.0, 7.0, 5.0], 4: [10.0, 10.0, 10.0]},
        index=idx,
    )
    q_obs = pd.DataFrame({"A": [30.0, 33.0, 28.0]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3, 4]},
        area_sqkm={1: 100.0, 2: 80.0, 3: 30.0, 4: 50.0},
        stop_segs=frozenset(),
        theta=0.5,
    )
    _q_corr, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    dq_o = np.array([10.0, 11.0, 9.0])
    dq2 = dq_o * (80.0 / 100.0) ** 0.5  # linear step 1->2
    q2 = np.array([16.0, 17.0, 15.0])  # per-timestep junction split uses per-t flow
    q3 = np.array([6.0, 7.0, 5.0])
    q4 = np.array([10.0, 10.0, 10.0])
    np.testing.assert_allclose(dq[2].to_numpy(), dq2, rtol=1e-9)
    np.testing.assert_allclose(dq[3].to_numpy(), dq2 * (q3 / q2), rtol=1e-9)
    np.testing.assert_allclose(dq[4].to_numpy(), dq2 * (q4 / q2), rtol=1e-9)


def _simple_case():
    """A 3-timestep single-tree case reused by the dq_o shape-contract tests."""

    from troute.scaling_da.gage_tree import build_one_gage_tree

    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    q_model = pd.DataFrame(
        {1: [20.0, 22.0, 19.0], 2: [16.0, 17.0, 15.0], 3: [6.0, 7.0, 5.0], 4: [10.0, 10.0, 10.0]},
        index=idx,
    )
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3, 4]},
        area_sqkm={1: 100.0, 2: 80.0, 3: 30.0, 4: 50.0},
        stop_segs=frozenset(),
        theta=0.5,
    )
    return q_model, {"A": tree}, {"A": 1}


def test_dq_o_too_short_raises_but_overlap_is_allowed():
    """Too SHORT must fail loudly; longer is the chunked call's forward overlap.

    The kernel reads dq_o with bounds checking disabled, so a short row reads out of
    bounds and silently corrupts the correction. A row LONGER than nts is legitimate:
    a chunked spread hands each tree the overlap its shifts read into, and the kernel
    takes the innovation length from the array rather than from the output window.
    """

    q_model, trees, gage_to_fp = _simple_case()  # nts == 3
    with pytest.raises(ValueError, match="at least nts"):
        apply_scaling_da(
            q_model, None, gage_to_fp, trees,
            dq_o_by_site={"A": np.array([5.0, 5.0])},  # 2 < 3
        )
    # 5 > 3: accepted, and the extra values are only reachable through a shift.
    out, _ = apply_scaling_da(
        q_model, None, gage_to_fp, trees,
        dq_o_by_site={"A": np.array([5.0, 5.0, 5.0, 9.0, 9.0])},
    )
    assert out.shape[0] == 3


def test_mixed_length_dq_rows_match_single_site_results():
    """Batching sites with different innovation lengths must not change any site.

    _stack_dq pads shorter rows to the batch width, and the compiled kernel takes
    ONE innovation length for the whole batch -- so a padded read lands inside
    the array and the kernel's own edge decay never fires for it. Padding with
    the RAW last value therefore gave a no-halo site an undecayed constant tail
    whenever any other site carried a halo, while the same site alone decayed
    from its true end. Padding now pre-applies the decay, so batch composition
    is irrelevant.
    """
    from troute.scaling_da.gage_tree import build_one_gage_tree

    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    q_model = pd.DataFrame(
        {1: [20.0, 20.0, 20.0], 2: [8.0, 8.0, 8.0],
         10: [15.0, 15.0, 15.0], 11: [5.0, 5.0, 5.0]},
        index=idx,
    )
    tree_a = build_one_gage_tree(
        gage_fp=1, rconn={1: [2]}, area_sqkm={1: 100.0, 2: 40.0},
        stop_segs=frozenset(), theta=0.5,
    )
    tree_b = build_one_gage_tree(
        gage_fp=10, rconn={10: [11]}, area_sqkm={10: 100.0, 11: 40.0},
        stop_segs=frozenset(), theta=0.5,
    )
    lag_a = (np.ones(2), np.array([0, 2], dtype=np.int64))  # child reads past its row
    lag_b = (np.ones(2), np.array([0, 0], dtype=np.int64))
    dq_a = np.array([5.0, 5.0, 5.0])                        # no halo: len == nts
    dq_b = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])        # halo: len 2*nts

    alone, _ = apply_scaling_da(
        q_model.copy(), None, {"A": 1}, {"A": tree_a},
        dq_o_by_site={"A": dq_a}, lag_by_site={"A": lag_a}, edge_decay=0.5,
    )
    batched, _ = apply_scaling_da(
        q_model.copy(), None, {"A": 1, "B": 10}, {"A": tree_a, "B": tree_b},
        dq_o_by_site={"A": dq_a, "B": dq_b},
        lag_by_site={"A": lag_a, "B": lag_b}, edge_decay=0.5,
    )
    np.testing.assert_allclose(
        batched[[1, 2]].to_numpy(), alone[[1, 2]].to_numpy(), rtol=0, atol=ATOL
    )
    # And the decayed tail is the NumPy-reference value, not a padded constant:
    # seg 2 at t=1 reads one step past dq_a's end -> 8 + 5*0.5*(40/100)^0.5.
    np.testing.assert_allclose(
        batched[2].to_numpy()[1], 8.0 + 5.0 * 0.5 * (40.0 / 100.0) ** 0.5,
        rtol=0, atol=ATOL,
    )


def test_dq_o_scalar_broadcasts_like_reference():
    """A length-1 dq_o broadcasts across all timesteps, matching the NumPy reference.

    The reference applied ``dq_o[:, None] * mult``; a scalar dq_o therefore spread over
    every timestep. The compiled path cannot broadcast internally, so the adapter must
    expand it -- and the result must equal passing the same value at every timestep.
    """

    q_model, trees, gage_to_fp = _simple_case()  # nts == 3
    _q1, dq_scalar = apply_scaling_da(
        q_model, None, gage_to_fp, trees, dq_o_by_site={"A": np.array([7.0])},
    )
    _q2, dq_full = apply_scaling_da(
        q_model, None, gage_to_fp, trees, dq_o_by_site={"A": np.array([7.0, 7.0, 7.0])},
    )
    np.testing.assert_allclose(dq_scalar.to_numpy(), dq_full.to_numpy(), rtol=0, atol=ATOL)
    # and the gage row itself carries the scalar correction at every timestep
    np.testing.assert_allclose(dq_scalar[1].to_numpy(), [7.0, 7.0, 7.0], rtol=0, atol=ATOL)
