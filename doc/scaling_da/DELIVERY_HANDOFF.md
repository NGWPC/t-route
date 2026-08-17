# Handoff: closing out the scaling DA for delivery

Replaces `CK_TRACE_HANDOFF.md`, which covered only the backward ck trace. That
work is done (section 5); what is left is delivery, and every open item is listed
here with what it costs to leave it and how to check it is fixed.

Written 2026-08-16 after the trace's partition-invariance defect was closed;
section 2 rewritten 2026-08-17 when the API was collapsed to a single flag. Read
sections 1 and 2 before touching anything: the shipped arm changed, so every
delivery number now describes a configuration that no longer exists (section
3.3).

Companion documents: `METHOD.md` is the self-contained journal-style method
description of what is implemented (the one to read first for WHAT the scheme
is). `TIMING_DECISION.md` is the append log of what was measured and retracted
(its header table wins over its body). `HANDOFF.md` and `osse_travel_time.md`
are SUPERSEDED and banner-marked.

---

## 1. State of play

| | state |
|---|---|
| feature | area-scaled ("simple scaling") streamflow DA, opt-in alternative to nudging, NHF only |
| branch | `feat/streamflow-scaling-da`, 6 commits plus an uncommitted working tree |
| tests | 379 passed, 8 skipped, 1 xfailed (`pixi r pytest test/`) |
| lint | new modules zero ruff + zero pyright BY PATH; the repo at large is not clean and is not being cleaned here |
| lag surface | ONE flag, `travel_time_lag`, default true; the backward ck trace is the only estimator |
| `max_loop_size` invariance | 0.0000e+00 over 11,327 segments on four pairs, including a short final remainder |
| analysis skill | re-measured 2026-08-17 under current code: no-lag arm 29.6% -> 11.5% (exact reproduction); the shipped lag arm (clamped, 48 h span) scores 13.1%, unchanged within 4-gage resolution |
| forecast skill | re-measured 2026-08-17: no-lag arm day 1 45.7% -> 32.3% (exact reproduction); the DEFAULT arm cannot run the 24 h-analysis BMI workflow at all, and at the largest span that fits (12 h) scores 45.1%, statistically noda (TIMING_DECISION last section) |
| OSSE timing, trace | re-measured with the clamped trace 2026-08-17: median 2.00 h (was 3.00), mean 3.26 h (was 5.74), drift +0.140 h/km (was +0.328), against 6.00 h / 7.21 h for no timing; TIMING_DECISION last section has the prediction scorecard |
| the 2.4x | RESOLVED: the solver clamps `Km = max(dt, dx/Ck)`; the trace now mirrors it with `min(cn, 1)` (section 3.5) |

Nothing in the working tree is committed. That is deliberate (the branch is being
built piecewise) but it means the delivery state currently exists only on one
machine.

## 2. Decided 2026-08-17: one flag, one estimator, default RESOLVED TO OFF

> **RESOLVED 2026-08-17 (late): `travel_time_lag` defaults FALSE.** The owner
> directed a full re-measurement under final code with a pre-registered rule
> (default off if the lag arm is more than 2 points worse than no-lag at any
> runnable cadence; 2 points = the 4-gage resolution). Everything below in
> this section describing "default on" is the earlier state of the decision,
> kept for the record. The evidence (TIMING_DECISION, final section):
> analysis timing improves (OSSE 2.00 h vs 6.00 h, current), held-out skill
> is unchanged within resolution (13.1% vs 11.5%), and the forecast arm
> scores at the no-DA baseline at EVERY runnable span -- 12 h (45.1), 24 h
> (44.8), and the full 48 h with complete tau coverage (25.20 vs no-lag
> 23.92 vs noda 25.38; day 2 identical to noda). The mechanism is the launch
> edge, not coverage: the final analysis timestep seeds the forecast, its
> upstream correction reads dQ_o(t + tau) past the analysis edge, and that
> read decays to nothing for any material tau. The lag remains the opt-in
> analysis-timing tool; the untimed arm is what delivers the forecast.

The project owner resolved the standing contradiction between the record (which
recommended no timing) and the code (which shipped `lag_direction: forward`).
**The lag is on by default, the backward ck trace is the only estimator, and the
whole surface is one boolean.**

```yaml
streamflow_scaling_parameters:
  travel_time_lag: true      # default
  lag_window_h: 48.0         # the span traced, and so the longest resolvable tau
```

`lag_direction` and `lag_source` are GONE, along with the two estimators they
selected: `_tree_tau_hydrograph` (cross-correlation of the routed hydrographs
plus the significance gate it needed) and `_tree_tau_backward` /
`_reach_transit_steps` (the per-timestep Courant integration). The backward
direction is gone too, and with it the `_lead` / `_prev_dq` history block in
`apply_in_kernel` that existed only to feed it.

What justified deleting rather than keeping them:

| | evidence |
|---|---|
| `courant` | documented dead end: `cn` reports AMBIENT kinematic celerity, median 0.124 m/s over 29.7M segment-steps, which gives 168 h over 39 km |
| `hydrograph` | held-out skill INDISTINGUISHABLE from applying no lag at all: median \|PBIAS\| 11.5% against 11.5%, mean 11.8% against 11.7%, median NSE -0.75 against -0.75. It resolves 9.9% of segments against the trace's 87.7%, and needed a gate to stop it inventing lags from noise (93% spurious before gating) |
| `backward` | `2*tau` late by construction; 12.00 h OSSE error against 6.00 h for no timing |

**Cost of the default, measured same-batch** (Ohio subset, 11,327 segments, two
windows, median of three reps; cross-batch comparison is not safe here, the same
config measured 30.83 s in an earlier batch and 27.78 s in this one):

| arm | total | vs no lag |
|---|---:|---:|
| scaling DA, `travel_time_lag: false` | 27.78 s | -- |
| scaling DA, `travel_time_lag: true` | 29.20 s | +1.42 s, +5.1% |

Two changes went in alongside the collapse, both verified bit-identical
(`0.0000e+00` over 11,327 segments against the pre-collapse trace arm):

- the shift is now **1-D and non-negative**, so the batched COMPILED spread
  kernel is used instead of the NumPy fallback the old `np.repeat` to
  `[n_seg, nt]` forced, and the chunking path is always available;
- `_assemble_cn` and the kernel's `return_courant` export are both **dropped
  once every tree is traced**. Per-window spread cost fell from 3.39 s to 1.83 s
  (against 1.66 s with no lag at all). On Ohio the total barely moves; the point
  is structural, since what was removed is an `[nts, N_seg]` float64 allocation
  per window, and that is a CONUS-scale claim which has NOT been measured.

**What this default costs elsewhere, and it is not small:** enabling the scaling
DA now implies windows of at least `lag_window_h`, because the span is sliced
from the first window. `_scaling_da_spread_h` therefore returns 48 h rather than
the 12 h innovation spread unless the lag is turned off, so `max_loop_size` is
enlarged accordingly on both drivers. A coupling cadence that cannot supply 48 h
must set `lag_window_h` down or `travel_time_lag: false`.

Still true, and belongs in the delivery text rather than in a default: the lag
improves the ANALYSIS (OSSE 3.00 h against 6.00 h) and costs the FORECAST at day
1 (34.9% against 32.3%), because a launch step cannot read an innovation from
after it.

### 2.1 Four defects the collapse's own review turned up, all fixed

An adversarial pass (Codex) on the collapsed diff found four real problems. All
are fixed and re-verified; they are recorded because three of them were invisible
to the measurements that had already returned zero.

1. **The window must cover the span PLUS the spread, not the larger of the two.**
   The innovation is averaged forward over `innovation_spread_h` and the lag then
   reads that average at `t + tau`, so the tail of a window needs raw innovation
   out to `tau_max + spread`. Sizing on `max(...)` left that read past the halo,
   where it fell back to the terminal value, and how far past depended on the
   boundary. `_scaling_da_spread_h` and the BMI equivalent now ADD them (48 h at
   the defaults since the final review set `innovation_spread_h` to 0; 60 h with
   the spread at 12). Verified: `d12` (enlarged 12 -> 24 files),
   `d24` and `d48`, all `0.0000e+00` over 11,327 segments with
   `innovation_spread_h: 12`. Every earlier invariance arm ran with the spread at
   ZERO, which is why none of them could see this.
2. **The trace was still anchored on the first window carrying an observation.**
   Every early return in `apply_in_kernel` is keyed on this window's own
   innovations, and which window first carries one moves with `max_loop_size`, so
   the traced span moved with it too. The fill now happens in `_ensure_trace`,
   before any of those returns, on the first window that has a Courant field at
   all. Pinned by a unit test that drives a window with an all-zero nudge and
   asserts the span is set anyway.
3. **A run that explicitly asked for Courant OUTPUT stopped getting it.** The
   -V5 driver folded the user's `return_courant` request together with the
   trace's, so dropping the trace's request after the cache filled also dropped
   the user's, while the output writer still expected the field. They are two
   flags again. (The BMI driver already ORed the user's request first and was not
   affected, contrary to the review.)
4. **The chunked spread wrote discharge but not depth.** `_scatter_back` carries
   the correction into depth, which is STATE (`h0` seeds the next window and MC
   derives celerity from it); the chunk loop wrote `r[1][:, 4*c0:4*c1:4]`
   directly and skipped it. So a chunked CONUS-scale run handed the forecast a
   different geometry while reporting identical discharge, and the equivalence
   test compared only the discharge columns. The chunk loop now goes through
   `_scatter_back` with a column offset, and the test compares the WHOLE result
   array.

Two further findings from the same pass are open and recorded in section 3.1:
checkpoint/resume does not carry the trace cache, and explicit
`qlat_forcing_sets` still bypass the sizing. The reviewer also argued the lag
should default OFF; that is the decision recorded above, made with the forecast
cost in hand, and it stands.

## 3. Open items, in the order they should be taken

### 3.1 Invariance gaps that are documented but NOT closed

`max_loop_size` is a memory knob and must not change discharge. That now holds
for the file-driven CLI. It does not hold everywhere.

1. ~~**A BMI checkpoint does not carry the traced travel time.**~~ CLOSED
   2026-08-17 (final review, flagged independently by the Codex pass):
   `create_state` now serializes the trace with the identity it was measured
   under (dt, `lag_window_h`, a tree fingerprint) via
   `ScalingDA.trace_checkpoint`, and `load_state` restores it only when the
   identity matches -- every other path (no entry, older state file, changed
   identity) CLEARS the cache and warns, so a stale trace can never survive a
   load. Unit-tested for round-trip, stale-clear, and both identity mismatches
   (`TestTraceCheckpoint`). Still worth doing once with real data: one
   uninterrupted run against one checkpoint/resume run, bit-for-bit, lag on.

2. **Explicit `qlat_forcing_sets` bypass both the enlargement and the fold.**
   `AbstractNetwork.build_forcing_sets` returns user-supplied windows unchanged,
   so neither the span enlargement nor `_fold_short_final_set` runs. A short first
   window then becomes the permanent cached span. `_build_lag` now LOGS A WARNING
   naming the shortened span; it does not fail closed, because a run genuinely
   shorter than `lag_window_h` is legitimate.
   *Decide:* whether an explicit run set with a lag enabled should be a hard
   error, or whether the warning is the contract. Then test that branch.

3. **Incremental BMI coupling cannot run the lag at the default span, and now
   that is the default path.** An update supplying fewer forcing columns than
   `lag_window_h` raises rather than silently shortening the span, which is the
   intended fail-closed behavior. Since 2026-08-17 the lag is on by default, so
   an ngen coupling that drives short updates hits this on a config that merely
   enables the scaling DA. `update_until` drives ONE update and clears its inputs
   afterwards, so the driver never accumulates a span across calls.
   *Decide:* either define cross-update span accumulation, or keep failing closed
   and make the error the first thing an integrator reads (it already names
   `lag_window_h`).
   *Verify:* an end-to-end repeated `update_until` test, not just the existing
   rejection test.
   *Demonstrated live 2026-08-17:* the forecast_mode harness (24 h BMI
   analysis) hit this raise on its first run under the new default, exactly as
   documented; the error text was sufficient to diagnose and act on. And the
   workaround it steers integrators toward is not benign: at the largest span
   a 24 h update can host (12 h), 87.8% of tree segments are unresolved and
   dropped, and the forecast collapses to noda (TIMING_DECISION, last
   section). This item and the default-on decision (section 2 banner) are the
   same problem.

4. **The eager all-trees trace has one fallback that is not partition-proof.**
   Every tree is now traced together on the first window that carries a Courant
   field, which is what removed the old "cached on first appearance" dependence.
   If a site is somehow absent from `self.trees` at that moment and spreads
   later, `_build_lag` warns and applies its correction UNTIMED rather than
   tracing it from a later window. That is the safe direction, but which sites are
   affected could in principle depend on the partition.
   *Fix shape:* decide whether that case is reachable at all (it needs `trees` to
   grow after construction, which the current network build does not do); if it
   is not, make it an error rather than a warning.

### 3.2 Trace refinements, ordered by how likely they are to bite

1. ~~**Diffusive reaches have no `cn`.**~~ Decided 2026-08-17: unresolved IS the
   behavior -- the trace cannot follow what the kernel does not report, so
   diffusive reaches and their subtrees fall out of the correction rather than
   getting a guessed speed. Stated in the `travel_time_lag` config docstring, and
   the unresolved log line now counts these separately as `no_cn` (item 5).

2. **The span is anchored at the START of the run.** Since the invariance fix the
   trace reads `cn[:lag_window_h]` of the first window, so if the event is later
   in the run it reads pre-event AMBIENT celerity. That is the failure mode that
   killed the earlier frozen-field estimator, readmitted in a bounded form as the
   price of partition invariance. A time-varying tau (the 2-D path already
   exists) would be more faithful but costs memory. Measure whether it matters
   before paying for it. NOTE, post-clamp (section 3.5): on any segment where
   ambient celerity already has `cn >= 1`, tau is one step per segment at EVERY
   flow level, so the ambient-anchor bias now lives only where ambient `cn < 1`
   while event `cn` is materially different; the measurement should condition on
   that split.

3. ~~**The 2-D repeat is wasteful and forces the slow path.**~~ Done
   2026-08-17: the shift is 1-D and non-negative, the batched compiled kernel is
   used, and `apply_scaling_da`'s `_numpy_only_lag` guard is now unreachable in
   practice. It is kept as a guard on the kernel's contract, not as a live path.

4. **~~Sub-timestep offsets are dropped at each hop~~, analyzed 2026-08-17 and
   downgraded.** For constant cn the arithmetic is EXACT: `steps - frac`
   telescopes to `1/cn` regardless of where `floor(t0)` lands, so the floor does
   not accumulate error in tau; it only shifts WHICH samples are read, by less
   than one step, in a temporally smooth field. And under the clamp most capped
   segments read a constant effective cn of 1 anyway. Second order at most; not
   worth code until a measurement shows otherwise.

5. ~~**All-zero `cn` and "record too short" are both reported as unresolved.**~~
   Done 2026-08-17: `_tree_tau_trace` returns a per-reason count dict
   (`inherited` / `no_cn` / `dry` / `short` / `lower_bound`) and `_build_lag`
   logs the breakdown with a reading guide (only `short` is a `lag_window_h`
   problem). Tested for the dry-versus-short split and the NaN edge.

6. **Interaction with `max_reach_km` is unstated.** Both mechanisms drop far
   segments. Check whether the trace's `inf` set is a subset of what the distance
   limit already excludes; if so, one of them is redundant on this domain and the
   overlap should be stated rather than discovered later.

### 3.3 Measurements owed, and this is now the largest open item

**Every delivery number was produced under a configuration that no longer
exists.** Two things changed under them:

1. **The default arm changed.** The published headline (analysis 29.6% -> 11.5%,
   forecast day 1 45.7% -> 32.3%) was measured with NO timing, and the
   `lag_source: hydrograph` arms are gone entirely. A config that merely enables
   the scaling DA now runs the traced lag. On the four held-out gages the traced
   lag scores 13.1% median / 13.2% mean absolute PBIAS against 11.5% / 11.7% with
   no lag, and median NSE -0.26 against -0.75 while the MEAN NSE is -0.79 against
   -0.73, which is a median artifact, not an improvement. Four gages cannot
   resolve any of that, which is exactly why the delivery numbers have to be
   re-stated for the arm that actually ships rather than argued about.
2. **The window layout changed.** A final remainder shorter than the span is
   folded into the window before it, and enabling the scaling DA now forces
   windows of at least `lag_window_h`, so arms that used a shorter
   `max_loop_size` ran a different partition than the published one.

Owed, before any of these numbers are quoted again, all with the shipped default:

- ~~`scripts/spread_sweep.py <arm> ...`~~ done 2026-08-17 (`s_off` exact
  reproduction 11.5%/11.7%; `s_tr` clamped 13.1%/13.3%, unchanged within
  4-gage resolution; `s_hyd` retired, deleted code)
- ~~`scripts/forecast_mode.py --analysis-hours 24 --forecast-hours 72`~~ done
  2026-08-17, four arms with the arm stated (`noda` 45.7 / `nudging` 41.2 /
  `scaling_nolag` 32.3, all exact reproductions; `scaling_lag12` 45.1, and the
  shipped 48 h span cannot run under a 24 h analysis at all). See
  TIMING_DECISION's last section, including why the old 34.9 lag-forecast
  number must not be quoted (produced by a silently shortened span that the
  fail-closed raise now forbids)
- the delivery deck's figures, which were generated from those CSVs: STILL
  OWED, and they now need an owner decision first (below) because the arm the
  deck describes depends on it
- ~~the OSSE~~ re-run 2026-08-17 with the clamped trace AND the one-flag
  configs (`configs_extra/osse_trace.yaml` / `osse_none.yaml` migrated to
  `travel_time_lag`); median 2.00 h, mean 3.26 h, drift +0.140 h/km.
  `doc/scaling_da/figures/` now carries this measurement; the pre-clamp table
  is kept at `pi10-subcase4/results/osse_timing_preclamp_backup.csv`

Do NOT re-run selectively and compare against the old CSVs on the arms you
happened to re-run. Either the whole set is regenerated or none of it is. And
state the arm in every table: "the DA" now means the DA plus a traced lag.

The paired configs from this session are a good starting point:
`configs_extra/{p_off,p_tr,s_off,s_tr}.yaml` differ only in
`travel_time_lag`.

### 3.4 Repo state for delivery

1. **Commit the working tree.** Five files carry this session's work
   (`scaling_da_apply.py`, `AbstractNetwork.py`, `troute_model.py`,
   `test_scaling_da_spread.py`, `test_scaling_da_trees.py`) on top of six
   existing commits. `test/troute-nwm/test_scaling_da_spread.py` is still
   UNTRACKED: it holds the trace and invariance tests, so a commit that misses it
   ships the feature without its regression cover.
2. **Decide what happens to the five untracked briefs at `doc/` top level**
   (`scaling_da_pr_review.md`, `scaling_da_lag_decision_brief.md`,
   `scaling_da_lag_review2.md`, `scaling_da_option_analysis.md`,
   `scaling_da_celerity_brief.md`). All five are banner-marked SUPERSEDED and all
   five quote numbers from the circular OSSE. They are working notes, not
   deliverables. Either delete them or move them under `doc/scaling_da/` with the
   banners intact; do not ship them at `doc/` top level where they read as
   current.
3. **The `pi10-subcase4` workspace holds a SEPARATE checkout of t-route**
   (`pi10-subcase4/t-route`, a plain directory inside that repo, not a symlink
   and not a submodule despite what `build_troute.sh` says). It is synced BY HAND.
   Every measurement in this document was produced there, so a change measured
   in one tree and committed from the other is a real hazard. Copy the five files
   across before measuring, and diff them afterwards.
4. `configs_extra/{tw,tf,bd,nd}*.yaml` in that workspace are the invariance
   configs this session added, and they are untracked there too.

### 3.5 The 2.4x: RESOLVED 2026-08-17, fix in the working tree

The synthetic-chain test this section prescribed was run and it identified the
mechanism in one line of the solver: `MCsingleSegStime_f2py_NOLOOP.f90:321`
clamps `Km = max(dt, dx/Ck)`, and the reach lag of the discrete Muskingum
scheme IS K, so the routed perturbation crosses at most one segment per
timestep. The `courant()` export the trace integrates is the UNCLAMPED
`cn = ck*dt/dx`, so wherever `cn > 1` the trace under-estimated tau by exactly
the factor cn. On a uniform chain the routed centroid speed pins at EXACTLY
`dx/dt` once `cn > 1` (0.500 m/s at dx=150, 1.000 at dx=300, dt=300 s),
amplitude-independent, while below `cn = 1` it matches the exported ck within
1%. Full table and method in `TIMING_DECISION.md`, last section.

**Fix (uncommitted, same five files):** `_tree_tau_trace` accumulates
`min(cn, 1)` per sample, mirroring the solver's clamp; regression class
`TestTraceFollowsTheSolverNotThePhysics` in `test_scaling_da_spread.py` pins
cn > 1, cn < 1, and the NaN-sample edge. Re-verified against routed arrivals:
obs/trace 0.963-1.000 across regimes (was 2.47x off at Cn = 2.47). No factor
was fitted; the clamp is the model's own.

**Still owed:** the OSSE and held-out numbers for the trace arm were measured
with the unclamped trace and are stale on top of being stale for the one-flag
collapse; the 3.3 re-measurement covers both. Every traced tau lengthens or
stays the same under the fix, so expect the OSSE timing error to drop and the
unresolved fraction to rise slightly (longer taus hit the span cap sooner).

## 4. What the trace is (so the next session need not reconstruct it)

The upstream leg needs a travel time `tau` from each tree segment to its gage, so
a gage innovation can be applied where the water actually was. The trace computes
it directly:

> From the gage at time `t`, step BACK one timestep at a time, accumulating the
> distance the wave covered in that step, until the accumulated distance crosses
> the reach. Then continue up the parent chain.

Two design choices, both load-bearing:

**(a) Backward through stored history.** Every value read is at a time already
routed, so the estimator never needs data that does not exist. The forward
equivalent needs celerity at `t + tau`; an earlier attempt worked around that by
freezing the field at `t`, which then read AMBIENT celerity in reaches where an
event was passing, and produced 168 h travel times over 39 km.

**(b) It traces the WAVE, not the water.** An innovation is a discharge
perturbation. By the Kleitz-Seddon law a perturbation moves at `c = dQ/dA =
beta*V`, which is `(5/3)V` for a hydraulically wide Manning channel. Tracing
water velocity would over-estimate travel time by that factor. `ck` is NOT
assumed: the kernel computes it for the compound trapezoidal section, including
the wetted-perimeter correction and the overbank blend, and exports it. Because
`cn = ck*dt/dx`, the trace accumulates `cn` until it reaches 1.0 and needs no
reach length at all.

Unresolvable segments return `inf` and are excluded, never guessed at. Two
things make a segment unresolvable: accumulated `cn` never reaches 1.0 within the
span, or the walk runs to the START of the record and crosses only on the oldest
sample it has (that tau is a lower bound, not a measurement).

## 5. Settled this session, do not re-litigate

**The trace is partition-invariant.** Three causes, all fixed:

1. tau was recomputed from every window's own `cn` field. It is now traced ONCE
   over a fixed span (`cn[:n_lag]`, `n_lag = lag_window_h` in steps) taken from
   the start of the first window and cached per `(site, span)`.
2. the walk truncated at the window start, so a longer window resolved MORE
   segments. The fixed span is now also the cap on what may be resolved.
3. **found by adversarial review, not by the first measurement:** a final
   remainder window shorter than the lag is not self-contained, and broke
   invariance for BOTH lag sources and BOTH directions. It is now folded into the
   window before it (`AbstractNetwork._fold_short_final_set`).

Measured, Ohio subset, 11,327 segments, `scripts/invariance.py`:

| pair | before | after |
|---|---|---|
| `tr24` vs `tr48` (the original failing pair) | 5.7381e+01 / 1,620 seg | **0.0000e+00** |
| `tw24` vs `tw48` (span 12 h, neither window enlarged) | not run | **0.0000e+00** |
| `tf24` vs `tf46` (forward lag, 4-file remainder) | 1.2943e+01 / 226 seg | **0.0000e+00** |
| `bd24` vs `bd46` (backward lag, 4-file remainder) | 1.7120e+01 / 221 seg | **0.0000e+00** |
| `nd24` vs `nd46` (lag off, same remainder) | 0.0000e+00 | 0.0000e+00 |

The `nd` row is the control: with the lag off, the short remainder never mattered.

**OSSE after the fix**, known-truth pulse:

| | median abs error | mean abs error | drift |
|---|---:|---:|---:|
| no timing | 6.00 h | 7.21 h | +0.314 h/km |
| trace, per-window (pre-fix) | 4.00 h | not recorded | +0.154 h/km |
| trace, fixed span (now) | **3.00 h** | **5.74 h** | +0.328 h/km |
| the same, excluding one row | 2.50 h | | +0.138 h/km |

1 h at hourly output is one grid cell, so read the median as unchanged, not
improved. The drift reads worse than no timing at all and that is ENTIRELY one
row: the segment at 45.1 km scores a 53 h error whose DA anomaly peaks at 4.5 cms
against a 53.0 cms truth, so the scorer is picking the largest of several small
bumps. It was first written up as a tau saturated at the span cap; **that was
wrong**, and the code change made to test it moved the OSSE by nothing at all.
That change (a walk reaching the start of the record returns `inf`) was kept
anyway, because the failure it blocks is real and it costs six lines, but it is
NOT validated by a measurement and must not be described as if it were.

## 6. Dead ends: do not retry

| approach | why it failed |
|---|---|
| constant celerity (0.8, 1.6 m/s) | indefensible for a continental domain; the old OSSE that flattered it was circular, the injection site was chosen using the constant itself |
| kernel Courant number, unbounded / bounded / ceiling | `cn` reports AMBIENT kinematic celerity; median 0.124 m/s over 29.7M segment-steps gives 168 h over 39 km |
| static geometry celerity (bankfull x 0.25) | needs a fitted reference depth |
| frozen-field characteristic trace | reads celerity at time `t` for every hop, so it reads ambient conditions where an event is passing |
| qlat-space correction with re-route | 20 h OSSE timing error |
| forward innovation window (temporal smear) | unmeasurable across 0/6/12/24 h; also acausal, and its width cannot reach a forecast |
| applying the measured lag BACKWARD | exactly `2*tau` late by construction; 12.00 h against 6.00 h for no timing |
| a fitted correction for the 2.4x | one pulse, one chain, one domain, hourly resolution: this is the fitting the work has refused throughout |

Also settled: **"correct the state before routing window k+1" and "seed `q0` after
window k" are the same edit** (`q0` is the only state crossing a window
boundary), so they are not alternatives. The prognostic-upstream question has
been attempted four ways and closed; see `TIMING_DECISION.md`.

## 7. Where the code is

| what | where |
|---|---|
| the trace | `scaling_da_apply.py::_tree_tau_trace` |
| the tau cache, filled for every tree at once | `scaling_da_apply.py::_build_lag`, the `by_trace` branch |
| "is the Courant field still needed" | `scaling_da_apply.py::_trace_cached`, read by both drivers |
| `cn`/`ck` assembly | `scaling_da_apply.py::_assemble_cn` (`cn` at `r[2][:, 0::3]`, `ck` at `1::3`) |
| kernel celerity | `src/kernel/muskingum/MCsingleSegStime_f2py_NOLOOP.f90`, `subroutine courant` |
| window enlargement, CLI | `AbstractNetwork._scaling_da_spread_h`, applied in `build_forcing_sets` |
| short-remainder fold | `AbstractNetwork._fold_short_final_set` |
| window enlargement, BMI | `troute_model.py`, the `scaling_active` block |
| `return_courant` forcing | `nhf_routing.py` and `troute_model.py`, both gated on `travel_time_lag` AND dropped again once `_trace_cached` is true |
| config | `compute_parameters.py::StreamflowScalingParams.travel_time_lag` and `.lag_window_h` |
| unit tests | `test/troute-nwm/test_scaling_da_spread.py`, `test/troute-network/test_scaling_da_trees.py` |

## 8. Reproducing the measurements

All harnesses live in the `pi10-subcase4` workspace. Copy the five source files
across first (section 3.4).

```bash
# partition invariance: build the arm pair, then compare
pixi run python -m nwm_routing -V5 -f configs_extra/tw24.yaml
pixi run python -m nwm_routing -V5 -f configs_extra/tw48.yaml
pixi run python scripts/invariance.py tw24 tw48        # target 0.0

# OSSE known-truth timing (the only test that can rank estimators)
pixi run python scripts/osse_prep.py                   # regenerates pulse + configs
pixi run python -m nwm_routing -V5 -f configs/osse_truth.yaml
pixi run python -m nwm_routing -V5 -f configs_extra/osse_trace.yaml
pixi run python scripts/osse_timing.py osse_trace

# held-out analysis skill, all arms
pixi run python scripts/spread_sweep.py <arm> [<arm> ...]

# forecast mode (assimilate to launch, then run free)
pixi run python scripts/forecast_mode.py --analysis-hours 24 --forecast-hours 72
```

Two traps that cost time this session:

- **zsh noclobber.** `> logs/x.log` fails with "file exists" and the run silently
  does not happen; use `>|`.
- **stale result directories.** The output writer appends, so a re-run into a
  directory that already holds a run produces "duplicate output timestamps" and
  the comparison reads the OLD run. `rm -rf results/<arm>` first.

The OSSE prep picks the injection site by network DISTANCE, deliberately, so no
celerity is involved in choosing where to test. The older prep chose the site at
whatever distance made `tau = 16 h` at 0.8 m/s and then scored 0.8 m/s, which is
why the historical constant-celerity numbers in older documents are not
trustworthy.

## 9. Measurement discipline for whoever picks this up

This line of work has produced several results that did not survive checking. The
recurring failure modes, all of which cost real time:

- **An equivalence measurement is only as strong as the structural variety of its
  inputs.** The first invariance pairs all divided the forcing evenly, so none of
  them ever produced a short final window, which is exactly where the boundary
  logic differs. Four clean zeros were true and uninformative. State which
  structural cases a measurement covered, not just its result.
- **State the prediction before the run**, and report a missed prediction as a
  result rather than rebuilding the story around the outcome. Two predictions
  missed this session; both are recorded above with what actually happened.
- **A change kept despite measuring nothing must say so.** Keeping it can be
  defensible; presenting it as validated is not.
- **Report the mean alongside the median.** On four held-out gages the median is
  the mean of the middle two, and a "3.4 point improvement" turned out to be a
  median artifact that the mean showed as a degradation.
- **Compare like with like.** A claimed reversal of an earlier result turned out
  to be 24 h windows and signed bias against 8 h windows and median absolute
  PBIAS.
- **The OSSE ranks, four gages do not.** Differences of 2 h at hourly output
  resolution are two grid cells on one pulse; the held-out set cannot separate
  timing mechanisms at all, because PBIAS is nearly blind to when a correction
  lands.
- **Run an independent adversarial pass before believing a claim is closed.** The
  third and largest invariance cause in section 5 came from one, after the
  measurement had already returned zero twice.

## 10. Final pre-PR review, 2026-08-17

Claude review plus a Codex adversarial pass over the full branch diff
(verdict: needs-attention, five findings). Disposition, each verified against
the code before acting:

1. **BMI checkpoint omits the traced travel time** -- accepted; this was open
   item 3.1.1, now CLOSED (see there).
2. **Production preprocessing silently drops a gage whose crosswalked segment
   is a lake id**, while `build_trees` documents and tests the opposite --
   accepted as a two-layer contradiction. The exclusion itself is CORRECT (the
   kernel routes lake segments as reservoir objects; reservoir DA owns them),
   so the fix keeps it, makes it LOUD (a warning naming the excluded gages),
   aligns the `build_trees` comment, and pins the production path with a test.
3. **`innovation_spread_h` defaulted to 12 while its own docstring calls 0
   "the honest default"** -- accepted as a doc/code contradiction, resolved to
   0 (the docstring's written rationale is measured and correct: exact
   confluence background, no next-window data, partition-independent, and the
   0/6/12/24 sweep separated by 1.4 points that four gages cannot resolve).
   The pi10 demo config now says `innovation_spread_h: 12.0` explicitly, so
   the demo arm's published numbers keep their meaning.
4. **Corrected Q with unchanged velocity and power-law depth is hydraulically
   inconsistent** -- REJECTED as a PR blocker. This is the documented,
   measured design (analysis output bit-identical with the depth transform on
   or off; the transform exists to remove the Q/h inconsistency at the
   forecast hand-off), and the exact dQ-to-dy transform is Edge Case 3 /
   Task 3, which the project's acceptance criteria explicitly exclude.
5. **Non-finite scaling parameters pass validation** -- accepted;
   `theta.default`, `by_vpu` values, and `min_flow_cms` now carry
   `allow_inf_nan=False` (the per-tree CSV reader already rejected non-finite
   values).

After the fixes: 389 passed, 8 skipped, 1 xfailed; ruff and pyright clean on
every module the branch adds, by path.

## 11. NumPy reference retired, 2026-08-17

Owner decisions during the final review session, applied in order:

1. The `_kernel_import_error` fallback is GONE: t-route is always installed
   compiled, so `scaling_da.py` imports `spread_trees` unconditionally and a
   missing extension raises at import instead of degrading to the NumPy loop.
2. The whole NumPy reference path is gone with it: `_tree_dq_nodes`,
   `_lagged_dq`, `_pruned_branch_flow`, the `method` parameter and the
   `area_scaling` ablation it selected. `apply_scaling_da` is compiled-only;
   a lag shape the kernel's contract cannot express (2-D or negative shift)
   now raises instead of rerouting.
3. The equivalence fuzz (`test_scaling_da_cython_equiv.py`, 40 seeds x 50
   batched cases) and the reference module are deleted too, per the owner:
   parity was proven and the gate has served its purpose. Both remain in git
   history. The pruned-confluence and lag-edge behaviors those files also
   pinned are re-stated as compiled-path tests through `apply_scaling_da`
   (`test_scaling_da_pruned_confluence.py`).

Suite after: 338 passed, 8 skipped, 1 xfailed; branch-added modules zero ruff
and zero pyright by path.

## 12. FINAL review, 2026-08-17 evening: Codex against METHOD.md plus independent pass

Second Codex adversarial round (this one against METHOD.md as the contract)
returned 8 findings; every one was verified against the code before acting.

Accepted and fixed:

1. **Checkpoint fingerprint was too weak** (site, size, root): a tree with the
   same root and size but a different interior would have accepted the old
   positional tau array. Now a sha256 over the full ordered seg_order and
   parent indices per site; rejection of a same-root/same-size changed
   interior is tested.
2. **Explicit `qlat_forcing_sets` failed OPEN on the span** (open item 3.1.2,
   now CLOSED): a first set shorter than `lag_window_h + innovation_spread_h`
   in a run long enough to supply it now raises before routing
   (`_require_span_covering_first_set`), with the genuinely-short-run case
   still permitted (the trace warns and caps). Tested both ways.
3. **A run with no Courant field silently became untimed** (open item 3.1.4,
   now CLOSED the other way than drafted): with `travel_time_lag` on and no
   cn exported (an all-diffusive domain), the run now fails closed instead of
   quietly running a different estimator; the unreachable per-site cache-miss
   fallback is an error too. Tested.
4. **Synthetic inputs could produce an exit-0 no-DA run**: `synthetic_obs_factor`
   now validates finite and positive, `synthetic_obs_baseline` is required by a
   model validator whenever the factor is set, and a missing or empty baseline
   directory raises at first read.
5. **`with_positions` silently dropped pruned branches** absent from the flow
   frame, quietly inflating surviving siblings' shares; it now warns naming
   the segments.
6. **Entrypoint-dependent spread default**: every executable fallback
   (ScalingDA class attribute and `params.get` defaults in the apply module,
   AbstractNetwork window sizing, BMI driver) now says 0.0, matching the
   schema.
7. **METHOD.md depth overclaim** softened: the transform REDUCES the
   discharge-depth inconsistency where the ratio is a meaningful signal
   (floors and a bounded band guard it); it does not eliminate it, and
   velocity is never modified.

Rejected, with the record:

8. **"Fractional trace phase discarded between reaches"** -- this is item
   3.2.4, analyzed and downgraded on 2026-08-17: the accumulation arithmetic
   is exact for constant cn (telescopes to 1/cn), the floor only shifts WHICH
   samples are read by under one step in a smooth field, and the final shift
   is rint-quantized to whole steps anyway; METHOD.md describes interpolation
   at the final step only, which is what the code does. Codex's crafted
   time-varying case moves the shift by one step, inside the quantization the
   estimator already accepts and far inside the OSSE's measured 2 h median.

Residue sweep also cleaned every stale reference the removals exposed
(area_scaling mentions, "NumPy path" comments, the gage_tree docstring, a
misleading test name). Suite after everything: 344 passed, 8 skipped,
1 xfailed; ruff and pyright zero on every branch-added module by path,
including the previously missed `test_save_state.py`.

## 13. The untimed hand-off, 2026-08-17 (final): the lag's forecast cost is zero

The owner mapped the forecast harness onto the operational NWM cycle (the
lookback is the analysis phase; the state at launch is the DA's whole
contribution) and asked for the hybrid: traced timing in the analysis record,
untimed correction in the state seed. Built as
`ScalingDA.apply_in_kernel(seed_untimed=True)`, passed by both drivers on
exactly the hand-off window; unit test discriminates the two reads with a ramp
innovation.

Registered prediction: the seeded state becomes the untimed arm's state, so
the lag-on forecast reproduces the untimed forecast exactly. HIT bit-for-bit:
max per-gage delta 0.0 (PBIAS and NSE) at both the 24 h and 48 h cadences.

Standing resolution of the default question, superseding the section 2
banner's skill argument: `travel_time_lag` remains FALSE by default for
RUNNABILITY and cost (the span must fit the opening update, fail-closed;
operational 3-28 h lookbacks cannot host the 48 h default span; about 5%
runtime), not for skill. Enabling it where the cadence hosts the span now
costs the forecast nothing and improves the analysis-timing record (OSSE
2.00 h against 6.00 h).
