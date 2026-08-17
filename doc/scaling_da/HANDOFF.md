> **SUPERSEDED 2026-08-15.** This describes a config surface that no longer
> exists: `celerity_mps`, `max_travel_time_h`, `screen_interval_h` and the
> source-trust screen were all removed, and the constant-celerity lag it is
> built around was rejected. Current records: `TIMING_DECISION.md` (what was
> measured and decided) and `DELIVERY_HANDOFF.md` (where the work continues).
> Kept as history only.

# Handoff: scaling DA travel-time lag + `max_loop_size` invariance

Written 2026-08-12; updated 2026-08-13 after the ALWAYS-ON refactor and the OSSE.
Supersedes the mid-work version; the regression described there is FIXED.

---

## 1. State of the tree

**Branch** `feat/streamflow-scaling-da`, 14 commits ahead of `origin/development`,
**unpushed**. HEAD = `35303423 build(docker): consolidate images into the multi-stage
Dockerfile.dev`. The committed series contains NO travel-time lag; the working tree
holds the complete, verified lag + invariance package, ready to commit. A PR body
draft for the committed series is at `doc/scaling_da/PR_BODY.md` (rescued from
session scratch; describes the series WITHOUT the lag).

**The lag and its reach limit are ALWAYS ON (owner decision 2026-08-13): they are
part of the method, not a switch.** `max_travel_time_h` must be positive (default
48); there is no off state, and both drivers run it.

**Verified on the working tree (2026-08-13, always-on code):**

| check | result |
|---|---|
| lagged (48 h horizon) 48 h vs 96 h `max_loop_size` | bit-identical, `max|diff| 0.0`, 0 of 11327 |
| un-lagged 24 h vs 96 h (measured 2026-08-12, pre-always-on) | bit-identical, 0.0 |
| OSSE timing (see §6 and `osse_travel_time.md`) | median error 1.0 h (lag) vs 4.5 h (no-lag reference) |
| full pytest | 364 passed, 8 skipped, 1 xfailed |
| ruff + pyright on `scaling_da_apply.py`, `scaling_da.py` | clean |

The lagged acceptance pair is 48v96, not 24v96: a window below the horizon cannot
carry the one-window-deep halo -- and windows now self-enlarge past it anyway (§2b).

## 2. The invariance package as landed

The handoff's five items, plus a sixth found during verification:

1. **Deferred windows carry their own obs frame.** `build_usgs_df` for window k+1
   overwrites the cached `_loop_obs` before the pending window k flushes, so the
   flush passes the snapshot (`loop_obs=` on `apply_in_kernel`; `_UNSET` sentinel
   because None is a real value). This was the 74.4 cms / 1172 segment regression.
2. **The halo arrives pre-screened.** `ScalingDA.screened_innovation` returns the
   window's innovation masked by its OWN epoch screen; fully screened (or zero)
   sites are explicit zeros, not absent keys (absent falls back to edge decay,
   which a single long window would not do). The driver hands that to the pending
   window instead of the raw `gather_innovation`.
3. **Epochs anchor to the run start.** `_anchor_t0` is captured at the first
   `build_usgs_df`; `_epoch_screen` computes boundaries as run-start + n*interval.
   Unix-epoch anchoring was rejected: for a run starting off the interval grid it
   splits boundary epochs across window edges, which the envelope cannot prevent.
   Run-start anchoring plus the envelope gives exact tiling for ANY t0.
4. **Celerity is a run constant.** `celerity_mps` (config, default 0.8) replaces
   the per-window routed-velocity median; `min_celerity_mps` is gone (nothing left
   to floor). The per-site lag (in_window, tshift) is cached for the run.
5. **Envelope validated on the REALIZED windows; BMI rejects the lag.**
   `validate_window_envelope` (called by the -V5 driver once run_sets exist)
   rejects any non-final window shorter than `max_travel_time_h` or not a whole
   multiple of `screen_interval_h` (the multiple check applies with the lag off
   too -- the epoch screen runs either way; the final window is exempt, it
   clamps at the run end identically under every partitioning). A parse-time
   version was written first and REMOVED after Codex review: `max_loop_size`
   counts forcing FILES (`AbstractNetwork.build_forcing_sets` slices the file
   list, and `stream_output_time` silently enlarges the count), so hour
   arithmetic on the config is wrong for non-hourly forcing in both directions.
   The BMI driver RAISES on a positive `max_travel_time_h` (no deferral/halo
   there; silently zeroing it would publish output under a config that was not
   run -- the build_da_sets failure class).
6. **Spread inclusion is read off the spread array itself** (found when item 4's
   A/B missed its prediction: 38.6 cms / 544 segments remained, all inside window
   1 and growing toward the boundary). Both the `np.any(nud)` candidate gate and
   the "trusted somewhere in this window" survivor set are window-shaped: a site
   with zero (or fully screened) OWN innovation but a nonzero halo must still
   spread, because the lag reads the halo at timesteps inside this window. The
   rule is now `np.any(concatenate(own_masked, halo))`.

**Deliberate deviation from the reviewer note in the old handoff** ("subtract the
full nudge, then mask a copy"): the background subtraction keeps the epoch-MASKED
delta (b=m). Subtracting the full nudge would write background over the kernel's
gage override at scatter time (regressing the pinned no-clobber test) and would
hang the subtraction on a window-shaped set. b=m is the per-timestep choice;
its one cost (another tree's pruned-branch denominator reads corrected flow at a
screened epoch) matches how fully screened sites already behave. Documented at the
mutation site in `apply_in_kernel`.

**Codex adversarial round (2026-08-12), all five findings accepted and fixed:**

| finding | resolution |
|---|---|
| envelope validator compared a FILE COUNT to hours | moved to `validate_window_envelope` over realized run_sets (see item 5) |
| `_stack_dq` padded short innovation rows with the RAW last value, so a no-halo site batched with halo sites read an undecayed constant tail (compiled path only) | padding now pre-applies `edge_decay**k`; algebra composes with the kernel's own decay past the batch end to equal the NumPy path exactly; pinned by a mixed-length equivalence test |
| epoch anchor absent from BMI checkpoint state; off-grid t0 floor-divided silently | `scaling_da_anchor_t0` in `create_state`/`load_state` (backward-compatible `.get`); `_epoch_screen` raises on a t0 not a whole number of dt steps from the anchor |
| BMI warn-and-zero ran a different model than configured | replaced with a hard `ValueError` at BMI init |
| deferral held two windows resident and lost window k's output if k+1 crashed, even un-lagged | deferral now engages ONLY when the lag is on; un-lagged spreads apply immediately (they never read past their window). With the lag on, the two-window residency and the crash-loss boundary REMAIN, as the halo's price -- documented in the config docstring; make pending output durable if this ever matters operationally |

## 2b. The always-on refactor (2026-08-13, owner decision)

The owner ruled the lag and the reach limit are delivery requirements that
"should just work" -- not flags. Changes:

- `max_travel_time_h`: `Field(48.0, gt=0)`; 0 is rejected everywhere (config,
  `ScalingDA.__init__`). The `<=0 -> no lag` branch is gone.
- **Windows self-size.** `AbstractNetwork.build_forcing_sets` (the -V5 path)
  enlarges `max_loop_size` -- a FILE count, converted via the file cadence -- so
  every window covers the horizon and tiles `screen_interval_h`, following the
  `stream_output_time` precedent in the same function. `Model._build_run_sets`
  (BMI) does the same in qlat-column units. `validate_window_envelope` remains
  as the driver-level backstop and should never fire.
- **BMI runs the lag** via the same deferral protocol, scoped WITHIN one
  `update()`/`update_until()` call (`pending` is a local of `Model.run`):
  non-final internal windows wait for the next window's screened innovation;
  each update's FINAL window closes with decayed persistence -- the same
  semantics as the run's final window on the CLI, and consistent with the
  existing per-window seeded-q0 rule ("every window may be the last of its own
  update"). Nothing is ever pending across `create_state`/`load_state`.
  Consequence: an ngen driving hourly updates gets edge closure at every update
  boundary (there is no future innovation inside a 1 h update); large
  `update_until` chunks (the harness pattern) get the full halo treatment.

**Codex adversarial round 2 (2026-08-13), on the always-on refactor:**

| finding | resolution |
|---|---|
| BMI never ran the envelope backstop, so a screen interval not expressible in whole qlat columns silently straddled epochs | `Model.run` now calls `validate_window_envelope` on its materialized run_sets, the same gate as the -V5 driver; a source-level test pins the wiring in BOTH drivers (the build_da_sets two-copies lesson) |
| `stream_output_time` (HOURS) was assigned raw onto `max_loop_size` (a FILE count) -- pre-existing, correct only for hourly forcing | converted via the file cadence before comparing (`ceil(hours*3600/dt_qlat)`), -1 exempt |
| `gt=0` admits `.inf`: `celerity_mps: .inf` zeroes every shift and silently resurrects the removed un-lagged mode | `allow_inf_nan=False` on `max_travel_time_h`, `celerity_mps`, `screen_interval_h`; runtime checks use `math.isfinite`; parametrized parse tests |
| a mid-update exception loses the pending window's unwritten output and leaves advanced q0/t0 (non-transactional) | ACCEPTED AS DOCUMENTED COST, not fixed: a durable staging layer would re-impose the serialization the deferral avoids, and mid-update failure recovery was checkpoint-restart (`load_state`) before the deferral too. Flagged for the owner if operational requirements ever demand transactional updates. |

**BMI/CLI equivalence measured:** the OSSE assimilation config run through the BMI
driver (one `update()`, two 48 h internal windows) is bit-identical to the -V5 run
(`max|diff| 0.0`, 0 of 11327) on matched forcing. First attempt differed because
the BMI fed all 97 forcing files while the CLI's `nts` capped at 96 -- an
experiment mismatch, not a protocol one; matched forcing gives exact equality.

## 2c. Courant-derived celerity: implemented, measured worse, not default

`celerity_source: constant|courant` (default `constant`). The Courant path reads
the kernel's own `cn` (`r[2]`, exported by forcing `return_courant` for ROUTING
only) and uses `transit = 1/cn` timesteps per reach. It is **measured worse** on
the OSSE -- median timing error 1.0 h -> 1.5 h, drift −0.156 -> +0.428 h/h --
because the frozen-field form reads AMBIENT celerity (median 0.149 m/s) where
the correction actually travels at EVENT celerity (~0.95 m/s). Full analysis in
`osse_travel_time.md` §"Round 2". Making it competitive needs a characteristic
trace plus a `cn` halo; the scaffolding (2-D lag arrays through kernel and NumPy
paths, `cn` assembly, per-timestep fallback) is in place for that work.

Codex round 3 findings on this path, all fixed: `_assemble_cn` is now PARTIAL
(a hybrid run's diffusive results carry a scalar `r[2]` placeholder and used to
disable `cn` for the whole run); a reach with no usable length in a network that
HAS lengths now fails closed past the horizon instead of taking zero transit
(zero put its whole subtree at the gage's own instant); and the lag arrays keep
ONE column when tau is time-constant (`nt_lag == 1`, kernel reads column 0 at
every timestep) instead of `[n_seg, nt]` -- the 2-D change had otherwise imposed
~10 GB at CONUS on the DEFAULT path for a feature that measures worse. The
per-tree tau concatenation behind the summary log is now a bounded histogram for
the same reason.

## 3. Remaining owner knobs

- **`celerity_mps` defaults to 0.8 m/s** -- a GEOGLOWS-matching placeholder, NOT
  a decided value. The OSSE (§6) gives the first calibration evidence: applied
  increments land ~0.16 h early per hour of tau on the test chain, implying an
  effective routed celerity ~0.95 m/s there. The constant sets the propagation
  reach (0.8 -> ~138 km at 48 h; 2.0 -> ~346 km).
- **`max_travel_time_h` value** (48 default) is the one knob for both the lag
  horizon and the reach limit.

## 4. What is pinned by tests

`test/troute-nwm/test_scaling_da_in_kernel.py`: the loop_obs snapshot (deferred
window screens its own obs), screened_innovation's epoch masking and explicit
zeros, run-start epoch anchoring (window starting mid-interval), zero-own-
innovation halo spread (the item-6 rule). `test_scaling_da_config.py`: the
realized-window envelope (`validate_window_envelope`: horizon coverage, epoch
tiling with the lag off, final-window exemption, single-window bypass).
`test_scaling_da_cython_equiv.py`: mixed-length innovation rows must not change
any site (the `_stack_dq` decayed-padding fix), plus the existing oracle.
`test_scaling_da_pruned_confluence.py`: the six earlier defects (edge fallback +
decay, chunk overlap, parent-chain tau, negative-shift clip) with `_tree_tau`
now taking a constant celerity.

Not separately unit-tested: the BMI lag rejection (raises at `Model.__init__`;
exercised only by BMI integration runs) and the driver-level flush wiring
(covered by the harness A/Bs, not by a unit test).

## 4b. Full-suite re-measurement (2026-08-13)

Everything rerun after the lag became mandatory. Every difference is in the
scaling-DA arms and nowhere else.

| suite | result |
|---|---|
| pi10 `noda`, `old` arms | bit-identical to `reference/` (max&#124;diff&#124; 0) |
| pi10 at-gage `gage_skill.csv` | byte-identical |
| pi10 `gap_short` outage (AC2) | unchanged (0.00e+00 between arms) |
| pi10 held-out (`new`) | CHANGED -- see below |
| NHF integration (`pytest -m integration`) | 6 passed, 6 skipped (skips = missing prepped data + a pre-existing API refactor) |
| Tier A benchmark vs `golden/` | flow/velocity/depth all `0.000e+00`, PASS |

**Held-out skill on the FINAL mechanism** (this supersedes every lag skill number
in the briefs, which all predate it):

| arm | mean NSE | median &#124;PBIAS&#124; |
|---|---:|---:|
| open loop (no DA) | -3.087 | 29.63 |
| un-lagged (`reference/`) | **-0.368** | 16.98 |
| lagged (current) | -1.117 | **15.62** |

Better bias, worse shape -- the "right amount of water, wrong timesteps"
signature already documented in `scaling_da_lag_decision_brief.md`, now
reproduced on the shipped mechanism. 4 gages, one event: the standing caveat.

**Attribution is measured, not assumed.** The `new` arm had two candidate causes
(lag now on; windows self-sized 24 -> 48). The second is ruled out by the
existing invariance result -- the un-lagged arm is bit-identical across
max_loop_size 24 vs 96 -- so the resize contributes exactly zero and the lag is
solely responsible. The three unchanged rows above are the boundary check: the
lag only reshapes the UPSTREAM spread, so anything measured AT a gage cannot
move, and nothing without the scaling DA can move at all.

**Tier A wall time.** Against the stored `pixi-tierA` baseline the current tree
looked 3.0 s faster (53.28 -> 50.26 s), which clears the 0.24 s noise floor. That
baseline is confounded: different commit (`5d6d8419`), different day, 5 runs vs
3. Measuring HEAD on the same machine in the same session gives 49.84 s, so the
working tree costs **+0.42 s (+0.8%), within ~2 sigma** -- no real change, as
expected since Tier A runs no scaling DA. Do not compare against stored
benchmark JSON from another commit; re-measure the baseline.

**Two findings beyond the numbers, both since acted on:**

1. **The reach limit never binds at 48 h on Ohio** (max tau 43.8 h, `0 of
   11,283,264` cut), so the demo could not show the localization half of the
   requirement at all. Resolved by measuring a horizon sweep and pivoting the
   demo -- see §4c.
2. **Self-sizing forked configured-vs-realized `max_loop_size`.** FIXED:
   `build_forcing_sets` now writes `max_loop_size_realized_h` back into
   `forcing_parameters` and preserves `max_loop_size_configured` alongside
   (the same write-back the function already does for `qts_subdivisions`),
   computed from `run_sets` so it holds in every branch. The enlargement log
   also prints hours, not just file counts.

## 4c. Demo horizon: measured, not inherited

The demo's threshold was suitable for the implementation it was tuned against.
On the current, more physical mechanism the shipped 48 h horizon is inert on
Ohio, so the demo showed the shift and never the reach limit. Swept on the
held-out arm (`pi10-subcase4/scripts/horizon_sweep.py`):

| horizon | segment-timesteps cut | mean NSE | median &#124;PBIAS&#124; |
|---|---:|---:|---:|
| open loop | -- | -3.087 | 29.63 |
| 48 h (shipped) | 0.0% | -1.117 | 15.62 |
| 24 h | 18.2% | -1.117 | 15.62 |
| **12 h (demo)** | **58.5%** | **-1.106** | **15.02** |
| 6 h | 84.2% | -3.021 | 29.63 |

**The skill columns above are largely tautological -- do not lean on them.**
Per gage, 3 of the 4 held-out gages are BIT-UNCHANGED between 48 h and 12 h,
because they sit within 12 h of their source; tightening beyond 12 h cannot
affect them BY CONSTRUCTION. The whole aggregate move is one gage. "58.5% cut
at no skill cost" is close to unfalsifiable on this gage set and was retracted
as evidence about the far field.

**The demonstration is per gage, on the arrays** (`scripts/heldout_reach.py`,
peak |arm - noda| at each held-out feature; 0 means the correction never
arrived):

| gage | source | 6 h | 12 h | 24 h | 48 h | implied travel time |
|---|---|---:|---:|---:|---:|---|
| 03065000 | 03069500 | 22.03 | 22.03 | 22.03 | 22.03 | <= 6 h |
| 03010655 | 03010820 | **0** | 11.18 | 11.18 | 11.18 | 6-12 h |
| 03028000 | 03029000 | **0** | 6.256 | 6.256 | 6.256 | 6-12 h |
| 03068800 | 03069500 | **0** | **0** | 4.183 | 4.183 | 12-24 h |

This is the reach limit working, visibly and per gage: each gage switches off as
the horizon drops below its travel time, and once inside the horizon its
correction is IDENTICAL at every wider setting -- a clean on/off gate, not a
taper. At the demo's 12 h, 03068800 is deliberately out of reach and falls back
to open loop exactly (NSE 0.397 vs no-DA 0.40).

Two earlier statements were wrong and are corrected here: 6 h does NOT put all
held-out gages out of reach (03065000 is reached at every horizon), and the four
gages do not share a single 6-12 h bracket.

**The far-field claim needs the CORRECTION FIELD, not gage skill**
(`scripts/horizon_footprint.py`, `new` at 12 h vs `new48` at 48 h, both against
`noda`):

| horizon | features corrected | summed &#124;dQ&#124; |
|---|---:|---:|
| 48 h | 2,483 | 1,205,046 |
| 12 h | 1,606 | 1,170,615 |

Tightening 48 h -> 12 h drops **35.3% of the corrected features while keeping
97.1% of the correction mass** -- reach falls ~12x faster than mass. State it as
that measured pair, NOT as "the far field is most of the reach": 35.3% is a
disproportionate share, not a majority. It reproduces an independent earlier
measurement (2,482 reaches unlimited vs 2,483 here).

The three DA demo arms are therefore pinned at `max_travel_time_h: 12.0`. A
bonus: 12 h needs no window enlargement (24 h already covers it and tiles the
screen interval), so the demo runs genuine 24 h windows and its "24 h here"
narrative is true again.

**Do NOT port 12 h to the shipping default.** Ohio's deepest tree is 43.8 h;
CONUS trees are far longer, so a horizon that is inert here would bind there.
This sweep says the demo needs a smaller number to be illustrative, and says
nothing about the right operational value -- choosing that from Ohio evidence
would be fitting a continental parameter to one subset.

## 5. Measurement status

- Both invariance pairs are green on the final code (table in §1). Reproduce with
  `~/repos/github/pi10-subcase4`: sync sources into `t-route/`, `pixi run build`,
  run `configs/{h48,d96lag,d24nolag,d96nolag}.yaml`, compare with
  `scripts/invariance.py <armA> <armB>` (committed there this session).
- Harness configs `d24lag`, `cc24`, `cc96`, `fix48`, `lagfloor` are now INVALID
  (envelope, or the removed `min_celerity_mps` key). They were superseded
  experiments; do not resurrect without editing.
- **Skill numbers for the lagged arm are STALE**: every held-out ablation in the
  briefs predates static celerity, the halo screen, and the inclusion rule.
  Re-measure before citing any lag skill number.
- **The OSSE has now RUN (2026-08-13)** -- see §6.

## 6. The OSSE (timing validation)

Full writeup: `doc/scaling_da/osse_travel_time.md`; artifacts under
`doc/scaling_da/figures/` (`osse_timing.png/.csv`, `osse_meta.json`); harness
scripts `pi10-subcase4/scripts/osse_prep.py` / `osse_timing.py`.

A +50 cms x 3 h pulse injected 16.0 h (static tau) up gage 03031500's tree; the
truth run's output served as synthetic observations for an assimilation run on
unpulsed forcing. Along the 16-flowpath scored chain, the applied increment
peaks within **median 1.0 h** of the routed pulse's true peak (far half: 1.5 h),
against a no-lag reference error equal to the routing time to the gage (median
4.5 h, 10.5 h on the far half, growing to 12 h). The lag places the correction
at the right TIME; the error drift (-0.156 h/h of tau) is the celerity
calibration signal (§3). One chain, one event, hourly output -- validation of
the mechanism, not a skill claim.

## 7. Background reading

- `doc/scaling_da_option_analysis.md`, `doc/scaling_da_lag_decision_brief.md`,
  `doc/scaling_da_lag_review2.md`, `doc/scaling_da_celerity_brief.md` -- the
  review rounds that produced the package (their measurement tables are
  superseded by §1).
- Literature summary lives in the decision brief; the standing conclusion: no
  cited method shifts a discharge correction by an estimated travel time; the
  strongest long-term direction is qlat-space correction (option 4), which
  dissolves the prognostic problem.
