# Upstream timing for the scaling DA

> **READ THIS FIRST.** This document is an APPEND LOG written as the work
> happened, so earlier sections state things that later sections retract. The
> current state is in the last three sections plus the list below. Where an
> earlier section disagrees with this list, this list wins.
>
> **Retracted or superseded, do NOT quote from the body:**
>
> | claim in the body | correction |
> |---|---|
> | median measured lag **15.2 h** | that statistic aggregated INHERITED lags; measured-only median is **6.2 h**. It was then used to argue the 48 h span was needed, which was circular. |
> | Ohio **9.4%** (forward) / 8.2% (backward) | measured BEFORE the significance gate. After gating, both give **11.5%**, identical to no timing. |
> | measured-lag segments hold **32.7%** of correction mass | counted `tau > 0`, which includes inherited lags. Correct figure is **20.0%**. |
> | "timing cannot change the forecast initial condition" | true for the SPREAD (tail padding), FALSE for the LAG (edge decay moves the final value). |
> | the recommendation to ship **no timing treatment** on 4-gage bias | superseded: the OSSE ranks timing, four gages cannot. Current recommendation is still `lag_direction: none`, but for the FORECAST reason below, not that one. |
> | "the 2.4x is left unexplained", and every OSSE/held-out number for the ck trace | EXPLAINED and fixed 2026-08-17 (last section): the solver clamps `Km = max(dt, dx/Ck)` while exporting unclamped cn; the trace now accumulates `min(cn, 1)`. All trace-arm numbers in this document were measured with the UNCLAMPED trace and are stale. |
>
> **Current state (2026-08-17 final):** the base area-scaled DA delivers
> (analysis 29.6% -> 11.5%, forecast day 1 45.7% -> 32.3%). All timing is
> OPTIONAL and defaults OFF. The lag improves the ANALYSIS record (OSSE
> 2.00 h vs 6.00 h) and, since the untimed hand-off (final section), costs
> the FORECAST NOTHING: the lag-on forecast equals the untimed arm's within
> float32 numerical noise (max per-gage delta 5e-5 PBIAS points) at every
> runnable cadence, engagement-verified after an earlier vacuous run of this
> comparison was caught and corrected (addendum section). It stays off by default for runnability
> (the span must fit the opening window, fail-closed; operational 3-28 h
> lookbacks cannot host the 48 h default span) and about 5% runtime, not
> for skill. The earlier "34.9% vs 32.3%" forecast cost is superseded. `lag_source: trace` is built
> and, since 2026-08-16, partition-invariant: it is traced once over a fixed
> `lag_window_h` span and cached, so `max_loop_size` no longer moves discharge
> (`0.0000e+00` over 11,327 segments, was 57.4 m3/s over 1,620). The same review
> pass found a THIRD cause that had nothing to do with the trace: a final
> remainder window shorter than the lag broke invariance for BOTH lag sources and
> BOTH directions (17.1 and 12.9 m3/s), and is now folded into the window before
> it. The cost of the fix is that the span is read at the START of the run, so
> tau reads ambient celerity where the event has not arrived yet. Three gaps
> remain open (BMI checkpoint/resume, explicit `qlat_forcing_sets`, short BMI
> updates). See `DELIVERY_HANDOFF.md` section 5.
>
> **2026-08-17, the API is now ONE FLAG.** `travel_time_lag: true|false` --
> default TRUE when first collapsed, RESOLVED TO FALSE later the same day by
> the full re-measurement (see this document's FINAL section: lag arm at the
> no-DA baseline at every runnable forecast span; pre-registered rule fired).
> The backward ck trace is the only estimator and `dQ_o(t + tau)` the only
> direction. `lag_direction` and `lag_source` are gone, and so are the
> hydrograph and courant estimators: on the four held-out gages the hydrograph lag
> was indistinguishable from applying no lag at all (median and mean absolute
> PBIAS 11.5% / 11.8% against 11.5% / 11.7%, median NSE -0.75 against -0.75). The
> shipped default therefore now includes a lag, which the numbers in this document
> do NOT: every held-out and forecast figure here was measured with no timing, and
> they have to be re-stated for the arm that ships. Cost of the lag, measured
> same-batch on Ohio: 29.20 s against 27.78 s, +5.1%. A window must now cover
> `lag_window_h + innovation_spread_h` (48 h at the defaults since the final
> review set `innovation_spread_h` to 0; 60 h with the spread at 12).


Working record for the decision that closes out the celerity work. Not part of
the PR.

## The question

The area-scaled DA injects a gage innovation at the gage segment and spreads it
upstream over that gage's contributing tree. The upstream leg has to decide WHEN
a gage innovation applies at an upstream segment. A segment's present water
reaches the gage later, so the correct quantity is `dQ_o(t + tau)`, forward in
time. Everything below is about how to get tau, or how to avoid needing it.

## Every mechanism measured

OSSE: a known lateral-inflow pulse injected 16 h upstream of a gage; the applied
increment's peak time scored against the routed pulse's true peak.

| mechanism | OSSE median abs peak error |
|---|---:|
| point shift, constant 0.8 m/s | 1.00 h |
| point shift, constant 1.6 m/s | 2.00 h |
| point shift, kernel Courant number, unbounded | 1.50 h |
| point shift, Courant bounded [0.5, 5.0] m/s | 1.50 h |
| point shift, Courant ceiling 1.0 m/s | 1.00 h |
| point shift, static geometry celerity | 1.00 h |
| qlat applied at obs time, then re-routed | 20.00 h |
| forward innovation window, 12 h | 9.50 h |
| backward characteristic trace | not scored; tau exceeds the window (see below) |

Held-out gages (withheld from assimilation entirely, scored on their own
observations, 24 h spin-up excluded, Ohio subset, 4 gages):

| arm | median abs PBIAS | median NSE |
|---|---:|---:|
| no DA | 29.6% | -0.32 |
| point shift, constant 0.8 m/s, 12 h horizon | 15.0% | -0.47 |
| forward window 0 h (no timing treatment) | 11.5% | -0.75 |
| forward window 6 h | 12.1% | -0.73 |
| forward window 12 h | 12.0% | -0.71 |
| forward window 24 h | 10.7% | -0.45 |

Measured after the confluence-denominator fix in finding 5. The 0 h arm came
back at 11.5% for a second time, which is the check on that fix: at zero width
the raw and smoothed innovations are the same array, so it has to be a no-op
there and it was.

The four forward-window arms span 1.4 points of median absolute PBIAS. Four
gages cannot resolve that. The DA itself is worth about 18 points; the timing
treatment is worth nothing measurable.

Prediction made before the sweep and MISSED: the smear was expected to preserve
PBIAS and cost NSE, since it conserves the time integral and attenuates the
peak. The widest arm (24 h) had the best NSE of the four. Recorded as a miss
rather than rewritten.

## Why the constant was dropped

Rejected by the user as indefensible at continental scale, which is correct: one
speed cannot describe a network spanning headwater streams to the Ohio
mainstem. The Courant-number alternatives all failed for a diagnosable reason.
`cn` reports the KINEMATIC celerity, while Muskingum-Cunge matches numerical to
physical diffusivity and routes a DIFFUSION wave whose peak lags it. `cn` is
also wrong in both directions on the SAME reaches: fast where an event is
passing, slower than any usable constant at ambient flow. No pair of bounds
fixes both.

## The backward characteristic trace, and why it also fails

Requested on the grounds that tau computed from ALREADY-ROUTED results is more
trustworthy than any picked celerity. The tractability half of that argument is
correct and was implemented:

    tau(s,t) = tau(p,t) + transit(p, t - tau(p,t))

Walking upstream from the gage, each hop reads the celerity field at a strictly
EARLIER column than the one before, so every read is at a time already routed and
in memory. The forward equivalent needs `transit(p, t + tau)`, which is exactly
why the earlier attempt froze the field at column t. No fallback celerity is
supplied: a reach the router cannot speak for is dropped with its subtree, since
substituting a constant is what this whole line of work exists to avoid.

Both sign conventions were measured over the same trace, with the spread at 0 so
the comparison is the timing mechanism alone:

| arm | median abs PBIAS | median NSE |
|---|---:|---:|
| no DA | 29.6% | -0.32 |
| no timing treatment | 11.5% | -0.75 |
| traced backward, `dQ_o(t - tau)` | 17.8% | -0.32 |
| traced forward, `dQ_o(t + tau)` | 24.0% | -0.19 |

The two are NOT the same mechanism seen from two index origins; they differ by
2*tau and measure differently.

**Both are worse on bias than applying no timing treatment at all**, and the log
says why. The traced travel time comes out at median 168 h, p90 1627 h, over tree
distances of median 39.1 km and p90 87.0 km. That is an implied celerity of
0.065 m/s at the median and 0.015 m/s at p90, against the ~0.95 m/s the OSSE
measured for the event wave. With a 96 h window and a median tau of 168 h, MORE
THAN HALF the corrected segments have their correction shifted entirely off the
window, where edge decay removes it. The arms are weak because the correction
mostly never lands.

This is the fifth independent measurement of one underlying fact: `cn` reports
the KINEMATIC celerity of the AMBIENT flow, which in small upstream reaches is
an order of magnitude below the speed at which a flood wave, and therefore an
innovation, actually travels.

The backward trace makes it worse rather than better, and for a structural
reason. Its well-posedness comes precisely from reading the PAST, and the past
of an event is the calm that preceded it, which is the slowest flow state on the
record. Measured against the frozen field's 0.149 m/s implied celerity, the
backward trace reads 0.065 m/s. **Well-posedness and physical relevance point in
opposite directions here**: the direction in which the trace is computable is the
direction in which it reads the wrong flow state. That is not fixable by choosing
a better estimator of the same quantity.

## Why distance-scaling the window is not the fix

The obvious refinement is to widen the window with distance, so near segments
approach persistence and far ones get the full smear. It does not survive
inspection: `T_s = T * d / D` has `D / T` in it, which is a speed. Fixing both
parameters fixes a celerity exactly, and hides it behind a ratio.

This generalises. Any rule that makes the timing treatment vary across segments
must convert distance into time, and that conversion is a celerity. A uniform
window is not a crude approximation to a better distance-aware scheme; it is the
only member of the family that is genuinely celerity-free.

## Findings that decide it

1. **The two defaults do not describe the same interval.** The claim that the
   window covers the arrival time because `max_reach_km` bounds it is
   dimensionally empty: 200 km is 69 h at 0.8 m/s, not 12 h. The width has no
   derivation, only an interpretation (how long an innovation stays
   informative). Config text asserting the bound has been corrected.

2. **The forecast hand-off cannot see the width.** The tail is padded with the
   last value, so `smooth(dQ)[-1] == dQ[-1]` for every T (verified: 7.000000 at
   T = 0, 6, 12 and 24). `new_q0` snapshots only the final routed timestep, and
   the spread seeds state only on the final window. So every width produces a
   bit-identical warm state. A nonzero width can change retrospective output and
   can never change a forecast initial condition.

   Confirmed end to end on the Ohio subset rather than by reading the code.
   Comparing the T = 0 and T = 24 h runs over all 11,327 segments:

   | comparison | max abs diff | segments differing |
   |---|---:|---:|
   | final timestep | 0.000000e+00 | 0 |
   | all timesteps | 1.325882e+02 | 1836 |

   The width moves 1836 segments by up to 132.6 m^3/s during the run and moves
   the hand-off state on none of them.

3. **The operator is acausal.** At time t it reads observations through t+T. In
   a real-time run those do not exist, so the mechanism degrades to persistence
   exactly at the analysis edge, which is the operationally important end.

4. **Volume survives only in the interior.** A 12-unit impulse under a 4 h
   window integrates to 2.4 at the series start, 12.0 inside, and 36.0 at the
   end. The leading loss is inherent to a causal forward mean; the trailing gain
   is the price of the padding that gives property 2. Both are now pinned by
   test rather than claimed away.

5. **The confluence background was corrupted by exactly the smoothing.**
   Subtracting the smoothed value left `background + raw - smoothed` as the
   flow-ratio denominator, so every upstream branch received a proportionally
   smaller share (background 10, raw 10, smoothed 2 delivered 1.11 instead of
   2.00). Fixed by subtracting the raw nudge and restoring the difference on the
   gage column afterwards. At T=0 the two are the same array and the question
   does not arise.

6. **Invariance needs a guard that only the BMI had.** The halo is one window
   deep, so a window shorter than T leaves its own tail on persistence and
   results depend on the partition. The 48 h vs 96 h pair verifies two friendly
   partitions, not the requirement. The -V5 path had no guard at all; one has
   been added to match the BMI. At T=0 nothing reads past a window boundary and
   the requirement disappears.

## Recommendation

Ship `innovation_spread_h = 0` as the default: the correction applies at the
observation time, localized to `max_reach_km` of network distance.

It measures as well as every nonzero width, and at zero the four defects above
are not mitigated but absent. It is one parameter instead of two, and the one
that remains is a distance, which is known exactly.

The cost is real and should be stated plainly: with T=0 there is no timing
treatment, and the delivery was scoped to include one. Finding 2 is what makes
that acceptable rather than a retreat, since no member of this family can affect
a forecast initial condition under the current execution order.

Keeping the parameter (defaulted to 0) preserves the option without asserting a
timing model nothing here can support. Removing it entirely would also delete the
halo, the deferred-window hold in both drivers, and the enlargement guards.

## The travel time is MEASURED, not estimated (current mechanism)

Every celerity attempt above estimated the wrong quantity. `ck` is the celerity
of the flow a reach is CURRENTLY carrying, and the router reports it honestly:
median 0.124 m/s, p10 0.061, p90 0.487, max 5.27 over 29.7M live segment-steps.
A gage tree is mostly small headwater reaches at baseflow, so integrating
`dx/ck` along one gives 168 h over 39 km. An innovation does not travel at
baseflow celerity; it rides the flood wave, which is where that same 5 m/s tail
comes from. The trace was arithmetically right and physically wrong.

The routed hydrographs already contain the arrival, so the lag is now measured
off them: each tree segment's routed hydrograph is cross-correlated against the
gage's, and the lag maximising the correlation is that segment's travel time.
Nothing is estimated and no celerity appears. On Ohio it returns a median of
15.2 h, which over the tree's distances is of order 1 m/s: a flood-wave speed.

Four properties, each forced by a measured failure rather than chosen:

1. **Differenced, not levels.** An innovation is a perturbation, so what must be
   timed is the arrival of CHANGES. Correlating levels locks onto whatever slow
   trend dominates: outside an event every hydrograph is in gentle recession,
   those recessions correlate best at the largest lag available, and the
   measurement silently returned the ceiling for EVERY segment in 2 of 3
   windows. Differencing recovers the known lag exactly on a synthetic and
   returns zero on a pure recession.
2. **Non-decreasing upstream.** Water cannot reach the gage sooner from further
   away, so a segment peaking earlier than its parent is holding noise.
3. **A flat segment inherits its parent.** No signal means no measurement; an
   argmax over noise is not one.
4. **Measured ONCE per run over a fixed span** (`lag_window_h`, default 48 h),
   never over the forcing window. The correlation span decides the answer, so a
   per-window measurement made `max_loop_size` change discharge: 96 h windows
   returned a median lag of 19.8 h against 15.2 h for 48 h windows, and 1807 of
   11327 segments differed by up to 43.6 m^3/s.

Applied BACKWARD, `dQ_o(t - tau)`. That direction needs innovations from BEFORE
the window, which were computed minutes ago, so its halo is the previous
window's tail carried forward: no deferred output, no held results, no
acausality. The forward direction needs innovations from after the window and
requires the whole deferred-window machinery to get them. Backward is strictly
cheaper and strictly more causal.

`max_loop_size` invariance: 0.0 over 11327 segments at 48 h vs 96 h, after the
backward halo. Without it the same pair differed by 43.6 m^3/s over 1680
segments, because every window's early steps clipped to their own first value.

Held-out skill (Ohio, 4 gages withheld entirely, 24 h spin-up excluded):

| arm | median NSE | mean NSE | median abs PBIAS | mean abs PBIAS | improved |
|---|---:|---:|---:|---:|---:|
| no DA | -0.32 | -3.09 | 29.6% | 30.2% | 0/4 |
| no timing treatment | -0.75 | -0.73 | 11.5% | 11.7% | 4/4 |
| innovation spread 12 h | -0.71 | -0.77 | 12.0% | 12.3% | 3/4 |
| measured lag, backward | -0.56 | -1.16 | 8.2% | 9.8% | 4/4 |
| measured lag, forward | -0.21 | -0.81 | 20.7% | 17.4% | 3/4 |

The backward measured lag gives the best bias on both statistics and improves
every held-out gage. NSE is mixed: better than no-timing on the median, worse
on the mean.

### What must NOT be done with `lag_window_h`

It changes results. The same arm measured 5.0% median absolute PBIAS with an
effective 24 h span and 8.2% with the shipped 48 h. It must therefore NOT be
tuned on this 4-gage set, which is the whole domain available: that would be
fitting a parameter to the only data able to score it. 48 h is chosen because a
correlation is supported over half its span, so it caps resolvable travel time
at 24 h, which covers the Ohio trees' measured 15.2 h median with margin. Any
future tuning needs a domain the tuning has not seen.

### Remaining limits, stated rather than hidden

- 8695 of 19589 segments have no signal to measure in a given window and inherit
  their parent's lag. Some are genuinely flat headwaters; the split has not been
  characterised.
- 45 segments hit the 24 h correlation ceiling and were truncated to it. They
  are counted and warned about rather than reported as measured.
- Ohio is one small VPU with 4 held-out gages. It can show that the method beats
  open loop; it cannot select between mechanisms, and was not used to.

## OSSE validation against known truth: FORWARD, and the old table was circular

The OSSE injects a +50 cms x 3 h pulse at a known place and time, so the truth
run's routed anomaly gives the ACTUAL arrival at every point on the chain.
`err_lag` is then the gap between where the DA put its correction and where the
water actually was.

**The old OSSE was rigged in favour of the constant.** `osse_prep.py` chose the
injection site at whatever distance made `tau = 16 h` AT 0.8 m/s, then the
comparison scored 0.8 m/s and found it excellent (1.00 h). The constant was
graded on a site defined by itself. Every celerity row in the table at the top
of this document inherits that flaw. The prep now picks the site by network
DISTANCE (55 km up gage 03031500, a 173-segment chain), with no celerity
anywhere.

On that honest layout, with the measured lag:

| arm | median abs timing error | far half of chain | drift |
|---|---:|---:|---:|
| no timing treatment | 6.00 h | 12.00 h | +0.314 h/km |
| measured lag, BACKWARD | 12.00 h | 18.00 h | +0.518 h/km |
| measured lag, FORWARD | 2.00 h | 6.00 h | +0.119 h/km |

True routed travel time along the chain: 1.0 to 19.0 h over 0.0 to 52.6 km.

**Backward is exactly 2x worse than applying no timing at all**, which is what
the direction argument predicts and what settles the sign question for good. The
gage sees the anomaly at T; the upstream segment experienced it at `T - tau`.
Applying `dQ_o(t - tau)` lands the correction at `T + tau`, so `2*tau` late,
while no timing lands at T, only `tau` late. Forward lands it where the water
was, and its drift is 2.6x flatter: the applied lag tracks the routed travel
time at every distance, which is the property being tested.

### A defect this found: the forward halo was being truncated

Chasing why forward looked weak on Ohio turned up `full = smooth(...)[:nt]`,
which sliced the halo off the innovation. Every forward-shifted segment was
therefore reading a clipped, edge-decayed last value instead of the NEXT
window's real innovation. Fixed by keeping an extended `ext` (own window plus
halo) for the shift to read while `full` stays window-shaped for the background
subtraction. The OSSE could not have caught this: it runs one 96 h window, so
there is no next window and no halo either way.

Effect on Ohio, forward arm: median absolute PBIAS 20.7% -> 9.4%.

## Where it lands

| arm | OSSE timing error | Ohio median abs PBIAS | Ohio median NSE |
|---|---:|---:|---:|
| no DA | -- | 29.6% | -0.32 |
| no timing treatment | 6.00 h | 11.5% | -0.75 |
| measured lag, backward | 12.00 h | 8.2% | -0.56 |
| measured lag, forward | 2.00 h | 9.4% | -0.50 |

`max_loop_size` invariance is 0.0 over 11327 segments for BOTH directions.
370 tests pass; ruff and pyright clean.

Forward wins decisively on the instrument built to judge timing, and ties with
backward on bias (8.2 vs 9.4 across four gages is not a difference worth
claiming). Both timing arms beat the no-timing arm on both instruments, so the
conflict that existed before the halo fix is gone.

**Shipped shape**: `lag_source: hydrograph`, `lag_direction: forward`,
`innovation_spread_h: 0`, `max_reach_km: 200`, `lag_window_h: 48`.

An earlier version of this document recommended no timing treatment at all, on
the strength of a 4-gage bias metric. That recommendation is withdrawn: PBIAS
integrates over time and is close to blind to WHEN a correction lands, so it was
never competent to answer the timing question. The known-truth OSSE is, and it
says a measured forward lag is three times better than applying none.


## Forecast mode, and the measured impact of the depth transform

The proposal's task 4: spin up with assimilation, save the channel state, reload
it for an observation-free forecast, compare against the existing scheme. Run
through BMI (the -V5 driver writes no restart): 24 h analysis, then 72 h free run
with no observations, scored at the 4 gages withheld from assimilation entirely.

| lead day | no DA | nudging | area-scaled |
|---|---:|---:|---:|
| median abs PBIAS, day 1 | 45.7% | 41.2% | **34.9%** |
| day 2 | 25.4% | 22.6% | 25.3% |
| day 3 | 30.7% | 31.0% | 30.7% |
| median NSE, day 1 | -12.63 | -8.75 | **-5.82** |
| day 2 | -4.20 | -3.14 | -4.21 |
| day 3 | -0.73 | -0.67 | -0.74 |

Day 1 is where the proposal says the value is ("particular focus on days 1-3,
when the QPF has more value"), and the area-scaled arm is decisively best there:
24% less absolute bias than open loop and better than nudging on both metrics.
By day 3 all three converge, which is what a one-shot state perturbation should
do against MC's `C3*qdp` memory.

Prediction that MISSED: nudging was expected to leave the held-out gages
untouched, as it does in analysis mode (`nse_old == nse_noda` exactly there). In
forecast mode it improves days 1 and 2, because its at-gage correction
propagates DOWNSTREAM through routing during the analysis and that lands in the
saved state too.

### The depth transform: real, isolated, and small

Carrying the discharge correction into depth (`h_new = h_old * (Q_new/Q_old)^0.6`)
was measured three ways:

1. **Analysis output: bit-identical.** Transform on vs off over a full CLI run,
   `max|diff| 0.0` across 11,327 segments. This follows structurally: the
   upstream spread is applied AFTER routing within a window, `new_q0` follows it
   only on the final window, and the -V5 driver writes no restart. So the change
   carries ZERO regression risk for analysis runs, and the forecast hand-off is
   the only place it can act.
2. **Forecast output: real but small.** 1,979 of 11,327 segments differ, largest
   absolute difference 0.0955 cms, largest relative change 2.07%, and NO segment
   differs by more than 0.1 cms.
3. **Held-out skill: unchanged** to three significant figures at every lead day.

The coefficient sensitivity is genuine in isolation: at the p99 depth ratio of
1.52 the wave celerity rises 32%, C1 by 37% and C3 falls 57%. But only about 1%
of segment-steps move that far, and depth enters MC by modulating how the
already-corrected discharge propagates rather than by changing its magnitude, so
the end-to-end effect is the 2% above. An earlier note in this document implied
the impact would be material; the measurement says it is not, on this domain.

Keep it because it removes a genuine inconsistency (the forecast was inheriting
corrected discharge against uncorrected depth, and MC derives celerity and X from
depth) at no measured cost, not because it buys skill.

### A deployment limit the harness exposed

`lag_window_h: 48` forces every window to 48 h, and under BMI the update IS the
window. The first forecast run failed outright:

    scaling DA: max_loop_size enlarged 24 -> 48 forcing columns
    MemoryError: available memory caps the run window at 24 forcing timesteps,
    below the configured max_loop_size of 48

ngen drives t-route in short updates, so the shipped default cannot run there.
Bisected: at `lag_window_h: 24` the same run proceeds. But that setting caps the
resolvable travel time at 12 h (a correlation is supported over half its span)
while the measured median travel time on this domain is 15.2 h, so most lags are
then censored at the ceiling. The timing extension is squeezed between the chunk
size operations can afford and the span its own measurement needs. The base
area-scaling method carries no such constraint.


## The forward lag helps the analysis and HURTS the forecast

Asked whether persisting the deferral halo across BMI updates would let the lag
work at ngen cadence. It would not, and working out why exposed a structural
cost that had not been measured.

`_lagged_dq` reads `dq_o[t + tau]`. Within a window that is free, because the
spread is a post-routing pass and the whole window's innovation already exists.
Past the window end it needs the NEXT window's innovation, which is the halo,
and the halo is exactly ONE window deep and local to a single BMI `Model.run()`
call. Under ngen every update has one window, always the final one, so the halo
never exists. At the measured 6.2 h median lag and dt = 300 s (74 steps):

| window | reads real innovation | clipped | weakest correction |
|---|---:|---:|---:|
| ngen 1 h | 0 / 12 | 12 | 4.6% |
| ngen 6 h | 0 / 72 | 72 | 4.6% |
| 24 h | 214 / 288 | 74 | 4.6% |
| 48 h | 502 / 576 | 74 | 4.6% |
| 24 h + halo | 288 / 288 | 0 | 100% |

Persisting the halo across updates does not fix it. One update of halo buys one
update of lookahead against a 6.2 h lag, so covering it means deferring about
seven 1 h updates, which is an output latency equal to the travel time. And the
FINAL timestep can never be fixed at all: the forecast seeds from it, and its
lagged read needs `dQ_o(t_end + tau)`, which does not exist at `t_end` by
definition. It clips and decays to `0.9592^74` = 4.6%.

So the lag strips the correction out of exactly the state the forecast inherits.
Measured (24 h analysis, 72 h free run, 4 withheld gages):

| lead day | no DA | scaling, lag forward | scaling, lag none |
|---|---:|---:|---:|
| median abs PBIAS, day 1 | 45.7% | 34.9% | **32.3%** |
| day 2 | 25.4% | 25.3% | **24.5%** |
| day 3 | 30.7% | 30.7% | 30.7% |
| median NSE, day 1 | -12.63 | -5.82 | **-4.79** |

The DA itself is worth 45.7 -> 32.3; the lag gives part of it back. The direction
was PREDICTED from the decay arithmetic before the run, so it is a confirmed
mechanism rather than a fishing result, though the magnitude on four gages is
small and should not be quoted as more than a direction.

**The timing extension therefore splits by product:**

- retrospective/analysis output: 3x better correction placement against known
  truth (2.00 h vs 6.00 h);
- forecast: measurably worse than no timing, structurally, because the launch
  timestep is the one place the required future innovation cannot exist.

Recommended shape: ship it available and documented, defaulting to `none`, with
both the retrospective benefit and the forecast cost stated. A forecast-oriented
delivery should not default it on.


## The DA's reach split: measured cost to the decomposition and to wall time

The in-kernel override must land on a reach boundary or it never propagates
downstream, so the DA passes its gage set to the plan builder and reaches split
there. That is required for correctness; this is what it costs.

Decomposition, Ohio topology (103,558 routed segments):

| gage set | reaches | mean length | single-segment | vs baseline |
|---|---:|---:|---:|---:|
| none | 10,029 | 10.33 | 205 (2.0%) | - |
| real (23) | 10,058 | 10.30 | 228 (2.3%) | +0.3% |
| 0.26% synthetic (269) | 10,508 | 9.86 | 523 (5.0%) | +4.8% |
| 1.0% (1,036) | 11,906 | 8.70 | 1,449 (12.2%) | +18.7% |
| 5.0% (5,178) | 19,176 | 5.40 | 6,502 (33.9%) | +91.2% |

CONUS is roughly 7,000-8,000 USGS gages over ~2.7M flowpaths, about 0.26%, so
the realistic cost is +4.8% reaches. NOTE the demo domain's own density is
0.022%, an order of magnitude BELOW CONUS, so Ohio alone under-samples this and
the synthetic sweep is what answers it.

Wall time, routing phase only, DA active in every arm so only the split varies,
3 repeats:

| density | serial, cpu_pool 1 | clustered, cpu_pool 6 |
|---|---:|---:|
| 0.00% | 27.47 s (spread 2.23) | 16.26 s (spread 0.51) |
| 0.26% | 26.27 s (-4.4%) | 16.41 s (+0.9%) |
| 1.00% | 26.96 s (-1.9%) | 16.73 s (+2.9%) |
| 5.00% | 28.75 s (+4.7%) | 16.67 s (+2.5%) |

Every difference is at or below the repeat spread, and the ordering is
non-monotonic (5% faster than 1% on both paths), which is noise rather than a
cost curve. **The split costs no measurable time at realistic gage density on
either the serial or the clustered parallel path.**

Two caveats: the synthetic gages are placed at random, while real gages sit on
larger channels inside longer reaches, so this likely overstates fragmentation;
and one domain on 8 cores rules out a per-reach cost and gross clustering
imbalance, not behaviour at CONUS on hundreds of cores. The parallel arm ran
1.69x faster than serial on 6 cores, consistent with the existing finding that
scaling there is memory-bound rather than kernel-bound.


## Per-window seeding: RETRACTED, the apparent gain is a median artefact

`should_seed_state` fires only on the FINAL window, so the upstream leg is
DIAGNOSTIC during an analysis: the spread rewrites every timestep of a window
AFTER routing, those timesteps are never read again, and only the last one
becomes state via `new_q0`. The downstream leg is prognostic (it is in the
kernel); the upstream leg is not, except once at the hand-off. That description
is accurate and worth keeping.

Per-window seeding was re-measured (lag off, spread 0, 8 h windows, 12 windows,
Ohio held-out) and initially reported here as a 3.4-point improvement in median
absolute PBIAS. **That claim is withdrawn.** Per-gage:

| gage | seed final | seed every |
|---|---:|---:|
| 03010655 | 12.2% | 7.3% |
| 03028000 | 23.1% | 29.5% |
| 03065000 | 0.4% | 4.1% |
| 03068800 | 10.9% | 8.8% |
| **median** | 11.55% | **8.05%** |
| **mean** | 11.65% | **12.43%** |

The median improves and the MEAN WORSENS. Two gages improve, two degrade, and
both extremes degrade. On four points the median is the mean of the middle two,
so it discards exactly that damage. There is no measured gain.

**There was also no reversal of the earlier result.** The recorded measurement
(-23.8% vs -29.1%) used 24 h windows and SIGNED held-out bias; this one used 8 h
windows and median ABSOLUTE PBIAS. Different window length, different statistic,
different quantity. The earlier result stands unchallenged, and the design
decision it supports is untouched.

**And prognostic cycling is incompatible with the delivered operator.** The
upstream spread reads `dQ_o(t + tau)`, a future innovation. A prognostic cycle
must have the corrected state BEFORE routing the next cycle, but obtaining that
future innovation requires routing it first. Circular. The A/B above avoided
this only by running with the lag off, so it never exercised cycling with the
operator that ships.

Further: with a measured tree lag around 15 h and an 8 h cycle, several upstream
pulses are in flight before the first reaches the source gage, so repeated
seeding can stack the same residual and oscillate as they arrive.

**Verdict: do not build the DA-cycle separation.** The current final-window
seeding stays.

### The one idea worth keeping from this

The upstream leg is diagnostic because it runs AFTER routing, not because it
seeds rarely. An alternative that was never tried: apply the PREVIOUS window's
correction to the state BEFORE routing the current window, so the correction is
present while MC routes and propagates downstream by the model's own physics.
That needs no new config field, no change to `max_loop_size` semantics, and no
travel-time estimate at all -- the router supplies the timing, which is the
entire problem the lag machinery exists to solve. It is worth prototyping on its
own merits, separately from anything above.


## The prognostic-upstream question is closed

The upstream leg is DIAGNOSTIC: the spread runs after routing, its timesteps are
never read again, and the only state crossing a window is `q0`, one row taken
from the last routed timestep, corrected once on the final window. Four attempts
were made to change that. All failed, for different reasons, and the reasons are
worth keeping so this is not reopened by accident.

| attempt | outcome |
|---|---|
| seed `q0` every window | no gain once the MEAN is read (median 11.55->8.05 but mean 11.65->12.43, both extremes worse); breaks `max_loop_size` bit-identity, 36.5 m3/s over 2,374 of 11,327 segments |
| separate DA cycle length from the memory window | a cycle boundary must fall on a routing boundary, so the coupling relocates rather than disappears; does not address why the correction fades |
| qlat-space correction with re-route | 20 h OSSE timing error |
| sustained correction via the kernel override | see below |

Note also that "apply the correction before routing the next window" and "seed
`q0` after this one" are the SAME EDIT: `q0` is the only state that crosses a
window boundary. They are not alternatives.

The sustained-override variant failed on mechanism, not just on defensibility.
`simple_da` sets `replacement_val = target_val`, an absolute clamp rather than an
increment, and the DA lands at reach boundaries. Overriding every tree segment
therefore gives either repeated ERASURE (each downstream clamp discards what MC
carried from above) or repeated ACCUMULATION (each segment adds another derived
increment), never one area-scaled perturbation transported through the tree. It
would additionally write synthetic targets into `lastobs`, which is serialised
into restart state and drives the decay path for real observations, so the two
become indistinguishable after a restart. And the feedback has no general
convergence guarantee: with `e_k = r_k - K e_{k-1}`, `0<K<1` converges to
`r/(1+K)` rather than zero, `K=1` gives a two-cycle, and `K>1` diverges.

It would also NOT have removed the travel-time problem, as was claimed when it
was proposed. Applying the previous window's innovation as a sustained target is
`dQ_s(t) = a_s * dQ_o(t - one window)`, which replaces travel-time estimation
with an unvalidated persistence forecast and changes the estimand from spatial
distribution to temporal residual forecasting.

**Conclusion.** The diagnostic upstream leg is the measured best: analysis
held-out median absolute PBIAS 29.6% -> 11.5%, forecast day 1 45.7% -> 32.3%.
The source proposal's state-adjustment motivation is satisfied by the DOWNSTREAM
leg, which is genuinely prognostic inside the kernel, plus one upstream
injection at the analysis-to-forecast hand-off. That should be stated plainly in
the delivery rather than implying continuous upstream state adjustment.


## The backward ck trace (lag_source: trace)

Proposed by the project owner: from the gage at time t, step BACK one timestep
at a time, accumulating the distance the wave covered in each past step until it
crosses the reach, then continue up the parent chain. Two design choices make it
work where six previous celerity estimators failed:

- **Backward through stored history.** Every value read is from a time already
  routed, so the estimator is well posed. The forward equivalent needs celerity
  at `t + tau`, which is why an earlier attempt froze the field and read ambient
  celerity where an event was passing.
- **It traces the WAVE, not the water.** By the Kleitz-Seddon law a discharge
  perturbation moves at `c = dQ/dA = beta*V`, which is `(5/3)V` for a wide
  Manning channel. Tracing the water velocity would over-estimate travel time by
  that factor. `ck` is not assumed: the kernel computes it for the compound
  trapezoidal section (wetted-perimeter correction, overbank blend) and exports
  it. Since `cn = ck*dt/dx`, the trace accumulates `cn` to 1.0 and needs no reach
  length at all.

Measured on the OSSE known-truth pulse:

| arm | median abs timing error | drift |
|---|---:|---:|
| no timing | 6.00 h | +0.314 h/km |
| **ck trace** | **4.00 h** | **+0.154 h/km** |
| cross-correlation of routed hydrographs | 2.00 h | +0.119 h/km |
| measured lag applied backward | 12.00 h | +0.518 h/km |

**First celerity-based estimator to beat no timing.** It also resolves 87.7% of
segments (cross-correlation resolves 9.9%), needs no significance gate, and
CANNOT fabricate a lag from noise, which is the failure that required gating
cross-correlation from 93% spurious to 3%. Unresolvable segments (accumulated
`cn` never reaches 1.0 in the record) return inf and are excluded, not guessed:
2,411 of 19,589 = 12.3%.

### An unexplained 2.4x, and a fix that was proposed and withdrawn

Along the scored chain the traced speed is 1.46 m/s against a routed pulse speed
of 0.81 m/s, so tau is under-estimated by about 2.4x.

This was first attributed to `ck` being the kinematic leading-edge celerity
while MC routes a slower diffusive bulk, and a correction from the cell Reynolds
and Vedernikov numbers was considered. **That diagnosis is wrong and the fix was
withdrawn before implementation.** Standard theory (Ponce; USGS) is explicit
that kinematic and diffusion waves propagate at the SAME celerity and differ in
ATTENUATION. For the Hayami impulse response the centroid arrives at `L/c` and
the peak EARLIER still, so a late arrival has the sign wrong. A correction built
on the cell Reynolds number would also be unphysical, since `D` contains `dx`
and would change under reach subdivision; and t-route's kernel bounds `X` and
carries no Vedernikov term, so textbook diffusivity cannot be recovered from it.

The most likely remaining candidate, not verified: the trace is exact for the
ANALYTICAL kinematic wave, but the discrete MC recurrence propagates through its
`C1/C2/C3` weights at an effective numerical celerity that is not exactly `ck`.
That would be a property of the discretisation, not of the physics, and not
fixable by a celerity correction.

**The 2.4x is left unexplained rather than corrected by a fitted factor.** An
estimator validated against one pulse at hourly output resolution cannot be
tuned at this precision without fitting to it.

Shipped shape: `lag_source: trace` available alongside `hydrograph`, default
`lag_direction: none`.

## 2026-08-17, the 2.4x is EXPLAINED: the solver clamps K at one timestep

The "effective numerical celerity" candidate above is confirmed, in a sharper
form than proposed: it is not a property of the C1/C2/C3 dispersion, it is an
explicit clamp. `MCsingleSegStime_f2py_NOLOOP.f90` line 321 sets

    Km = max(dt, dx/Ck)

and the reach lag of the discrete Muskingum scheme IS K (the first moment of
its transfer function, independent of X). So the routed perturbation crosses at
most ONE segment per timestep no matter what the physical celerity says, while
the `courant()` export the trace integrates is the UNCLAMPED `cn = ck*dt/dx`.
Wherever `cn > 1` (dx < ck*dt; at the domain's ~316 m median dx and 300 s dt
that is any celerity above ~1.05 m/s, i.e. most event flow) the trace
under-estimated tau by exactly the factor cn.

Verified on a uniform synthetic chain (80-267 segments, trapezoid, steady
baseflow, paired control/impulse runs, predictions stated before each run):

| dx (m) | Cn | exported ck (m/s) | routed centroid speed (m/s) | ck/routed |
|---|---|---|---|---|
| 2000 | 0.19 | 1.234 | 1.238 | 0.997 |
| 1000 | 0.37 | 1.234 | 1.243 | 0.992 |
| 500 | 0.74 | 1.234 | 1.247 | 0.989 |
| 300 | 1.23 | 1.234 | **1.000 = dx/dt** | 1.234 |
| 150 | 2.47 | 1.234 | **0.500 = dx/dt** | 2.468 |

Below Cn = 1 the routed speed matches the exported ck within 1%, and the
exported ck matches the analytical trapezoid celerity to 3 decimals, so the
export and the trace arithmetic are both correct there. Above Cn = 1 the routed
speed pins at EXACTLY dx/dt, amplitude-independent (+20% and +100% impulses
identical), which is the signature of a structural cap, not a continuous
distortion. The prediction "within 20% at Cn = 2.5" MISSED; the miss is what
identified the clamp.

**Fix, same day:** `_tree_tau_trace` now accumulates `min(cn, 1)` per sample,
which reproduces `max(dt, dx/ck)` per hop exactly. The trace follows the MODEL,
not the physics; the docstring says so and a regression class
(`TestTraceFollowsTheSolverNotThePhysics`) pins cn > 1, cn < 1, and the
NaN-sample edge (`min(nan, 1.0)` is 1.0 in Python, so the finite guard must run
first). Re-verified against the routed chain: obs/trace 0.963 at Cn = 2.47
(was 2.47 under-estimated), 1.000 at Cn = 1.23, 0.990 unchanged below the cap.
The 3.7% residual at high Cn is same-timestep leakage through the in-reach
`quc` chaining (weight C2 per hop), second order.

Consequences:

- Every traced tau gets LONGER or stays the same; nothing gets shorter.
- The OSSE trace row above (4.00 h error at "+0.154 h/km") predates the clamp
  and is STALE for the shipped arm; the OSSE and the held-out sweeps must be
  re-run with the clamped trace (they were owed for the one-flag collapse
  anyway, DELIVERY_HANDOFF 3.3).
- "Do not fit a factor" held: the 2.4x needed no factor, it needed the model's
  own K clamp mirrored. Nothing here is tuned.

## 2026-08-17, OSSE re-measured with the clamped trace

Same prep, same truth run (the clamp lives in the trace, not the router), same
scorer; only `osse_trace` re-run, plus `osse_none` as a control of the synced
tree. Configs migrated to the one-flag schema (`travel_time_lag` replacing
`lag_direction`/`lag_source`).

Predictions stated before the run and their outcomes:

| prediction | outcome |
|---|---|
| median abs error <= 2.00 h | HIT: 2.00 h (was 3.00; no timing 6.00) |
| mean toward <= 4 h | HIT: 3.26 h (was 5.74; no timing 7.21) |
| drift flattens toward ~+0.1 h/km | HIT: +0.140 (was +0.328) |
| signed bias ~ 0 | MISS: errors remain ONE-SIDED positive; tau is still slightly under-estimated everywhere, +0.140 h/km of residual drift |
| unresolved rises noticeably | MISS: 15,919 -> 16,151, +1.5% only. Ambient cn median is 0.130 m/s, far below the clamp, so the clamp binds per SAMPLE only on fast segment-steps -- which is exactly the scored main stem, which is why the chain errors halved while the domain-wide resolved set barely moved (resolved tau median 15.7 -> 16.2 h) |
| scorable rows may drop 1-2 | MISS: 19 of 19 retained |

Far half of the chain (beyond 26.3 km): 3.00 h against 12.00 h for no timing.
The 45.1 km row that carried a 53 h error now carries 27 h; it is still the
dominant residual and still unexplained. The remaining one-sided error is NOT
the clamp (that is now mirrored exactly); candidates are the span's ambient
anchor where ambient cn < 1 (3.2.2) and the hourly scoring grid.

`doc/scaling_da/figures/{osse_timing.csv,osse_timing.png,osse_meta.json}` are
updated to this measurement; the pre-clamp table is preserved at
`pi10-subcase4/results/osse_timing_preclamp_backup.csv`.

## 2026-08-17, held-out and forecast re-measured with the clamped trace

Owed since the one-flag collapse (every published number described a
configuration that no longer exists) and doubly since the clamp. Arms stated
explicitly per the measurement discipline; predictions were recorded before
each run.

### Held-out analysis skill (spread_sweep, CLI, 4 withheld gages)

| arm | median abs PBIAS | mean | median NSE | mean NSE |
|---|---:|---:|---:|---:|
| noda (baseline, loaded) | 29.6% | 30.2% | -0.32 | -3.09 |
| s_off = DA, `travel_time_lag: false`, RE-RUN | 11.5% | 11.7% | -0.75 | -0.73 |
| s_tr = DA + clamped lag, 48 h span, RE-RUN | 13.1% | 13.3% | -0.26 | -0.87 |

- **s_off reproduced its pre-clamp scores to the last decimal** (max per-gage
  delta 0.0): the off-path is untouched by the clamp and the synced workspace
  is clean. Prediction HIT.
- **s_tr: prediction MISS.** Predicted the clamp would move the lag arm toward
  the no-lag score (11.5-12.5); it stayed at 13.1 median. The flows DID change
  (per-gage deltas up to 0.28 NSE / 0.24 PBIAS points) but the effect is below
  what four gages can resolve. The clamp fixes the TIMING of the correction
  (OSSE); held-out PBIAS is a volume metric and does not see it.
- `s_hyd` cannot be re-run: the hydrograph estimator is deleted code. Its row
  is retired with the collapse.

### Forecast mode (BMI, 24 h analysis then 72 h free run, 4 withheld gages)

| arm | day 1 | day 2 | day 3 | day-1 NSE |
|---|---:|---:|---:|---:|
| noda, RE-RUN | 45.7% | 25.4% | 30.7% | -12.63 |
| nudging, RE-RUN | 41.2% | 22.6% | 31.0% | -8.75 |
| scaling + lag, 12 h span (max that fits), RE-RUN | **45.1%** | 25.4% | 30.7% | -11.95 |
| scaling, `travel_time_lag: false`, RE-RUN | **32.3%** | 24.5% | 30.7% | -4.79 |

- noda, nudging, and the no-lag scaling arm all **reproduced their recorded
  numbers exactly** (45.7 / 41.2 / 32.3). Predictions HIT.
- **The shipped default cannot run this workflow at all.** With
  `travel_time_lag: true` the DA requires update windows of `lag_window_h +
  innovation_spread_h`; a 24 h ngen-style analysis cannot host 48+12 nor the
  demo template's 24+12. The fail-closed raise (DELIVERY_HANDOFF 3.1.3) fired
  exactly as designed, first try.
- **The largest span that fits (12 h) guts the forecast benefit: 45.1% day 1,
  statistically noda.** At a 12 h span, 17,203 of 19,589 tree segments (87.8%)
  are unresolved and therefore dropped; the corrected footprint shrinks to a
  median 9.9 km around the gages, and days 2-3 are IDENTICAL to noda.
  Prediction partly right (predicted the lag would not rescue the forecast,
  day 1 >= 34) but the mechanism was wrong: predicted a launch-edge decay cost
  of a few points, got a span-forced coverage collapse.
- The old 34.9% "forward lag" forecast number was produced before the span
  enforcement existed, i.e. by a silently shortened trace; it is not
  reproducible under current code and should not be quoted.

### What this says, together with the OSSE

The clamped trace is now MEASURABLY RIGHT about timing (OSSE median error
2.00 h vs 6.00 h) and measurably neutral-to-negative everywhere else: held-out
analysis skill is unchanged within resolution, and in the BMI forecast workflow
the default-on lag either refuses to run (span > update) or erases the DA's
forecast benefit (45.1% vs 32.3%). The forecast benefit that headlines the
delivery exists in the `travel_time_lag: false` arm. The 2026-08-17 default-on
decision was taken before these numbers existed and should be revisited with
them: this document's earlier recommendation logic (the lag helps the analysis
record and hurts the forecast) now holds with sharper numbers and one new fact,
that the shipped default and the ngen update cadence are structurally
incompatible (3.1.3 is no longer an edge case, it is the first thing an ngen
integrator hits).

Raw tables: `pi10-subcase4/results/spread_sweep.csv` and `forecast_mode.csv`
(arms labeled `scaling_lag12` / `scaling_nolag`), pre-clamp copies alongside as
`*_preclamp_backup.csv`. The forecast harness gained `--template` so sweep
templates live in configs_extra without touching the fingerprinted demo config.

## 2026-08-17 (late), full re-measurement under final code, and the default DECIDED

The owner challenged the record ("the lag was shown to improve results") and
directed a rerun of every benchmark with a decision to follow. The
reconciliation first: the improvement in the record is the ANALYSIS-TIMING
result (the OSSE), and it is real and current; the held-out and forecast
records never showed a lag benefit after the significance gate. Nothing was
off in the code: every control below reproduced its recorded number exactly.

Reruns under final code (clamped trace, spread default 0, all fail-closed
guards):

| measurement | result |
|---|---|
| OSSE trace arm | BIT-IDENTICAL to the morning rerun: median 2.00 h vs 6.00 h untimed |
| held-out sweep s_off / s_tr | BIT-IDENTICAL: 11.5% / 13.1% |
| forecast controls | noda 45.7, nudging 41.2, no-lag 32.3, all exact |

New forecast arms, runnable for the first time (the spread default change
freed the window budget):

| cadence | noda | nudging | scaling no-lag | scaling + lag |
|---|---:|---:|---:|---:|
| 24 h analysis, 72 h free (day 1) | 45.7 | 41.2 | **32.3** | 44.8 (24 h span) |
| 48 h analysis, 48 h free (day 1) | 25.38 | 24.67 | **23.92** | 25.20 (48 h span, FULL tau coverage) |

At 48 h the lag arm's day 2 is IDENTICAL to noda (30.71), and day-1 NSE is
-4.10 against -3.20 for no-lag. Coverage was never the binding constraint:
at the full span the taus resolve (median 22.1 h) and the forecast still
carries nothing, because the final analysis timestep -- the one that seeds
the forecast -- reads dQ_o(t + tau) past the analysis edge, where it decays
to nothing for any material tau. Demonstrated now at 12, 24, and 48 h spans.

**Decision, per the pre-registered rule** (off if >2 points worse than no-lag
at any runnable cadence): **`travel_time_lag` defaults FALSE.** 24 h cadence:
12.5 points worse. 48 h cadence: 1.28 points worse (within resolution, and
still the wrong sign). The lag stays as the opt-in analysis-timing tool; the
untimed arm is the delivery.

Two instrument notes from this batch, both measured before use:

- The BMI memory estimator's 200 bytes-per-link-timestep factor blocked the
  48 h arm on a machine that routes larger windows routinely. Measured ground
  truth: 7.86 GB whole-process peak RSS for a 96 h single-window run over the
  same 103,559 links (70.7 B per link-timestep including fixed process
  overhead) against an 11.1 GB claim for half that window. The factor is now
  100 (1.4x the measured whole-process ratio). This mattered only because
  this branch made an undersized window a hard error where it used to shrink
  silently.
- The forecast harness now filters its feed to the network's `vfp_divs`
  (results identical; the 24 h no-lag control reproduced 32.27 under the
  patched harness).

## 2026-08-17 (final), the untimed hand-off: the lag's forecast cost is now ZERO

The owner read the launch-edge mechanism against the operational picture (the
NWM cycle's lookback IS the analysis phase) and asked for the hybrid: keep the
traced timing in the analysis record, seed the forecast state untimed. Built as
`apply_in_kernel(seed_untimed=True)`, passed by both drivers on exactly the
window whose `new_q0` a forecast inherits: the hand-off instant is rewritten
with the untimed spread over the full in-reach tree (no resolved-tau
requirement), every earlier timestep keeps the traced timing. Unit test pins a
ramp innovation discriminating the two reads.

Prediction registered before the rerun: the seeded state becomes the untimed
arm's state, so the lag-on forecast should reproduce the untimed forecast
EXACTLY. Outcome: HIT, bit-for-bit -- max per-gage delta 0.0 in PBIAS and NSE
at both cadences:

| cadence | lag + untimed seed | untimed arm |
|---|---:|---:|
| 24 h analysis, day 1 / 2 / 3 | 32.27 / 24.51 / 30.71 | 32.27 / 24.51 / 30.71 |
| 48 h analysis, day 1 / 2 | 23.92 / 30.70 | 23.92 / 30.70 |

Standing state of the flag after this:

- The lag improves the analysis-timing record (OSSE 2.00 h vs 6.00 h) and now
  costs the forecast NOTHING, by construction.
- `travel_time_lag` stays FALSE by default for runnability and cost, not
  skill: the span must fit the opening window (fail-closed), operational
  lookbacks of 3-28 h cannot host the 48 h default span, and the lag adds
  about 5% runtime. Where the cadence hosts the span, enabling it is now
  cost-free to the forecast.

## 2026-08-17 (addendum), the seed experiments re-run non-vacuously, and rho-tau rejected

**A correction first.** The untimed-hand-off validation in the previous section
was VACUOUS as a benchmark: the forecast templates relied on the
`travel_time_lag` DEFAULT, which this same session had just flipped to false,
so both "arms" ran untimed and the reported bit-identity (max delta exactly
0.0) compared an arm with itself. Caught by a positive engagement check
(grepping the run logs for the trace's and the seed's own log lines: zero).
The unit test of the mechanism was always real; the benchmark claim was not.
Rule adopted: experiment arms pin their switches EXPLICITLY, never through
defaults, and no score is read before the treated arm's log shows the
treatment's signature.

Re-run with `travel_time_lag: True` pinned in the templates, engagement
verified in every log (trace line, seed line, and rho lines only in the rho
arms):

| arm, day 1 | 24 h cadence | 48 h cadence |
|---|---:|---:|
| untimed arm (control) | 32.27 | 23.92 |
| lag + untimed seed | 32.27 | 23.92 |
| lag + rho^tau seed | 41.32 | 24.69 |

- **Untimed seed, real this time:** identical to the untimed arm within
  float32 numerical noise (max per-gage delta 5e-5 PBIAS points, 1e-5 NSE).
  The conclusion of the previous section stands, now on a non-vacuous
  measurement: the lag's forecast cost is zero with the untimed hand-off.
- **rho^tau-discounted seed: REJECTED, as pre-registered.** The innovation's
  measured 1 h autocorrelation (median 0.809, range 0.27-0.99 over 23 gages)
  compounds to a median 72% of seed mass retained (18.8% minimum), and the
  day-1 skill collapses toward noda at the 24 h cadence (41.32 against 32.27).
  The far-tree seed it discounts is exactly the mass that delivers -- and its
  persistence-seeded corrections VERIFY downstream, so bias-dominated
  innovations persist far longer than an AR(1) extrapolation of their 1 h
  autocorrelation admits. The experimental env-gated path is removed; this
  entry is its record.

## 2026-08-17 (context), the operational cycle re-read, verified

Checked against NOAA/OWP documentation (water.noaa.gov/about/nwm; NOAA NWM on
the AWS Open Data registry): the AnA trunk cycles HOURLY with a 3 h lookback
(28 h Extended once daily feeding 19Z), restarts chained cycle to cycle;
Short-Range (hourly, 18 h) and Medium-Range (4x daily, ~10 d, 6 members)
forecasts initialize from the trunk's restart and do not cycle on their own
states.

Three consequences for this record:

1. The lag is structurally OUTSIDE the operational trunk: a 3 h window cannot
   host any useful trace span (48 h default fails closed; even 28 h hosts only
   a reduced span once daily). Default-off is topology, not preference.
2. "Timing by refresh": hourly re-branching re-estimates the upstream
   correction field every hour, superseding within-run tau-scheduling for any
   consumed forecast. The only unrefreshed regime is the medium-range tail of
   one branch, which is exactly what the seed experiments measured
   (persistence verifies; rho^tau discounting hurts).
3. The single-branch forecast harness is CONSERVATIVE at short leads relative
   to operations, so the measured scaling-vs-nudging advantage is a floor.

## 2026-08-17 (trunk-cadence check), the scaling arm is cadence-invariant

Robustness check against the operational topology: the 24 h analysis re-run as
EIGHT chained 3 h BMI updates (the trunk's cadence; template max_loop_size 3,
lag off, spread 0) against the single-update baseline.

| arm, day 1/2/3 | single update | 3 h chained updates |
|---|---|---|
| noda | 45.68 / 25.38 / 30.71 | identical |
| scaling | 32.27 / 24.51 / 30.71 | **identical** |
| nudging | 41.21 / 22.57 / 30.97 | 22.84 / 44.11 / 28.99 |

The scaling arm is EXACTLY cadence-invariant, as predicted: the cycling
background is never seeded, so the analysis partition cannot enter the state,
and the branch seed is the freshest innovation under any chaining. No harness
changes are required for the conclusions in this document.

Nudging is strongly cadence-sensitive (a fresher point-nudge in the launch
state buys day 1, then the uncorrected upstream mass reasserts and day 2 falls
below noda). CAVEAT before quoting those numbers: nudging under chained BMI
updates exercises lastobs/observation plumbing this harness has not separately
verified; treat as provisional. The scaling-vs-nudging comparison that is safe
to quote remains the single-update one, plus the observation that scaling's
lead-time profile is FLAT where nudging's swings, which is itself the
mass-field-vs-point-correction mechanism showing through.
