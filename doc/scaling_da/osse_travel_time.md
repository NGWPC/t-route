> **SUPERSEDED 2026-08-15, AND ITS EXPERIMENT WAS CIRCULAR.**
>
> The prep used here chose the injection site at whatever distance made
> `tau = 16 h` AT 0.8 m/s, and then scored 0.8 m/s and found it excellent
> (1.00 h). The constant was graded on a site defined by itself, so **every
> celerity number in this document is untrustworthy**, including the comparison
> table other documents once quoted from it.
>
> The prep now picks the injection site by network DISTANCE, with no celerity
> involved. On that honest layout the current numbers are: no timing 6.00 h,
> ck backward trace 4.00 h, cross-correlation 2.00 h, measured lag applied
> backward 12.00 h.
>
> The constant-celerity implementation this document describes (0.8 m/s, 48 h
> horizon) no longer exists. See `TIMING_DECISION.md` for the current record and
> `DELIVERY_HANDOFF.md` for where the work continues.
>
> Kept only as the historical record of the first OSSE design.

# OSSE: does the travel-time lag place the correction at the right time?

Run 2026-08-13 on the Ohio harness (`~/repos/github/pi10-subcase4`), on the
always-on lag implementation (constant celerity 0.8 m/s, 48 h horizon,
deferred-window halo). This is the experiment every review round asked for: a
known disturbance with a known origin time, so timing skill is measured against
truth instead of against four held-out gages.

## Design

Three arms, identical except where stated:

| arm | lateral inflow | DA |
|---|---|---|
| `osse_truth` | +50 cms for 3 h at one upstream flowpath, from 2019-05-30 00:00 | none |
| `osse_assim` | unpulsed | scaling DA, observations = the truth run's output at every gage (`synthetic_obs_baseline`, factor 1.0) |
| `noda` | unpulsed | none (common reference) |

The injection site sits on gage 03031500's tree at a static travel time of
exactly 16.0 h (the deepest tree; 145-link chain, 17 flowpaths). Per chain
flowpath: `anomaly = truth - noda` is the routed pulse (ground truth of what
happened, where, when); `increment = assim - noda` is what the DA applied.
An un-lagged scheme applies the gage innovation at the gage's own time, so its
timing error at any segment equals the routing time from that segment to the
gage; that reference costs no second run.

Predictions stated before the runs: the lag's increment peaks near the truth
anomaly's peak (within ~2 h, hourly output); a systematic negative drift with
distance if the routed celerity exceeds the 0.8 m/s constant; the no-lag error
grows to ~16 h at the injection site.

## Result

The pulse reached the gage in 16 h (anomaly peak 54.8 cms at 2019-05-30 16:00).
16 of 17 chain flowpaths carried a scorable signal:

| metric | lag (as built) | no-lag reference |
|---|---:|---:|
| median &#124;peak-timing error&#124; | **1.0 h** | 4.5 h |
| far half of the chain (tau > 7 h) | **1.5 h** | 10.5 h |

The applied increment tracks the routed pulse's true timing along the whole
chain (`figures/osse_timing.png`, `figures/osse_timing.csv`); the no-lag error
grows linearly with distance, reaching 12 h at the farthest scored flowpath,
exactly the temporal inconsistency the lag was specified to remove.

**Celerity calibration evidence:** the lag error drifts -0.156 h per hour of
static tau (the increment lands slightly EARLY upstream), implying an effective
routed celerity of ~0.95 m/s on this chain and event -- the 0.8 m/s constant is
close and slightly low. The residual error is bounded by the hourly output
quantization plus this drift.

Amplitudes: the applied increment is the area-scaled fraction of the innovation
(`(A_s/A_o)^0.77` plus confluence splits), 5-19 cms against a 52-55 cms truth
anomaly. That is the method's design (proportional correction), not a timing
finding.

## Round 2: celerity from the kernel's Courant number (NEGATIVE)

The obvious next step was to stop guessing celerity and read the router's own:
`cn = ck*dt/dx` is dimensionless, so `1/cn` is a reach's transit time in
timesteps directly, with no `dx` and no unit conversion. Implemented as
`celerity_source: courant` (frozen-field form: `tau[j,t] = tau[parent,t] +
1/cn[parent,t]`, reading `cn` only at the instant `t` so tau stays invariant to
`max_loop_size` without a second halo).

Predicted before running: the −0.156 h/h drift is the constant-celerity bias, so
reading the router's own celerity should drive it to ~0 and leave the median at
its ~1 h quantization floor.

Measured, same OSSE:

| metric | constant 0.8 m/s | Courant-derived |
|---|---:|---:|
| median &#124;peak-timing error&#124; | **1.0 h** | 1.5 h |
| far half of the chain | **1.5 h** | 3.5 h |
| drift vs static tau | −0.156 h/h | **+0.428 h/h** (sign flipped) |
| `max_loop_size` invariance | 0.0 | 0.0 (held) |

**Worse, and the diagnostic says why.** A run-time log of the implied celerity
(`dx/(transit*dt)`) across tree reaches reports **median 0.149 m/s**, p10 0.075,
p90 0.445 -- consistent with this model's routed velocities (median v ~0.09 m/s
in headwater reaches, `ck ~ (5/3)v`). That is the AMBIENT celerity. The wave
carrying the correction travels at the EVENT celerity, ~0.95 m/s here. Frozen
field evaluates `cn` at the instant the correction is applied, when the reach is
still at baseflow, so tau came out ~5x too long, pushed 49% of segment-timesteps
past the 48 h horizon, and left the surviving increments peaking at arbitrary
times.

The working assumption behind the frozen-field choice -- that the time-variation
is a second-order term -- is false on this network: celerity varies by an order
of magnitude between baseflow and event, so the variation IS the dominant term.
Making the Courant path competitive requires the exact characteristic trace
(evaluate each hop at the time the water actually reaches it), which reads
forward past the window end and therefore needs the halo to carry `cn` as well
as the innovation.

`celerity_source` therefore defaults to `constant`. The Courant path is kept,
tested, and selectable so the trace upgrade has its scaffolding, but it is not
recommended and is not memory-validated at CONUS scale.

## Caveats

One chain, one gage, one synthetic event, hourly output. The pulse is additive
on a quiet baseline (0.1 cms at the injection flowpath), so the signal is far
cleaner than an operational innovation. The drift estimate conflates celerity
error with the pulse's 3 h centroid. None of this changes the qualitative
conclusion -- the mechanism shifts the correction to the right time, and the
no-lag alternative is wrong by the routing time -- but the -0.156 h/h drift is
a single-event estimate, not a calibration.

## Reproduce

```
cd ~/repos/github/pi10-subcase4
pixi run python scripts/osse_prep.py     # picks site, writes pulsed forcing + configs
pixi run python -m nwm_routing -V5 -f configs/osse_truth.yaml
pixi run python -m nwm_routing -V5 -f configs/osse_assim.yaml
pixi run python -m nwm_routing -V5 -f configs/noda.yaml
pixi run python scripts/osse_timing.py   # table, summary, figure
```
