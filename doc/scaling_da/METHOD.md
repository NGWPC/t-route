# Simple-scaling streamflow data assimilation in t-route: method

Self-contained method description of the implementation on
`feat/streamflow-scaling-da`, written as a journal-style method section.
Scheme after Ogden and Clark (OWP/CIROH R2O Proposal 5, 2025), with the
implementation choices that proposal leaves open resolved as documented here.
Applies to Muskingum-Cunge (MC) routing on NextGen hydrofabric (NHF) networks,
through both the CLI driver and the BMI interface.

## 1. Setting and overview

t-route routes discharge over a vector river network with MC, advancing in
timesteps of length $\Delta t$ (300 s operationally) over forcing windows of
$W$ timesteps. At a set of USGS gages with hourly discharge observations, the
method (i) forms the innovation, the difference between observed and modeled
discharge at the gage, (ii) inserts it at the gage segment inside the routing
solve, so it propagates downstream through the ordinary routing operators, and
(iii) distributes it upstream over the gage's contributing tree by drainage
area scaling, split at confluences by relative flow contribution, optionally
shifted by a traced travel time. The upstream leg runs after each window is
routed and rewrites that window's output (a diagnostic update); its final
timestep is handed to the next phase as the channel warm state, which is the
prognostic entry point (Section 8).

## 2. Innovation at the gage

For gage $o$ with observed discharge $Q_{\mathrm{obs}}$ and modeled discharge
$Q$ at the gage segment,

$$\Delta Q_o(t) = Q_{\mathrm{obs}}(t) - Q(t). \tag{1}$$

Observations are read from USGS TimeSlice files, resampled to whole minutes,
and interpolated onto the model time grid from the surrounding reports
(hourly gages fill every $\Delta t$ step; interpolation is bidirectional at
record edges). The observation matrix is aligned so column $j$ sits at
exactly $t_0 + j\,\Delta t$, with column 0 reserved for the initial
condition; every producer passes through one alignment function so no
consumer can read an observation at the wrong step. Inside the kernel the
innovation is inserted at the gage segment each timestep; where observations
go stale the insertion decays exponentially with the same decay coefficient
the operational nudging scheme uses. Model background at the gage is taken as
the analyzed value minus the already applied nudge, so the upstream
distribution never double counts its own correction.

## 3. Gage trees and stop rules

Each assimilated gage is the root of a tree built once, at network
construction, by breadth-first search along the upstream adjacency:

- The walk stops at every other assimilation SOURCE gage and at every
  waterbody segment (proposal Edge Case 1): corrections never cross another
  gage's domain or enter a lake or reservoir. Non-source gages (for example
  gages withheld for evaluation) do not stop the walk.
- The root segment itself is never a stop. A gage whose crosswalked segment
  is itself a lake id is excluded from assimilation entirely, loudly, because
  the kernel routes lake segments as reservoir objects and reservoir DA owns
  them; a gage on the channel below a reservoir assimilates normally, with
  its tree stopping at the waterbody upstream.
- Branches severed by a stop are recorded per junction (compressed sparse
  rows of pruned-branch flow positions), so the confluence partition can
  charge the stopped branch its share of flow (Section 5) instead of
  inflating the surviving siblings.

Trees carry, per segment in BFS order with the gage at index 0: the routed
segment id, the parent index, the contributing drainage area $A_s$ (km$^2$),
a junction flag, and one scaling exponent $\theta$ for the whole tree. The
tree set is static for the run; anything that decides how reaches are split
for computation (Section 8) derives from this static set, never from which
gages happen to report in a given window.

## 4. Upstream distribution: area scaling

Along a linear (junction-free) chain the correction at a segment $s$ with
contributing area $A_s$ under a gage with area $A_o$ is the proposal's
simple-scaling relationship,

$$\Delta Q(s, t) = \Delta Q_o(t + \tau_s)\left(\frac{A_s}{A_o}\right)^{\theta}, \tag{2}$$

applied with the traced travel time $\tau_s$ of Section 6 ($\tau_s = 0$ when
the lag is disabled). The per-step recursion telescopes to this closed form
only for a constant exponent, so $\theta$ is one value per tree: the finest
regionalization the method admits. $\theta$ resolves per gage in the order
per-gage CSV (ids matched as strings, preserving leading zeros), then the
gage's hydrofabric VPU, then a global default of 0.77 (Ogden and Dawdy 2003,
fitted at 0.3 to 21.2 km$^2$ in a semi-humid watershed; the configuration
surface exists precisely because one constant is not defensible at
continental scale). All three levels validate finite, positive values.

## 5. Confluence partition

Where channels $A$ and $B$ merge to form $C$, the proposal assigns
$\Delta Q_A = \Delta Q_C\,(Q_A / Q_C)$. Routed flows do not satisfy
$Q_A + Q_B = Q_C$ at a timestep (routing lag; measured on a spun-up flood,
the branch sum exceeds the downstream flow at 38% of active
confluence-timesteps), and with the literal denominator the branch fractions
then sum above one and the split manufactures water. The implementation
therefore partitions by

$$f_j = \frac{Q_j}{\max\!\left(Q_p,\ \sum_k Q_k\right)}, \qquad f_j \in [0, 1], \tag{3}$$

where $Q_p$ is the modeled flow at the parent (downstream) segment and the
sum runs over ALL branches at the junction, including branches the stop rule
pruned from the tree. Each fraction and their sum are bounded by one, so the
split is non-expansive by construction, reduces to the proposal's rule
whenever flows balance, and the pruned branch's share simply leaves the
correction (it is not renormalized onto survivors). Flows below a floor
(`min_flow_cms`, default $10^{-6}$) contribute no share, so a dry parent
cannot produce a division blowup. The modeled flows in (3) are read from the
routed window at the same timestep the correction applies to, with the gage
root's background taken as analyzed minus nudge (Section 2).

## 6. Travel-time lag: a backward Lagrangian trace of the model's wave

An upstream segment's present discharge anomaly reaches the gage $\tau$
later, so the innovation that corrects segment $s$ at time $t$ is the one
the gage reports at $t + \tau_s$; equation (2) uses exactly that value.
Applying $\Delta Q_o(t - \tau)$ instead is $2\tau$ late by construction and
measurably worse than no timing at all.

$\tau$ is measured, not assumed, by following the kinematic wave (not the
water: by the Kleitz-Seddon law a discharge perturbation travels at
$c_k = \partial Q/\partial A$, which is $\tfrac{5}{3}V$ for a hydraulically
wide Manning channel) backward from the gage through the model's own routed
celerity field. The MC kernel computes $c_k$ for the compound trapezoidal
section, including the wetted-perimeter correction and overbank blend, and
exports the Courant number $C_n = c_k\,\Delta t/\Delta x$ per segment and
timestep. Since $C_n\,\Delta x$ is the distance the wave covers in one
timestep, the trace works in units of reach length and needs no $\Delta x$:
walking up the tree, each segment crosses its PARENT's reach $r$ by stepping
back through time until

$$\sum_{k} \min\!\left(C_n(r,\, t - k),\ 1\right) \ \geq\ 1, \tag{4}$$

with linear interpolation inside the final step, and

$$\tau_s = \tau_{\mathrm{parent}} + (\mathrm{steps} - \mathrm{frac}). \tag{5}$$

Because $t - \tau_{\mathrm{parent}}$ recedes as the walk climbs, every
celerity read is at a time already routed and held in memory: the backward
trace is well posed where a forward trace would need celerity at times not
yet computed. All children of a junction cross the same parent reach, so
siblings share $\tau$ exactly and the confluence partition (3) stays
synchronized with both branch corrections.

The $\min(C_n, 1)$ clamp in (4) makes the trace follow the MODEL rather than
the physics: the MC solver floors its storage constant at one timestep,
$K = \max(\Delta t,\ \Delta x/c_k)$, and the reach lag of the discrete
Muskingum scheme is $K$ (the first moment of its transfer function,
independent of the weighting factor $X$), so the routed perturbation crosses
at most one segment per timestep regardless of the physical celerity. The
exported $C_n$ is the unclamped physical value; tracing it raw
under-estimates $\tau$ by exactly the factor $C_n$ wherever $C_n > 1$, which
at NHF reach lengths (median near 316 m) and $\Delta t = 300$ s is any
celerity above about 1 m/s, that is, most event flow. Verified on a uniform
synthetic chain: the routed centroid speed pins at exactly
$\Delta x/\Delta t$ once $C_n > 1$, amplitude independent, and matches the
exported celerity within 1% below the cap.

The trace runs ONCE per run over a fixed span of `lag_window_h` hours
(default 48) sliced from the start of the first routed window, and is cached
per gage and span. Both drivers enlarge the first window to cover the span
plus the spread of Section 7, and a final remainder window shorter than the
span is folded into the window before it, so the measured span is the same
data under every memory partition (`max_loop_size` is a memory knob and must
not change discharge; verified to $0.0$ over 11,327 segments). Segments the
span cannot speak for return $\tau = \infty$ and are EXCLUDED from the
correction, never given a fallback speed, with the reasons counted
separately: parent unresolved, reach absent from the Courant export (the
diffusive domain, deliberately: the trace cannot follow what the kernel does
not report), no live sample in the record (dry channel or reservoir pool),
record too short (the one class a longer span could resolve), and crossings
that only just fit the record (a lower bound, not a measurement). The
integer shift applied to the innovation series is
$\mathrm{rint}(\tau)$, one non-negative value per segment, which keeps the
compiled application kernel on its batched path. Under the BMI, an update
that supplies fewer forcing timesteps than the span fails closed with an
actionable error rather than silently shortening the measured span.

## 7. Temporal spreading of the innovation

Independently of the lag, the innovation series may be averaged forward over
a window of `innovation_spread_h` hours ($T$) before distribution:

$$\overline{\Delta Q_o}(t) = \frac{1}{T}\int_{t}^{t+T} \Delta Q_o(t')\,\mathrm{d}t'. \tag{6}$$

This is the estimator of $\Delta Q_o(t + \tau)$ when $\tau$ is treated as
unknown on the interval, and it also represents what a point shift cannot:
MC routes a diffusing wave, so one gage increment corresponds to a range of
upstream times. The window is one-sided (never before $t$, which would
correct present water with information about water already past the gage),
conserves the innovation's time integral in the series interior, and pads
the tail with the last value so the final timestep equals its own raw
innovation. The default is $T = 0$ (raw innovation): it claims nothing about
timing, keeps the confluence background exact, and needs no data beyond the
current window. With both mechanisms enabled the smoothed series is read at
$t + \tau$, and the required window length is the sum of the two horizons.

## 8. Application inside the solve, downstream leg, and state

Insertion at the gage is IN the routing solve: gage segments are made
single-segment reaches when the execution plan is built (the plan is
constructed once from the static gage set and reused for every window; the
kernel routes a whole reach in one call, so only a reach-final segment hands
its corrected outflow to the next reach), and the inserted value reaches
downstream segments within the same timestep through the ordinary upstream
inflow gathering. Downstream propagation of the correction is therefore
inherent, in contrast to post-processing schemes that structurally cannot
reach downstream.

The upstream leg is applied after each window is routed, rewriting the
window's discharge output over the tree (equations 2, 3, 6), with the reach
limited to `max_reach_km` (default 200) along the network distance from the
gage; segments beyond the limit, beyond an unresolved trace, or outside
every tree are untouched. Depth is carried with the corrected discharge
through the wide-channel Manning relation $h' = h\,(Q'/Q)^{0.6}$, guarded:
the transform applies only where both discharges exceed the low-flow floor
and the ratio lies within a bounded band, so a near-dry background or an
extreme correction leaves depth untouched rather than extrapolating the
power law outside its validity. This REDUCES the discharge-depth
inconsistency at the forecast hand-off where the ratio is a meaningful depth
signal; it does not eliminate it, and velocity is never modified. The exact
$\Delta Q\!\to\!\Delta y$ transform of the proposal's hydraulic-routing case
is out of scope (MC only). Since only the final timestep of a window
becomes the next window's initial condition, the upstream rewrite is
diagnostic within a window and prognostic across the phase boundary: the
corrected final state (seeded $q_0$) is recorded aside and installed only
when a phase ends (the analysis-to-forecast hand-off), while a continuing
analysis keeps the uncorrected cycling background so the next innovation is
not debited by the correction just applied. With the travel-time lag enabled,
the hand-off instant itself is always seeded with the UNTIMED correction over
the full in-reach tree: the lagged correction at that instant reads
$\Delta Q_o(t+\tau)$ beyond the analysis edge, where it decays to nothing, so
seeding it lagged would hand the forecast an uncorrected state (measured: the
lagged hand-off scored at the no-DA baseline at 12, 24 and 48 h spans, and
the untimed hand-off restores bit-identity with the untimed arm's forecast).
Every earlier timestep of the analysis record keeps the traced timing. Under the BMI, `create_state`
serializes both warm states and the traced travel time together with the
identity it was measured under ($\Delta t$, `lag_window_h`, a fingerprint of
the tree set); `load_state` restores the trace only under a matching
identity and otherwise clears it with a warning, so an uninterrupted run and
a checkpoint/resume run read the same $\tau$ or say loudly why they cannot.

## 9. Configuration

All behavior is under `compute_parameters.data_assimilation_parameters.
streamflow_da.streamflow_scaling_parameters`, mutually exclusive with
nudging: `theta` (`default` / `by_vpu` / `per_tree_csv`), `max_reach_km`,
`innovation_spread_h`, `travel_time_lag` with `lag_window_h`,
`min_flow_cms`, `holdout_sites_file` (evaluation withholding; unknown ids
are a hard error), `synthetic_obs_factor` and `synthetic_obs_baseline`
(observing-system simulation inputs), and `spread_chunk_timesteps` (a
memory cap on the application batch). Every quantitative parameter
validates finite; observation ingest failures raise rather than degrade to
a silent no-assimilation run.

## 10. Operational context

The NWM runs a permanently cycled Analysis and Assimilation (AnA) trunk:
hourly cycles with a 3 h lookback, each restarting from the previous cycle's
state, with a once-daily 28 h Extended AnA feeding the 19Z cycle. Forecasts
branch from the trunk's latest restart (Short-Range hourly to 18 h;
Medium-Range four times daily to about 10 days) and never cycle on their own
states (NOAA/OWP, water.noaa.gov/about/nwm). The scheme's pieces map onto
that topology directly:

- The `load_state` rule is the trunk/branch split: a resuming analysis
  installs the uncorrected cycling background (so the next innovation is not
  debited), a DA-less forecast installs the seeded state. The seeded state is
  the delivery: the forecast's skill is the corrected upstream mass field,
  with persistence of the correction set by travel time rather than a decay
  constant.
- The travel-time lag cannot run on the operational trunk: its span requires
  ``lag_window_h`` of routed history inside one window, and the trunk's
  window is 3 h (28 h once daily). This is a topology constraint, not a
  tuning choice, and is why the lag defaults off; its home is retrospective
  analyses with long windows, where record timing is the product.
- Hourly re-branching performs "timing by refresh": every launch re-estimates
  the whole upstream correction field one hour further along with one hour
  newer observations, superseding within-run tau-scheduling for any consumed
  forecast. A single-branch experiment (Section 8's harness) is therefore
  conservative for short leads relative to operations.
