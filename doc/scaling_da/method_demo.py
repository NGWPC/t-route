"""Worked example of the scaling DA method on a three-segment chain.

Nexus points A -> B -> C -> D with the gage at D, so three segments:

    S1 (A->B, headwater), S2 (B->C), S3 (C->D, the gage segment).

Every number is chosen to be checkable by hand, and the travel-time trace and
the innovation smoothing run through the SHIPPED code (``_tree_tau_trace``,
``_smooth_innovation``), not a reimplementation, so the walkthrough cannot
drift from the implementation. Companion to METHOD.md; run from the repo root:

    pixi run python doc/scaling_da/method_demo.py

Prints a verification table and writes figures/method_demo.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nwm_routing.scaling_da_apply import ScalingDA

from troute.scaling_da import build_gage_trees_from_mappings

# ---------------------------------------------------------------- the setup
DT = 600.0                      # s, so 6 steps per hour
NT = 30                         # record length in steps
THETA = 0.77                    # the Ogden-Dawdy default

# Segment ids and topology: 3 is the gage segment (outlet at D).
RCONN = {3: [2], 2: [1], 1: []}
AREA = {3: 100.0, 2: 50.0, 1: 25.0}      # km2: A_o = 100 at the gage

# Physical celerity per reach, expressed as the exported Courant number
# cn = ck*dt/dx. Reach S3 is slow (cn = 0.5 -> 2 steps to cross); reach S2 is
# FAST (cn = 2.0): physics says half a step, but the MC solver floors its K at
# one timestep, so the model's own wave takes a full step and the trace clamps.
CN = {3: 0.5, 2: 2.0, 1: 0.25}           # S1's own reach is never crossed

# Innovation at the gage: model flat at 20, observed +10 on steps 10..15.
Q_MODEL_GAGE = 20.0
OBS_BUMP, BUMP_LO, BUMP_HI = 10.0, 10, 15
BACKGROUND = {3: 20.0, 2: 10.0, 1: 5.0}  # modeled flow at each segment

SPREAD_H = 0.5                  # the optional forward average, 3 steps at DT

C = {"obs": "#4269d0", "model": "#9498a0", "s1": "#ff725c",
     "s2": "#efb118", "s3": "#4269d0", "clamp": "#6cc5b0"}


def innovation() -> np.ndarray:
    dq = np.zeros(NT)
    dq[BUMP_LO : BUMP_HI + 1] = OBS_BUMP
    return dq


def traced_tau() -> tuple[dict[int, float], dict[int, float]]:
    """(tau by segment from the SHIPPED trace, unclamped 1/cn hand value)."""
    trees = build_gage_trees_from_mappings(RCONN, {"D": 3}, AREA,
                                           theta_default=THETA)
    tree = trees["D"]
    colpos = {3: 0, 2: 1, 1: 2}
    cn = np.column_stack([np.full(NT, CN[s]) for s in (3, 2, 1)])
    # Private access is the point: the walkthrough must run the SHIPPED trace.
    tau, counts = ScalingDA._tree_tau_trace(None, tree, NT, cn, colpos)  # noqa: SLF001
    if sum(counts.values()):
        msg = f"trace left segments unresolved: {counts}"
        raise RuntimeError(msg)
    by_seg = dict(zip(map(int, tree.seg_order), tau))
    # what tracing the raw physical cn would have given, for the comparison
    raw = {3: 0.0, 2: 1.0 / CN[3], 1: 1.0 / CN[3] + 1.0 / CN[2]}
    return by_seg, raw


def area_factor(seg: int) -> float:
    return (AREA[seg] / AREA[3]) ** THETA


def corrected(dq_gage: np.ndarray, tau: dict[int, float]) -> dict[int, np.ndarray]:
    """dQ(s, t) = dQ_o(t + tau_s) * (A_s/A_o)^theta, then added to background."""
    out = {}
    for seg in (3, 2, 1):
        shift = int(np.rint(tau[seg]))
        moved = np.full(NT, 0.0)
        moved[: NT - shift] = dq_gage[shift:]        # read FORWARD, t + tau
        out[seg] = BACKGROUND[seg] + moved * area_factor(seg)
    return out


def main() -> None:  # noqa: PLR0915  (linear figure assembly, no logic to extract)
    dq = innovation()
    tau, tau_raw = traced_tau()
    da = ScalingDA.__new__(ScalingDA)
    da.innovation_spread_h = SPREAD_H
    dq_bar = da._smooth_innovation(dq, dt=DT)  # noqa: SLF001  (shipped code, see above)
    corr = corrected(dq, tau)

    # ------------------------------------------------- verification table
    print(f"theta = {THETA}; factors (A_s/A_o)^theta:")
    for seg, name in ((3, "S3 gage"), (2, "S2"), (1, "S1")):
        print(f"  {name:7s} A={AREA[seg]:5.0f} km2  factor = "
              f"({AREA[seg]:.0f}/100)^{THETA} = {area_factor(seg):.4f}")
    print("traced tau (steps), shipped code vs raw-physics 1/cn:")
    for seg, name in ((3, "S3 gage"), (2, "S2"), (1, "S1")):
        print(f"  {name:7s} tau = {tau[seg]:.2f}   unclamped would be "
              f"{tau_raw[seg]:.2f}")
    t_peek = BUMP_LO - int(np.rint(tau[2]))
    print(f"spot check, S2 at t={t_peek}: {BACKGROUND[2]:.0f} + "
          f"{OBS_BUMP:.0f}*{area_factor(2):.4f} = {corr[2][t_peek]:.4f}")

    # --------------------------------------------------------- the figure
    steps = np.arange(NT)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for a in ax.ravel():
        a.grid(alpha=0.25, linewidth=0.6)
        a.spines[["top", "right"]].set_visible(False)

    # (a) the gage: observation, model, innovation
    a = ax[0, 0]
    obs = Q_MODEL_GAGE + dq
    a.plot(steps, obs, color=C["obs"], lw=1.8, label="observed", drawstyle="steps-mid")
    a.plot(steps, np.full(NT, Q_MODEL_GAGE), color=C["model"], lw=1.8,
           label="modeled", drawstyle="steps-mid")
    a.fill_between(steps, Q_MODEL_GAGE, obs, color=C["obs"], alpha=0.18,
                   linewidth=0, step="mid")
    a.annotate(r"$\Delta Q_o(t) = Q_{obs}-Q$" + f" = +{OBS_BUMP:.0f}",
               xy=(12.5, 25), ha="center", fontsize=9)
    a.set_title("(a) innovation at the gage D (Eq. 1)", fontsize=10)
    a.set_xlabel("timestep")
    a.set_ylabel(r"discharge (m$^3$/s)")
    a.legend(frameon=False, fontsize=8)

    # (b) the backward trace: ONE cumulative walk along the path from the gage.
    # The wave first crosses reach S3 (cn = 0.5 per step), then reach S2
    # (cn = 2.0, clamped to 1 per step). y = reach lengths covered so far.
    a = ax[0, 1]
    per_step = [min(CN[3], 1.0)] * 2 + [min(CN[2], 1.0)] * 2
    per_step_raw = [CN[3]] * 2 + [CN[2]] * 2
    k = np.arange(0, len(per_step) + 1)
    walk = np.concatenate([[0.0], np.cumsum(per_step)])
    walk_raw = np.concatenate([[0.0], np.cumsum(per_step_raw)])
    a.plot(k, walk, color=C["obs"], lw=1.8, marker="o", ms=4,
           label=r"shipped trace, $\sum\min(c_n,1)$")
    a.plot(k, walk_raw, color=C["clamp"], lw=1.4, ls="--", marker="o", ms=3,
           label=r"raw physics, $\sum c_n$")
    for y, lbl in ((1.0, "reach S3 crossed"), (2.0, "reach S2 crossed")):
        a.axhline(y, color="#888888", lw=0.9, ls=":")
        a.annotate(lbl, xy=(4.0, y + 0.06), ha="right", fontsize=8,
                   color="#666666")
    a.annotate(r"$\tau$(S2) = 2", xy=(2, 1.0), xytext=(2.35, 0.62),
               fontsize=9, arrowprops={"arrowstyle": "-", "color": "#666666"})
    a.annotate(r"$\tau$(S1) = 3", xy=(3, 2.0), xytext=(3.3, 1.55),
               fontsize=9, arrowprops={"arrowstyle": "-", "color": "#666666"})
    a.annotate(r"raw physics would say $\tau$(S1) = 2.5:"
               "\nthe model's wave is slower (K floor)",
               xy=(2.5, 2.0), xytext=(0.25, 2.55), fontsize=8,
               color=C["clamp"],
               arrowprops={"arrowstyle": "-", "color": C["clamp"]})
    a.set_title("(b) backward Lagrangian trace (Eq. 4), walked from the gage",
                fontsize=10)
    a.set_xlabel("timesteps walked back from the gage")
    a.set_ylabel("reach lengths covered")
    a.set_xlim(0, 4.05)
    a.set_ylim(0, 3.4)
    a.legend(frameon=False, fontsize=8, loc="lower right")

    # (c) the corrections: shifted forward in read-time, scaled by area
    a = ax[1, 0]
    for seg, key, name in ((3, "s3", "gage seg S3"), (2, "s2", "S2"),
                           (1, "s1", "S1")):
        shift = int(np.rint(tau[seg]))
        moved = np.zeros(NT)
        moved[: NT - shift] = dq[shift:]
        a.plot(steps, moved * area_factor(seg), color=C[key], lw=1.8,
               drawstyle="steps-mid")
        lo = BUMP_LO - shift
        a.annotate(
            f"{name}: " + r"$\Delta Q_o(t+" + f"{shift}" + r")\cdot$"
            + f"{area_factor(seg):.3f}",
            xy=(lo + 0.2, OBS_BUMP * area_factor(seg) + 0.25),
            fontsize=8, color=C[key],
        )
    a.plot(steps, dq_bar * area_factor(1), color=C["s1"], lw=1.2, ls="--",
           alpha=0.7, drawstyle="steps-mid")
    a.annotate("S1 with the optional spread\n(T=0.5 h) instead of the lag",
               xy=(16.5, 2.2), fontsize=7.5, color=C["s1"])
    a.set_title("(c) upstream corrections (Eq. 2): earlier in $t$, smaller in "
                "amplitude", fontsize=10)
    a.set_xlabel("timestep")
    a.set_ylabel(r"$\Delta Q(s,t)$ (m$^3$/s)")

    # (d) corrected discharge at every segment
    a = ax[1, 1]
    for seg, key, name in ((3, "s3", "S3 (gage)"), (2, "s2", "S2"),
                           (1, "s1", "S1")):
        a.plot(steps, np.full(NT, BACKGROUND[seg]), color=C[key], lw=1.0,
               alpha=0.45)
        a.plot(steps, corr[seg], color=C[key], lw=1.8, drawstyle="steps-mid")
        a.annotate(f"{name}", xy=(0.4, BACKGROUND[seg] + 0.5), fontsize=8.5,
                   color=C[key])
    a.annotate("thin = background, thick = corrected", xy=(29.5, 27.6),
               ha="right", fontsize=8, color="#666666")
    a.set_title("(d) corrected discharge: the bump arrives, scaled, at each "
                "segment", fontsize=10)
    a.set_xlabel("timestep")
    a.set_ylabel(r"discharge (m$^3$/s)")

    out = Path(__file__).parent / "figures" / "method_demo.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
