"""One window partition, shared by the CLI and the BMI.

Both drivers chunk a run's forcing into windows, and the same config has to produce
the same windows on either. They had drifted: the CLI filled to ``max_loop_size`` and
folded a short final remainder into its neighbor, while the BMI spread the update
evenly, so 100 columns at a window of 24 came out ``[24, 24, 24, 28]`` on one and
``[20, 20, 20, 20, 20]`` on the other.

The even split is the one that survives both constraints. Filling to a fixed width
leaves a remainder that is either shorter than the DA span, which makes discharge
depend on where the boundary fell, or folded into its neighbor, which makes that
window wider than the memory the caller sized for.
"""

from __future__ import annotations

import math

__all__ = ["AUTO_WINDOW", "plan_windows", "resolve_window"]

# Forcing columns per window when max_loop_size is 0 (automatic). The Tier A sweep in
# benchmark/RESULTS.md plateaus in wall time here while peak RSS keeps climbing past it,
# so a wider window buys memory pressure and no speed. Was the schema default.
AUTO_WINDOW = 24

WindowBounds = list[tuple[int, int]]


def plan_windows(n_columns: int, window: int, span: int = 0) -> WindowBounds:
    """Half-open ``(start, stop)`` bounds covering ``n_columns`` forcing columns.

    No window is wider than ``window``, and widths differ by at most one, so a
    window never exceeds the ceiling of the average and the memory one costs stays
    bounded by what the caller asked for.

    Every window also reaches ``span``, the DA's own horizon in columns, whenever any
    split can: a window below it reads across a boundary it cannot supply. When none
    can, the run goes in a single window, which always covers it. Callers that have a
    memory ceiling check the widest returned bound against it; this function has no
    opinion about memory.
    """
    if n_columns <= 0:
        return []
    width = max(1, min(window, n_columns))
    count = math.ceil(n_columns / width)
    if span > 0 and count > 1 and n_columns // count < span:
        # Give up the CAP minimally, not the split entirely. The most windows whose
        # floor still reaches the span is n // span, and that exceeds the requested
        # width by at most about one column. Collapsing to a single window instead
        # turned a 2161-column run at span 60 into one 2161-column window, ~120 GB,
        # while 36 windows of 60 and 61 hold the span; one column of run length was
        # the difference. The caller's own memory check gates the small excess.
        count = max(1, n_columns // span)
    base, extra = divmod(n_columns, count)
    bounds: WindowBounds = []
    start = 0
    for i in range(count):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def resolve_window(
    configured: int | None, span: int = 0, output_cols: int = 0
) -> int:
    """Columns per window from the config alone, identically for either driver.

    ``configured`` of 0 or None is the schema's "automatic" and resolves to
    :data:`AUTO_WINDOW`. ``span`` raises it either way, since a window under the DA's
    horizon changes discharge, which no memory knob is allowed to do. ``output_cols``
    raises it too: one window writes one output part, so a window under the requested
    ``stream_output_time`` cannot fill one.

    The output cadence lives HERE rather than in one driver because it changes the
    routing partition. The CLI enlarged for it and the BMI did not, so the same config
    with stream_output_time 48 routed 48-column windows on one and 24 on the other,
    and under a nonzero DA span that is a difference in discharge.

    Only the memory cap is left to a driver, since only the BMI measures memory.
    """
    return max(int(configured or AUTO_WINDOW), span, output_cols, 1)
