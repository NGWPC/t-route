"""One config must produce one window partition, on the CLI and under the BMI.

The two drivers each had their own chunking. The CLI filled to ``max_loop_size`` and
folded a short final remainder into its neighbor; the BMI spread the update evenly. On
the same config that is ``[24, 24, 24, 28]`` against ``[20, 20, 20, 20, 20]``, and the
folded window is wider than the memory the caller sized for.

Both now call the functions here, so the partition is shared by construction rather
than by two implementations agreeing.
"""

from __future__ import annotations

import inspect
from itertools import pairwise

import pytest

from troute.window_plan import AUTO_WINDOW, plan_windows, resolve_window


class TestResolveWindow:
    def test_zero_is_automatic(self):
        assert resolve_window(0) == AUTO_WINDOW
        assert resolve_window(None) == AUTO_WINDOW

    def test_an_explicit_value_is_kept(self):
        assert resolve_window(9) == 9

    def test_the_span_raises_either_form(self):
        assert resolve_window(0, 48) == 48
        assert resolve_window(9, 48) == 48

    def test_a_span_under_the_window_changes_nothing(self):
        assert resolve_window(24, 12) == 24

    def test_never_zero(self):
        assert resolve_window(0, 0) >= 1


class TestPlanWindows:
    @pytest.mark.parametrize(
        ("n", "window", "span"),
        [(100, 24, 12), (96, 24, 0), (47, 24, 24), (3, 24, 0), (1, 24, 0),
         (144, 24, 12), (25, 24, 12), (48, 48, 48), (7, 3, 3)],
    )
    def test_bounds_tile_the_run_without_gap_or_overlap(self, n, window, span):
        bounds = plan_windows(n, window, span)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        assert all(a[1] == b[0] for a, b in pairwise(bounds))

    @pytest.mark.parametrize(
        ("n", "window", "span"),
        [(100, 24, 12), (96, 24, 0), (144, 24, 12), (25, 24, 12), (7, 3, 0)],
    )
    def test_no_window_exceeds_the_requested_width(self, n, window, span):
        """The fold this replaced could double it: 47 columns at 24 became one of 47."""
        widths = [stop - start for start, stop in plan_windows(n, window, span)]
        assert max(widths) <= window

    @pytest.mark.parametrize(("n", "window", "span"), [(7, 3, 3), (47, 24, 24)])
    def test_the_one_case_that_may_exceed_it_is_the_single_window(self, n, window, span):
        """When no split holds every window at the span, one window is the only
        artifact-free answer, and it is necessarily wider than asked for.

        Callers with a memory ceiling must notice this: it is the single case where
        the partition is wider than what they sized for. Both drivers warn, and the
        BMI raises when the ceiling cannot hold it.
        """
        bounds = plan_windows(n, window, span)
        assert len(bounds) == 1
        assert bounds[0][1] - bounds[0][0] == n > window

    @pytest.mark.parametrize(
        ("n", "window", "span"),
        [(100, 24, 12), (144, 24, 12), (25, 24, 12), (47, 24, 24), (7, 3, 3)],
    )
    def test_no_window_falls_under_the_span(self, n, window, span):
        widths = [stop - start for start, stop in plan_windows(n, window, span)]
        assert min(widths) >= span or len(widths) == 1

    def test_an_empty_run_has_no_windows(self):
        assert plan_windows(0, 24, 12) == []

    def test_widths_differ_by_at_most_one(self):
        widths = [stop - start for start, stop in plan_windows(100, 24, 12)]
        assert max(widths) - min(widths) <= 1


def test_both_drivers_call_the_shared_rule():
    """A regression guard: neither driver may grow its own chunking again.

    Reading the source, not the behavior, because the alternative is a full network
    build on one side and a BMI model on the other.
    """
    from troute_nwm_bmi import troute_model

    from troute import AbstractNetwork

    cli = inspect.getsource(AbstractNetwork.AbstractNetwork.build_forcing_sets)
    bmi = inspect.getsource(troute_model.Model._build_run_sets)
    for name, src in (("-V5 build_forcing_sets", cli), ("BMI _build_run_sets", bmi)):
        assert "plan_windows(" in src, f"{name} no longer uses the shared split"
        assert "resolve_window(" in src, f"{name} no longer uses the shared resolution"
