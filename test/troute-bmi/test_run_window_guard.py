"""The run-window guard must refuse a short update only when the partition matters.

Under the scaling DA the window partition can be part of the result, so
``_build_run_sets`` refuses to shrink the configured ``max_loop_size`` silently.
That is only true when the DA has a TIME span: with ``innovation_spread_h`` at 0
and the lag off, the spread is output-only and per timestep, so a shorter window
is bit-identical (pinned in test/troute-nwm/test_scaling_da_in_kernel.py).

The case that forced this: the NWM Standard AnA is 3 hours, so 3 forcing columns,
against a ``max_loop_size`` default of 24. Refusing there made the operational
config shipped in test/operational_configurations/ unable to run the DA at all.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from troute_nwm_bmi.troute_model import Model

_COLUMNS = ["202001010100", "202001010200", "202001010300"]  # 3 hourly columns


class _NetworkStub:
    t0 = datetime(2020, 1, 1, 0, 0)


class _ScalingDAStub:
    def __init__(self, spread_h: float, lag: bool = False):
        self.innovation_spread_h = spread_h
        self.travel_time_lag = lag
        self.lag_window_h = 48.0


def _model(max_loop_size: int, scaling_da: _ScalingDAStub | None) -> Model:
    model = Model.__new__(Model)  # skip __init__ (no config/network build)
    model._config = {
        "compute_parameters": {
            "forcing_parameters": {
                "max_loop_size": max_loop_size,
                "qts_subdivisions": 12,
                "dt": 300,
            }
        }
    }
    model.dt = 300
    model._network = _NetworkStub()
    model._scaling_da = scaling_da
    return model


def _qlats() -> pd.DataFrame:
    return pd.DataFrame(1.0, index=[10, 20], columns=_COLUMNS)


def test_a_short_update_runs_when_the_da_has_no_span():
    """Standard AnA shape: 3 columns against the max_loop_size default of 24."""
    runs = list(_model(24, _ScalingDAStub(0.0))._build_run_sets(_qlats()))
    assert len(runs) == 1, f"expected one window over 3 columns, got {len(runs)}"
    assert runs[0]["nts"] == 3 * 12  # routing timesteps, not columns


def test_a_short_update_is_refused_once_the_span_is_nonzero():
    """With a forward average the partition reaches the result, so shrinking the
    window silently would change discharge. That still has to fail loudly."""
    model = _model(24, _ScalingDAStub(2.0))
    with pytest.raises(ValueError, match="forcing timestep"):
        list(model._build_run_sets(_qlats()))


def test_a_span_that_fits_the_update_runs():
    """The span is only a problem when the update cannot cover it."""
    runs = list(_model(2, _ScalingDAStub(2.0))._build_run_sets(_qlats()))
    assert len(runs) == 2, f"expected windows of 2 and 1, got {len(runs)}"
    assert [r["nts"] for r in runs] == [2 * 12, 1 * 12]


def test_no_assimilation_is_unaffected():
    """Without the DA a short update was always served; keep it that way."""
    runs = list(_model(24, None)._build_run_sets(_qlats()))
    assert len(runs) == 1
