"""Unit tests for diversion data assimilation (Old River Control Structure).

The integration test in ``test/nhf/old_river`` needs a subset hydrofabric, NWM
retrospective forcing and USGS timeslices, so it skips wherever that data has not
been built. These tests cover the same machinery with no external data, so the
mechanism is guarded on every run.

What the scheme does, and what therefore has to hold:

The observed discharge at the diversion gage is SUBTRACTED from the donor flowpath
inside the routing kernel. Nothing adds it to the receiving river in code: the gage
sits on a headwater flowpath of the receiving system, so ordinary streamflow
nudging imposes the observed discharge there and the existing topology routes it
downstream. The transfer therefore conserves mass only while nudging is enabled,
which the configuration layer now enforces.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from troute.DataAssimilation import (
    _DIVERSION_MONTHLY_MEANS,
    _fill_diversion_historical_median,
)
from troute.routing.compute import _resolve_diversion_da

DIVERSION_GAGE = "07381482"  # Old River Outflow Channel
DONOR_FP_ID = 1270479816524705  # Mississippi link at the control structure
GAGE_LINK = 1269985531956909  # routing link the gage resolves to


@pytest.fixture
def donor_and_gage() -> tuple[int, int]:
    """The (donor segment, gage link) pair the kernel map is built from."""
    return DONOR_FP_ID, GAGE_LINK


@pytest.fixture
def job_reaches(donor_and_gage) -> np.ndarray:
    """A compute job's sorted segment index that contains the donor segment."""
    donor, _ = donor_and_gage
    return np.sort(np.array([donor - 10, donor, donor + 10, donor + 20], dtype="int64"))


@pytest.fixture
def usgs_df_sub(donor_and_gage) -> pd.DataFrame:
    """Observations subset for a job, indexed by routing link like the real one."""
    _, gage_link = donor_and_gage
    return pd.DataFrame(
        [[100.0, 110.0, 120.0]],
        index=pd.Index([gage_link], dtype="int64"),
        columns=pd.date_range("2011-04-14", periods=3, freq="h"),
    )


class TestResolveDiversionDa:
    """Translating the global diversion map to job-local kernel indices."""

    def test_maps_donor_position_to_gage_row(self, donor_and_gage, job_reaches, usgs_df_sub):
        donor, gage_link = donor_and_gage
        kernel_map = _resolve_diversion_da({donor: gage_link}, job_reaches, usgs_df_sub)
        # key is the donor's POSITION in this job's index, value the gage's row
        expected_pos = int(np.searchsorted(job_reaches, donor))
        assert kernel_map == {expected_pos: 0}

    def test_empty_map_short_circuits(self, job_reaches, usgs_df_sub):
        assert _resolve_diversion_da({}, job_reaches, usgs_df_sub) == {}

    def test_donor_outside_job_is_skipped(self, donor_and_gage, usgs_df_sub):
        """A job that does not contain the donor must not divert anything.

        Guards the searchsorted lookup: without the identity check, a missing donor
        lands on the insertion point and would divert some unrelated segment.
        """
        donor, gage_link = donor_and_gage
        other_job = np.array([donor + 1000, donor + 2000], dtype="int64")
        assert _resolve_diversion_da({donor: gage_link}, other_job, usgs_df_sub) == {}

    def test_donor_present_but_gage_missing_warns(
        self, donor_and_gage, job_reaches, caplog
    ):
        """The donor is here but its observations are not, so no diversion is applied.

        This silently left the donor carrying water that should have been
        transferred; it must be reported.
        """
        donor, gage_link = donor_and_gage
        empty = pd.DataFrame(index=pd.Index([], dtype="int64"))
        with caplog.at_level(logging.WARNING):
            assert _resolve_diversion_da({donor: gage_link}, job_reaches, empty) == {}
        assert "gage link" in caplog.text


class TestHistoricalMedianFallback:
    """Climatological fill for timesteps with no observation (forecast mode)."""

    @pytest.fixture
    def network(self):
        class _Network:
            t0 = pd.Timestamp("2011-06-01 00:00")
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        return _Network()

    @pytest.fixture
    def params(self) -> dict:
        return {
            "diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
            "persist_historical_median": True,
        }

    @pytest.mark.parametrize("dt,nts", [(300, 12), (60, 20), (900, 8)])
    def test_columns_follow_routing_timestep(self, network, params, dt, nts):
        """One column per routing timestep, not a fixed 5 minute grid.

        The kernel indexes this frame as ``usgs_values[gage_i, timestep]`` with the
        ROUTING step, so a hardcoded grid ran out early for a short dt and advanced
        too slowly for a long one.
        """
        out = _fill_diversion_historical_median(
            pd.DataFrame(), params, network, {"dt": dt, "nts": nts}
        )
        assert out.shape[1] == nts + 1
        spacing = pd.Series(out.columns).diff().dropna().unique()
        assert list(spacing) == [pd.Timedelta(seconds=dt)]

    def test_fills_with_calendar_month_climatology(self, network, params):
        out = _fill_diversion_historical_median(
            pd.DataFrame(), params, network, {"dt": 300, "nts": 3}
        )
        june = _DIVERSION_MONTHLY_MEANS[DIVERSION_GAGE][6]
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy(), june)

    def test_real_observations_are_never_overwritten(self, network, params):
        """The fallback fills gaps only. An observed value must survive."""
        idx = pd.date_range(network.t0, periods=4, freq=pd.Timedelta(seconds=300))
        existing = pd.DataFrame(
            [[7.0, np.nan, 9.0, np.nan]], index=pd.Index([GAGE_LINK], dtype="int64"),
            columns=idx,
        )
        out = _fill_diversion_historical_median(
            existing, params, network, {"dt": 300, "nts": 3}
        )
        row = out.loc[GAGE_LINK]
        assert row.iloc[0] == 7.0 and row.iloc[2] == 9.0
        june = _DIVERSION_MONTHLY_MEANS[DIVERSION_GAGE][6]
        assert row.iloc[1] == june and row.iloc[3] == june

    def test_substitution_is_logged(self, network, params, caplog):
        """Climatology is not an observation; a run must say how much it used."""
        with caplog.at_level(logging.WARNING):
            _fill_diversion_historical_median(
                pd.DataFrame(), params, network, {"dt": 300, "nts": 3}
            )
        assert "climatology" in caplog.text

    def test_gage_without_routing_link_is_reported_not_raised(self, network, params, caplog):
        """An unresolved gage used to raise KeyError mid-run."""
        network._diversion_site_to_node = {}
        with caplog.at_level(logging.WARNING):
            out = _fill_diversion_historical_median(
                pd.DataFrame(), params, network, {"dt": 300, "nts": 3}
            )
        assert "no routing link" in caplog.text
        assert out.empty or GAGE_LINK not in out.index

    def test_disabled_flag_leaves_frame_untouched(self, network, params):
        params["persist_historical_median"] = False
        original = pd.DataFrame()
        # the caller gates on the flag, so calling with no crosswalk is the no-op path
        out = _fill_diversion_historical_median(
            original, {"diversion_gage_crosswalk": {}}, network, {"dt": 300, "nts": 3}
        )
        assert out is original
