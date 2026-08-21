"""TimeSlice discovery must not assume the 15 minute cadence token.

``build_da_sets`` spells candidate names ``<stamp>.15min.<family>.ncdf`` because
that is the NWM convention, but a producer writes the cadence it actually has.
An hourly directory named ``60min`` matched nothing, so every file was reported
missing and the run continued with no observations at all.
"""

from __future__ import annotations

import pytest

from troute.hyfeature_network_utilities import _check_timeslice_exists as hyf_check
from troute.nhd_network_utilities_v02 import _check_timeslice_exists as nhd_check

_STAMPS = ["2023-12-15_00:00:00", "2023-12-15_01:00:00"]


def _wanted(cadence: str = "15min") -> list[str]:
    return [f"{s}.{cadence}.usgsTimeSlice.ncdf" for s in _STAMPS]


@pytest.fixture(params=[hyf_check, nhd_check], ids=["hyfeature", "nhd"])
def check(request):
    """Both build_da_sets implementations carry their own copy."""
    return request.param


def test_finds_files_written_at_another_cadence(tmp_path, check) -> None:
    for s in _STAMPS:
        (tmp_path / f"{s}.60min.usgsTimeSlice.ncdf").touch()
    assert check(_wanted(), tmp_path) == [
        f"{s}.60min.usgsTimeSlice.ncdf" for s in _STAMPS
    ]


def test_exact_names_still_win(tmp_path, check) -> None:
    for s in _STAMPS:
        (tmp_path / f"{s}.15min.usgsTimeSlice.ncdf").touch()
    assert check(_wanted(), tmp_path) == _wanted()


def test_absent_timestamps_are_dropped(tmp_path, check) -> None:
    (tmp_path / f"{_STAMPS[0]}.60min.usgsTimeSlice.ncdf").touch()
    assert check(_wanted(), tmp_path) == [f"{_STAMPS[0]}.60min.usgsTimeSlice.ncdf"]


def test_another_family_is_not_borrowed(tmp_path, check) -> None:
    """A usace file must not satisfy a usgs request at the same timestamp."""
    for s in _STAMPS:
        (tmp_path / f"{s}.60min.usaceTimeSlice.ncdf").touch()
    assert check(_wanted(), tmp_path) == []


def test_ambiguous_cadences_are_reported(tmp_path, check, caplog) -> None:
    """Two products for one timestamp is not the NWM layout; say so rather than pick quietly."""
    for cadence in ("60min", "05min"):
        (tmp_path / f"{_STAMPS[0]}.{cadence}.usgsTimeSlice.ncdf").touch()
    with caplog.at_level("WARNING"):
        got = check([_wanted()[0]], tmp_path)
    assert len(got) == 1
    assert any("Ambiguous TimeSlice" in r.message for r in caplog.records)


def test_a_stale_exact_name_still_wins(tmp_path, check) -> None:
    """Documents the precedence: an exact 15min file shadows another cadence."""
    (tmp_path / f"{_STAMPS[0]}.15min.usgsTimeSlice.ncdf").touch()
    (tmp_path / f"{_STAMPS[0]}.60min.usgsTimeSlice.ncdf").touch()
    assert check([_wanted()[0]], tmp_path) == [f"{_STAMPS[0]}.15min.usgsTimeSlice.ncdf"]
