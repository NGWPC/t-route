"""A lite channel restart reaches the kernel in the kernel's column order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from troute.NHF import NHF, Q0_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

LINKS = [11, 12, 21]  # two links on flowpath 1, one on flowpath 2
FP_OF_LINK = {11: 1, 12: 1, 21: 2}
T0 = pd.Timestamp("2011-04-19")


def _network() -> NHF:
    """A bare NHF with only the state the restart reader touches, as the sibling tests build it."""
    net = NHF.__new__(NHF)
    net._dataframe = pd.DataFrame(  # pyright: ignore[reportPrivateUsage]
        {"fp_id": [FP_OF_LINK[link] for link in LINKS]},
        index=pd.Index(LINKS, name="up_node_id"),
    )
    net._waterbody_df = pd.DataFrame(  # pyright: ignore[reportPrivateUsage]
        index=pd.Index([], name="lake_id"),
    )
    net.break_points = {"break_network_at_waterbodies": False}
    # Set by __init__, which this bare instance skips; read by the flowpath branch.
    net.div_reverse_lookup = {}  # pyright: ignore[reportAttributeAccessIssue]
    return net


def _read(path: Path) -> pd.DataFrame:
    net = _network()
    net.restart_parameters = {"lite_channel_restart_file": str(path)}
    net.initial_warmstate_preprocess(from_files=True, value_dict={})
    return net.q0


def test_a_flowpath_restart_is_broadcast_in_kernel_order(tmp_path: Path) -> None:
    """The hot-start layout: one row per flowpath, columns in its own order.

    The kernel reads the row by position, so the order is pinned literally and every
    slot carries a distinct value.
    """
    hot = pd.DataFrame({
        "feature_id": [1, 2], "qd0": [12.0, 22.0], "h0": [0.5, 0.7],
        "qu0": [11.0, 21.0], "ql0": [0.1, 0.2], "time": T0,
    })
    hot.to_pickle(tmp_path / "restart.pkl")
    out = _read(tmp_path / "restart.pkl")
    assert list(out.columns) == ["qu0", "qd0", "h0", "ql0"]
    assert out.loc[12].to_numpy().tolist() == pytest.approx([11.0, 12.0, 0.5, 0.1])
    assert out.loc[21].to_numpy().tolist() == pytest.approx([21.0, 22.0, 0.7, 0.2])
