"""An omitted data-assimilation sub-block must not crash its consumers.

Config validation leaves a block the user did not write present and NULL, while
consumers reach through it as ``.get(block, {}).get(flag, default)``. That chain dies
on ``NoneType.get``, so a config that never mentions reservoir DA took down the NHF
network build before routing a step.

Normalizing inside ``Config.model_dump`` fixes roughly twenty such chains at once, at
a choke point no future caller can bypass.

Note what the normalized shape does and does not promise: every block is a dict and
every FLAG inside it is absent, so nested ``.get(flag, False)`` reads answer
"configured off". But ``reservoir_da`` is filled with its two sub-blocks and is
therefore TRUTHY, unlike the ``None`` it replaces, so nothing may gate on
``if reservoir_da:``.
"""

from __future__ import annotations

from troute.config.config import normalize_da_blocks


def _dumped(da: dict) -> dict:
    return {"compute_parameters": {"data_assimilation_parameters": da}}


def _da(cfg: dict) -> dict:
    return cfg["compute_parameters"]["data_assimilation_parameters"]


def test_null_blocks_become_empty_dicts() -> None:
    da = _da(normalize_da_blocks(
        _dumped({"streamflow_da": None, "reservoir_da": None, "diversion_da": None})
    ))
    assert da == {
        "streamflow_da": {},
        "diversion_da": {},
        "reservoir_da": {"reservoir_persistence_da": {}, "reservoir_rfc_da": {}},
    }


def test_the_chain_that_crashed_now_resolves() -> None:
    """Exactly the expression in nhf_preprocess._great_lakes_for_da."""
    da = _da(normalize_da_blocks(_dumped({"reservoir_da": None})))
    assert (
        da.get("reservoir_da", {})
        .get("reservoir_persistence_da", {})
        .get("reservoir_persistence_greatLake", False)
    ) is False


def test_a_nested_null_is_normalized_too() -> None:
    """A written reservoir_da with an unwritten sub-block is the same hazard."""
    da = _da(normalize_da_blocks(
        _dumped({"reservoir_da": {"reservoir_persistence_da": None,
                                  "reservoir_rfc_da": None}})
    ))
    assert da["reservoir_da"]["reservoir_persistence_da"] == {}
    assert da["reservoir_da"]["reservoir_rfc_da"] == {}


def test_configured_blocks_are_left_alone() -> None:
    da = _da(normalize_da_blocks(_dumped(
        {"reservoir_da": {"reservoir_persistence_da":
                          {"reservoir_persistence_usgs": True}}}
    )))
    assert da["reservoir_da"]["reservoir_persistence_da"] == {
        "reservoir_persistence_usgs": True
    }


def test_the_nested_flags_read_as_configured_off() -> None:
    """The invariant consumers may actually rely on.

    NOT parent falsiness: reservoir_da is filled with its sub-blocks and is truthy.
    What holds is that every flag inside is absent, so the nested reads answer False.
    """
    da = _da(normalize_da_blocks(_dumped({"reservoir_da": None})))
    assert da["reservoir_da"], "filled reservoir_da is truthy; gate on the flags"
    persistence = da["reservoir_da"]["reservoir_persistence_da"]
    assert persistence.get("reservoir_persistence_usgs", False) is False
    assert da["reservoir_da"]["reservoir_rfc_da"].get(
        "reservoir_rfc_forecasts", False) is False
    assert not da["streamflow_da"]
    assert not da["diversion_da"]


def test_a_config_without_the_section_is_untouched() -> None:
    assert normalize_da_blocks({"compute_parameters": {}}) == {"compute_parameters": {}}
    assert normalize_da_blocks({}) == {}
