"""Config-level guards for the simple-scaling DA.

Every failure mode here shares one signature: the run completes, writes a full set
of output, and exits 0 with NO assimilation (or with the wrong gages assimilated),
indistinguishable from success. So each is rejected at parse time, and these tests
pin the rejections to the config layer where every pydantic entry point (-V4/-V5
CLI, BMI, model_DAforcing) inherits them.
"""

from __future__ import annotations

import argparse

import pytest
import yaml
from nwm_routing.input import _input_handler_v03
from pydantic import ValidationError

from troute.config import Config


def _config(network_type: str = "NHF", tmp_path=None, **scaling) -> dict:
    ts = tmp_path / "timeslices"
    ts.mkdir(exist_ok=True)
    gpkg = tmp_path / "domain.gpkg"
    gpkg.touch()
    return {
        "network_topology_parameters": {
            "supernetwork_parameters": {
                "geo_file_path": str(gpkg),
                "network_type": network_type,
            },
        },
        "compute_parameters": {
            "forcing_parameters": {"nts": 24, "dt": 300},
            "data_assimilation_parameters": {
                "usgs_timeslices_folder": str(ts),
                "streamflow_da": {
                    "streamflow_nudging": False,
                    "streamflow_scaling": True,
                    "streamflow_scaling_parameters": dict(scaling),
                },
            },
        },
    }


class TestNetworkTypeGuard:
    """scaling_da is implemented for NHF only; anywhere else no driver constructs it."""

    def test_nhf_is_accepted(self, tmp_path):
        Config(**_config("NHF", tmp_path))

    @pytest.mark.parametrize("network_type", ["HYFeaturesNetwork", "NHDNetwork"])
    def test_other_network_types_are_rejected(self, network_type, tmp_path):
        with pytest.raises(ValidationError, match="NHF network only"):
            Config(**_config(network_type, tmp_path))

    @pytest.mark.parametrize("network_type", ["HYFeaturesNetwork", "NHDNetwork"])
    def test_disabled_scaling_da_passes_everywhere(self, network_type, tmp_path):
        data = _config(network_type, tmp_path)
        da = data["compute_parameters"]["data_assimilation_parameters"]
        da["streamflow_da"]["streamflow_scaling"] = False
        Config(**data)

    def test_v03_raw_yaml_path_is_guarded_too(self, tmp_path):
        """-V3 bypasses pydantic entirely, so it carries its own copy of the guard."""
        gpkg = tmp_path / "domain.gpkg"
        gpkg.touch()
        cfg = tmp_path / "v3.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "log_parameters": {},
                    "network_topology_parameters": {
                        "supernetwork_parameters": {"geo_file_path": str(gpkg)}
                    },
                    "compute_parameters": {
                        "data_assimilation_parameters": {
                            "streamflow_da": {"streamflow_scaling": True}
                        }
                    },
                    "output_parameters": {},
                }
            )
        )
        args = argparse.Namespace(custom_input_file=str(cfg))
        with pytest.raises(ValueError, match="-V3 driver does not"):
            _input_handler_v03(args)


class TestMethodSelection:
    def test_nudging_and_scaling_are_mutually_exclusive(self, tmp_path):
        data = _config("NHF", tmp_path)
        data["compute_parameters"]["data_assimilation_parameters"]["streamflow_da"][
            "streamflow_nudging"
        ] = True
        with pytest.raises(ValidationError, match="mutually exclusive"):
            Config(**data)

    def test_legacy_scaling_da_block_is_rejected_with_migration_hint(self, tmp_path):
        data = _config("NHF", tmp_path)
        da = data["compute_parameters"]["data_assimilation_parameters"]
        da["streamflow_da"] = {"streamflow_nudging": False}
        da["scaling_da"] = {"enabled": True}
        with pytest.raises(ValidationError, match="has moved"):
            Config(**data)


class TestDeclaredFieldsSurviveTheDump:
    """The model is extra='forbid', and the drivers read model_dump(), so an
    implemented option that is not declared is unreachable from any valid config."""

    def test_operational_fields_validate(self, tmp_path):
        cfg = Config(**_config("NHF", tmp_path, min_flow_cms=1e-3, celerity_mps=1.25))
        sda = cfg.compute_parameters.data_assimilation_parameters.streamflow_da
        assert sda.streamflow_scaling
        dumped = cfg.model_dump()["compute_parameters"]["data_assimilation_parameters"][
            "streamflow_da"
        ]["streamflow_scaling_parameters"]
        assert dumped["min_flow_cms"] == 1e-3
        assert dumped["celerity_mps"] == 1.25

    def test_unknown_keys_still_forbidden(self, tmp_path):
        with pytest.raises(ValidationError):
            Config(**_config("NHF", tmp_path, upstream_spread=True))


class TestOperationalKnobs:

    def test_chunk_zero_means_explicitly_off(self, tmp_path):
        Config(**_config("NHF", tmp_path, spread_chunk_timesteps=0))

    def test_negative_chunk_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError):
            Config(**_config("NHF", tmp_path, spread_chunk_timesteps=-1))


class TestWindowEnvelope:
    """The envelope is validated on REALIZED run sets, not config arithmetic.

    max_loop_size counts forcing FILES: with 15-minute files, max_loop_size=24
    realizes 6 h windows, and stream_output_time can silently enlarge the count.
    So the driver validates the windows the run actually built (window hours
    must cover the lag horizon -- the halo is ONE window deep -- and be a whole
    number of screen intervals). The final window is exempt: it clamps at the
    run end identically under every partitioning.
    """

    @staticmethod
    def _sda(**kw):
        from nwm_routing.scaling_da_apply import ScalingDA

        o = ScalingDA.__new__(ScalingDA)
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def test_window_shorter_than_lag_is_rejected(self):
        from nwm_routing.scaling_da_apply import validate_window_envelope

        sda = self._sda(max_travel_time_h=48.0, screen_interval_h=24.0)
        runs = [{"nts": 288, "dt": 300}, {"nts": 288, "dt": 300}, {"nts": 10, "dt": 300}]
        with pytest.raises(ValueError, match="one window deep"):
            validate_window_envelope(runs, sda)  # 24 h windows < 48 h horizon

    def test_covering_windows_pass_and_final_window_is_exempt(self):
        from nwm_routing.scaling_da_apply import validate_window_envelope

        sda = self._sda(max_travel_time_h=48.0, screen_interval_h=24.0)
        validate_window_envelope(
            [{"nts": 576, "dt": 300}, {"nts": 100, "dt": 300}], sda
        )

    def test_non_positive_horizon_is_rejected_at_parse(self, tmp_path):
        # The lag and its reach limit are part of the method, not a switch:
        # max_travel_time_h: 0 must fail config validation.
        with pytest.raises(ValidationError):
            Config(**_config("NHF", tmp_path, max_travel_time_h=0.0))

    @pytest.mark.parametrize(
        "field", ["max_travel_time_h", "celerity_mps", "screen_interval_h"]
    )
    def test_infinite_values_are_rejected_at_parse(self, tmp_path, field):
        # gt=0 alone admits .inf; celerity_mps=inf makes every dx/c zero, which
        # silently resurrects the removed un-lagged mode.
        with pytest.raises(ValidationError):
            Config(**_config("NHF", tmp_path, **{field: float("inf")}))

    def test_both_drivers_wire_the_envelope_validator(self):
        # This repo has shipped a per-driver gate in only ONE of its two driver
        # copies before (see build_da_sets in CLAUDE.md), with a test pointed at
        # the copy the feature never calls. Pin the wiring in BOTH.
        import inspect

        import nwm_routing.nhf_routing as cli

        bmi = pytest.importorskip("troute_nwm_bmi.troute_model")
        assert "validate_window_envelope" in inspect.getsource(cli)
        assert "validate_window_envelope" in inspect.getsource(bmi.Model.run)

    def test_single_window_run_has_nothing_to_check(self):
        from nwm_routing.scaling_da_apply import validate_window_envelope

        sda = self._sda(max_travel_time_h=48.0, screen_interval_h=24.0)
        validate_window_envelope([{"nts": 5, "dt": 300}], sda)
