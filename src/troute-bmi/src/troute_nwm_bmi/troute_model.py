"""Basic Model Interface backing model for NGEN t-route."""
import yaml
import numpy as np
import pandas as pd
from datetime import timedelta
from troute.config import Config

from troute.NHDNetwork import NHDNetwork
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from troute.DataAssimilation import DataAssimilation

import troute.hyfeature_network_utilities as hnu

import nwm_routing.__main__ as nwm_routing
from nwm_routing.output import nwm_output_generator
from nwm_routing.log_level_set import log_level_set

class Model:
    dt: int

    def __init__(self, config_file: str):
        self._time = 0.0

        with open(config_file) as reader:
            data = yaml.load(reader, Loader=yaml.SafeLoader)
        self._config: dict = Config.with_strict_mode(**data).dict()

        log_level_set(self.log_parameters)

        self.dt = int(self.forcing_parameters["dt"])

        if self.supernetwork_parameters["network_type"] == "HYFeaturesNetwork":
            self._network = HYFeaturesNetwork(
                supernetwork_parameters=self.supernetwork_parameters,
                waterbody_parameters=self.waterbody_parameters,
                data_assimilation_parameters=self.data_assimilation_parameters,
                restart_parameters=self.restart_parameters,
                compute_parameters=self.compute_parameters,
                forcing_parameters=self.forcing_parameters,
                hybrid_parameters=self.hybrid_parameters,
                preprocessing_parameters=self.preprocessing_parameters,
                output_parameters=self.output_parameters,
                ## verbose and showtiming?
            )
        elif self.supernetwork_parameters["network_type"] == "NHDNetwork":
            self._network = NHDNetwork(
                supernetwork_parameters=self.supernetwork_parameters,
                waterbody_parameters=self.waterbody_parameters,
                restart_parameters=self.restart_parameters,
                forcing_parameters=self.forcing_parameters,
                compute_parameters=self.compute_parameters,
                data_assimilation_parameters=self.data_assimilation_parameters,
                hybrid_parameters=self.hybrid_parameters,
                output_parameters=self.output_parameters,
                ## versbose and showtiming?
            )
        else:
            raise Exception("Supernetwork network type must be HYFeaturesNetwork or NHDNetwork")

        self._run_sets = self._network.build_forcing_sets()

        # Data data assimilation
        if self.data_assimilation_parameters:
            self._da_sets = hnu.build_da_sets(self.data_assimilation_parameters, self._run_sets, self._network.t0)
        else:
            self._da_sets = []
        self._data_assimilation = DataAssimilation(
            network=self._network,
            data_assimilation_parameters=self.data_assimilation_parameters,
            run_parameters={},
            waterbody_parameters=self.waterbody_parameters,
            from_files=True,
            value_dict=None,
            da_run=self._da_sets[0] if len(self._da_sets) else {},
        )

        # Pass empty subnetwork list to nwm_route. These objects will be calculated/populated
        # on first iteration of for loop only. For additional loops this will be passed
        # to function from inital loop.     
        self._subnetwork = [None, None, None]

        self._df_data = {}


    def update(self, bmi_values: dict):
        qlat_values = bmi_values["land_surface_water_source__volume_flow_rate"]
        time = self._network.t0 + timedelta(seconds=self.time)
        timestamp = time.strftime("%Y%m%d%H%M")
        self._df_data[timestamp] = np.array(qlat_values)
        self._time += self.dt


    def run(self, bmi_values: dict):
        nts = self.nts
        qts_subdivisions = self.forcing_parameters.get('qts_subdivisions', 12)

        ## setup the qlats dataframe from the update() data
        qlats = pd.DataFrame(data=self._df_data, index=bmi_values["land_surface_water_source__id"])
        # Take flowpath ids entering NEXUS and replace NEXUS ids by the upstream flowpath ids
        qlats = qlats.rename(index=self._network.downstream_flowpath_dict)
        # create zero values for missing values
        missing = self._network.segment_index[~self._network.segment_index.isin(qlats.index)]
        zeros = pd.DataFrame(data=0.0, index=missing, columns=qlats.columns)
        qlats = pd.concat([qlats, zeros]).sort_index()

        if len(bmi_values["upstream_id"]) > 0:
            flowveldepth_interorder = {bmi_values['upstream_id'][0]: {"results": bmi_values['upstream_fvd']}}
        else:
            flowveldepth_interorder = {}

        run_results, self._subnetwork = nwm_routing.nwm_route(
            downstream_connections=self._network.connections,
            upstream_connections=self._network.reverse_network,
            waterbodies_in_connections=self._network.waterbody_connections,
            reaches_bytw=self._network._reaches_by_tw,
            parallel_compute_method=self.compute_parameters.get("parallel_compute_method", "serial"),
            compute_kernel=self.compute_parameters.get("compute_kernel"),
            subnetwork_target_size=self.compute_parameters.get('subnetwork_target_size'),
            cpu_pool=self.cpu_pool,
            t0=self._network.t0,
            dt=self.dt,
            nts=nts,
            qts_subdivisions=qts_subdivisions,
            independent_networks=self._network.independent_networks,
            param_df=self._network.dataframe,
            q0=self._network.q0,
            qlats=qlats,
            usgs_df=self._data_assimilation.usgs_df,
            lastobs_df=self._data_assimilation.lastobs_df,
            reservoir_usgs_df=self._data_assimilation.reservoir_usgs_df,
            reservoir_usgs_param_df=self._data_assimilation.reservoir_usgs_param_df,
            reservoir_usace_df=self._data_assimilation.reservoir_usace_df,
            reservoir_usace_param_df=self._data_assimilation.reservoir_usace_param_df,
            reservoir_rfc_df=self._data_assimilation.reservoir_rfc_df,
            reservoir_rfc_param_df=self._data_assimilation.reservoir_rfc_param_df,
            great_lakes_df=self._data_assimilation.great_lakes_df,
            great_lakes_param_df=self._data_assimilation.great_lakes_df,
            great_lakes_climatology_df=self._network.great_lakes_climatology_df,
            da_parameter_dict=self._data_assimilation.assimilation_parameters,
            assume_short_ts=self.compute_parameters.get('assume_short_ts', False),
            return_courant=self.compute_parameters.get('return_courant', False),
            waterbodies_df=self._network._waterbody_df,
            data_assimilation_parameters=self.waterbody_parameters,
            waterbody_types_df=self._network._waterbody_types_df,
            waterbody_type_specified=self._network.waterbody_type_specified,
            diffusive_network_data=self._network.diffusive_network_data,
            topobathy_df=self._network.topobathy_df,
            refactored_diffusive_domain=self._network.refactored_diffusive_domain,
            refactored_reaches=self._network.refactored_reaches,
            subnetwork_list=self._subnetwork,
            coastal_boundary_depth_df=self._network.coastal_boundary_depth_df,
            unrefactored_topobathy_df=self._network.unrefactored_topobathy_df,
            from_files=False,
            flowveldepth_interorder=flowveldepth_interorder,
        )

        # create initial conditions for next loop iteration
        # self._network.new_t0(self.dt, nts)
        self._network.new_q0(run_results)
        self._network.update_waterbody_water_elevation()
        
        # update reservoir parameters and lastobs_df
        self._data_assimilation.update_after_compute(run_results, self.dt * nts)

        run_params = {
            "t0": self._network.t0,
            "dt": self.dt,
            "nts": nts,
            "timesteps": self._waterbodies_timesteps(),
        }
        nwm_output_generator(
            run=run_params,
            results=run_results,
            supernetwork_parameters=self.supernetwork_parameters,
            output_parameters=self.output_parameters,
            parity_parameters=self.parity_parameters,
            restart_parameters=self.restart_parameters,
            parity_set={},
            qts_subdivisions=qts_subdivisions,
            return_courant=self.compute_parameters.get("return_courant", False),
            cpu_pool=self.cpu_pool,
            waterbodies_df=self._network.waterbody_dataframe,
            waterbody_types_df=self._network.waterbody_types_dataframe,
            duplicate_ids_df=getattr(self._network, "_duplicate_ids_df", pd.DataFrame()),
            data_assimilation_parameters=self.data_assimilation_parameters,
            lastobs_df=self._data_assimilation.lastobs_df,
            link_gage_df=self._network.link_gage_df,
            link_lake_crosswalk=self._network.link_lake_crosswalk,
            nexus_dict=self._network.nexus_dict,
            poi_crosswalk=self._network.poi_nex_dict or {},
        )


    @property
    def nts(self) -> int:
        return self.forcing_parameters["nts"]

    @property
    def cpu_pool(self) -> int:
        return self.compute_parameters["cpu_pool"]

    @property
    def bmi_parameters(self) -> dict:
        return self._config.get("bmi_parameters", {})

    @property
    def log_parameters(self) -> dict:
        return self._config.get("log_parameters", {})

    @property
    def compute_parameters(self) -> dict:
        return self._config.get("compute_parameters", {})

    @property
    def network_topology_parameters(self) -> dict:
        return self._config.get("network_topology_parameters", {})

    @property
    def output_parameters(self) -> dict:
        return self._config.get("output_parameters", {})

    @property
    def preprocessing_parameters(self) -> dict:
        return self.network_topology_parameters.get("preprocessing_parameters", {})

    @property
    def waterbody_parameters(self) -> dict:
        return self.network_topology_parameters.get("waterbody_parameters", {})

    @property
    def supernetwork_parameters(self) -> dict:
        return self.network_topology_parameters.get("supernetwork_parameters", {})

    @property
    def forcing_parameters(self) -> dict:
        return self.compute_parameters.get("forcing_parameters", {})

    @property
    def restart_parameters(self) -> dict:
        return self.compute_parameters.get("restart_parameters", {})

    @property
    def hybrid_parameters(self) -> dict:
        return self.compute_parameters.get("hybrid_parameters", {})
    
    @property
    def data_assimilation_parameters(self) -> dict:
        return self.compute_parameters.get("data_assimilation_parameters", {})

    @property
    def parity_parameters(self) -> dict:
        return self.output_parameters.get("wrf_hydro_parity_check", {})

    @property
    def time(self) -> float:
        return self._time

    def _waterbodies_timesteps(self):
        """Sort the timestamps in the dataframe data dictionary, 
        then convert them to a format that can be read by the empty waterbodies parser 
        (YYYY-MM-DD hh:mm:ss)."""
        timestamps: list[str] = sorted(self._df_data)
        for i, ts in enumerate(timestamps):
            year = ts[:4]
            month = ts[4:6]
            day = ts[6:8]
            hour = ts[8:10]
            minute = ts[10:]
            timestamps[i] = f"{year}-{month}-{day} {hour}:{minute}:00"
        return timestamps

