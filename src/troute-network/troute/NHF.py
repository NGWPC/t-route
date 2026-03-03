import time
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from .AbstractNetwork import AbstractNetwork
from troute.nhf_topology import (
    build_link_connections,
    get_terminal_nexus_ids,
    validate_connections,
)
from troute.nhf_discretize import (
    discretize_flowpaths,
    distribute_catchment_discharge,
)
from troute.nhf_preprocess import (
    NHFPreprocessMixin,
    read_geo_file,
    read_file,
)

__verbose__ = False
__showtiming__ = False


class NHF(NHFPreprocessMixin, AbstractNetwork):
    """ """

    __slots__ = [
        "_upstream_terminal",
        "_nexus_latlon",
        "_duplicate_ids_df",
        "_flow_scaling_segment_df",
        "_links_df",
        "_nodes_df",
        "_reference_flowpaths",
        "_fp_to_dn_nex",
        "_nex_to_dn_fp",
        "_upstream_inflow_df",
        "_nexus_virtual_seg_ids",
    ]

    def __init__(
        self,
        supernetwork_parameters,
        waterbody_parameters,
        data_assimilation_parameters,
        restart_parameters,
        compute_parameters,
        forcing_parameters,
        hybrid_parameters,
        preprocessing_parameters,
        output_parameters,
        verbose=False,
        showtiming=False,
        from_files=True,
        value_dict={},
        bmi_parameters={},
    ):
        """ """
        self.supernetwork_parameters = supernetwork_parameters
        self.waterbody_parameters = waterbody_parameters
        self.data_assimilation_parameters = data_assimilation_parameters
        self.restart_parameters = restart_parameters
        self.compute_parameters = compute_parameters
        self.forcing_parameters = forcing_parameters
        self.hybrid_parameters = hybrid_parameters
        self.preprocessing_parameters = preprocessing_parameters
        self.output_parameters = output_parameters
        self.verbose = verbose
        self.showtiming = showtiming

        if self.verbose:
            print("creating NHF supernetwork connections set")
        if self.showtiming:
            start_time = time.time()

        # ------------------------------------------------
        # Load hydrofabric information
        # ------------------------------------------------
        if self.preprocessing_parameters.get("use_preprocessed_data", False):
            raise NotImplementedError("Preprocessed data reads not implemented")
            # self.read_preprocessed_data()
        else:
            # FIXME: Temporary solution, from_files should only be from command line.
            # Update this once ngen framework is capable of providing this info via BMI.
            from_files_copy = from_files
            if not from_files_copy:
                from_files = True
            if from_files:
                nhf = read_geo_file(
                    self.supernetwork_parameters,
                    self.waterbody_parameters,
                    self.compute_parameters,
                    self.compute_parameters.get("cpu_pool", 1),
                )

                # Handle different key column names between flowpaths and flowpath_attributes
                flowpaths = nhf["flowpaths"]
                waterbodies = nhf["waterbodies"]
                gages = nhf["gages"]
                reference_flowpaths = nhf["reference_flowpaths"]
                virtual_flowpaths = nhf["virtual_flowpaths"]
                virtual_nexus = nhf["virtual_nexus"]
                nexus = nhf["nexus"]
                hydrolocations = nhf["hydrolocations"]
            else:
                raise NotImplementedError("BMI loading not implemented for the NHF")
                # flowpaths, lakes, network = load_bmi_data(
                #     value_dict,
                #     bmi_parameters,
                # )
            # FIXME: See FIXME above.
            if not from_files_copy:
                from_files = False

            # Preprocess network objects
            discretization_len = self.supernetwork_parameters.get("nhf_discretization_len", 300.0)
            self.preprocess_network(
                flowpaths, reference_flowpaths, virtual_flowpaths, virtual_nexus,
                nexus, discretization_len,
            )

            self.crosswalk_nex_flowpath_poi(
                virtual_flowpaths,
                hydrolocations,
                waterbodies,
                gages,
                reference_flowpaths,
            )

            # Preprocess waterbody objects
            self.preprocess_waterbodies(waterbodies, virtual_nexus)

            # Preprocess data assimilation objects #TODO: Move to DataAssimilation.py?
            self.preprocess_data_assimilation(
                flowpaths,
                reference_flowpaths,
                virtual_flowpaths,
                virtual_nexus,
                waterbodies,
                gages
            )


        if self.verbose:
            print("supernetwork connections set complete")
        if self.showtiming:
            print("... in %s seconds." % (time.time() - start_time))

        super().__init__(from_files, value_dict)

        # Create empty dataframe for coastal_boundary_depth_df. This way we can check if
        # it exists, and only read in SCHISM data during 'assemble_forcings' if it doesn't
        self._coastal_boundary_depth_df = pd.DataFrame()

    def extract_waterbody_connections(rows, target_col, waterbody_null=-9999):
        """Extract waterbody mapping from dataframe.
        TODO deprecate in favor of waterbody_connections property"""
        return rows.loc[rows[target_col] != waterbody_null, target_col].astype("int").to_dict()

    @property
    def downstream_flowpath_dict(self):
        return self._flowpath_dict

    @property
    def waterbody_connections(self):
        """
        A dictionary where the keys are the reach/segment id, and the
        value is the id to look up waterbody parameters
        """
        return self._waterbody_connections

    @property
    def gages(self):
        """
        FIXME
        """
        return self._gages

    @property
    def great_lakes_climatology_df(self):
        return pd.DataFrame()

    @property
    def waterbody_null(self):
        return np.nan  # pd.NA

    @property
    def segment_index(self):
        """
            Segment IDs of all reaches (links) in parameter dataframe
            and diffusive domain.
        """
        # list of all segments in the domain (MC + diffusive)
        self._segment_index = self._links_df.index
        if self._routing.diffusive_network_data:
            for tw in self._routing.diffusive_network_data:
                self._segment_index = self._segment_index.append(
                    pd.Index(self._routing.diffusive_network_data[tw]['mainstem_segs'])
                )
        return self._segment_index

    @property
    def links_df(self):
        return self._links_df


    def preprocess_network(
        self, flowpaths, reference_flowpaths, virtual_flowpaths, virtual_nexus,
        nexus=None, discretization_len_m=300.0,
    ):
        assert not virtual_flowpaths.empty, "No virtual flowpaths read to memory from .gpkg"
        if nexus is None:
            nexus = pd.DataFrame(columns=['nex_id', 'dn_fp_id'])

        # Store reference_flowpaths for use in build_qlateral_array
        self._reference_flowpaths = reference_flowpaths

        vfp_to_fp_map = reference_flowpaths[reference_flowpaths['virtual_fp_id'].notna()][
            ['virtual_fp_id', 'fp_id', 'div_id']
        ].copy()
        # NHF 1.1.2: VFP rows may have NULL fp_id; derive from div_id
        vfp_to_fp_map['fp_id'] = vfp_to_fp_map['fp_id'].fillna(vfp_to_fp_map['div_id'])
        _vfp = virtual_flowpaths.merge(
            vfp_to_fp_map,
            left_on='virtual_fp_id',
            right_on='virtual_fp_id',
            how='left'
        )
        result = _vfp.merge(
            flowpaths,
            left_on='fp_id',
            right_on='fp_id',
            how='left',
            suffixes=('', '_flowpath')  # Keep vfp columns as-is, suffix flowpath columns
        )
        cols_to_drop = [col for col in result.columns if col.endswith('_flowpath')]
        result = result.drop(columns=cols_to_drop)
        # Drop geometry columns carried from flowpath merge (not needed for VFP routing)
        if 'geometry' in result.columns:
            result = result.drop(columns=['geometry'])
        self._dataframe = result

        # make the flowpath linkage (kept for VFP qlat distribution compatibility)
        self._flowpath_dict = dict(zip(
            result.loc[:, 'dn_virtual_nex_id'],
            result.loc[:, 'virtual_fp_id']
        ))

        self._dataframe.set_index("virtual_fp_id", inplace=True)
        self._dataframe = self.dataframe.sort_index()

        # Discretize flowpaths into links and nodes
        self._links_df, self._nodes_df = discretize_flowpaths(
            flowpaths=flowpaths,
            virtual_flowpaths=virtual_flowpaths,
            virtual_nexus=virtual_nexus,
            reference_flowpaths=reference_flowpaths,
            nexus=nexus,
            discretization_len_m=discretization_len_m,
        )

        # Build link connections
        self._connections = build_link_connections(
            links_df=self._links_df,
            nexus=nexus,
        )

        # Store mappings for upstream inflow routing (used by distribute_catchment_discharge)
        self._fp_to_dn_nex = dict(zip(
            flowpaths['fp_id'].astype(int),
            flowpaths['dn_nex_id'].astype(int),
        ))
        self._nex_to_dn_fp = {}
        if not nexus.empty:
            valid_nex = nexus[nexus['dn_fp_id'].notna()]
            if not valid_nex.empty:
                self._nex_to_dn_fp = dict(zip(
                    valid_nex['nex_id'].astype(int),
                    valid_nex['dn_fp_id'].astype(int),
                ))

        # Initialize upstream inflow state (populated during forcing assembly)
        self._upstream_inflow_df = None
        self._nexus_virtual_seg_ids = {}

        # Validate link connections
        is_valid, orphaned = validate_connections(self._connections)
        if not is_valid:
            raise ValueError(
                f"Invalid link connections: {len(orphaned)} downstream IDs not found. "
                f"First 10: {list(orphaned)[:10]}"
            )

        # Build terminal codes from regular nexuses where dn_fp_id IS NULL (network outlets)
        if not nexus.empty:
            self._terminal_codes = set(
                nexus.loc[nexus['dn_fp_id'].isna(), 'nex_id'].astype(int)
            )
        else:
            self._terminal_codes = get_terminal_nexus_ids(virtual_nexus)

        # Build upstream terminal: links whose dn_node_id is in terminal_codes
        self._upstream_terminal = {}
        for nex_id in self._terminal_codes:
            terminal_links = self._links_df[
                self._links_df['dn_node_id'] == nex_id
            ].index.tolist()
            if terminal_links:
                self._upstream_terminal[nex_id] = set(terminal_links)

        # Store a dataframe containing info about nexus points. This will be reprojected to lat/lon
        # and filtered for only diffusive domain tailwaters in AbstractNetwork.py.
        # Location information will be used to advertise tailwater locations of diffusive domains
        # to the model engine/coastal models
        self._nexus_latlon = virtual_nexus

    def _create_upstream_virtual_segments(self, upstream_inflow_df):
        """
        Create virtual segments for upstream inflow injection via offnetwork_upstreams.

        For each target link that has nonzero upstream inflow, create a virtual
        segment that will carry the pre-filled flow-velocity-depth timeseries.
        Virtual segments are added to _connections (flowing into the target link),
        and to _links_df (with channel params inherited from the target link).

        On repeated calls (multi-loop runs), previous virtual segments are
        removed before creating new ones.

        Parameters
        ----------
        upstream_inflow_df : pd.DataFrame
            Upstream inflow indexed by target link_id, columns are timestamps.
        """
        # Find target links with nonzero flow
        nonzero_mask = upstream_inflow_df.abs().sum(axis=1) > 1e-10
        target_links = upstream_inflow_df.index[nonzero_mask]

        if target_links.empty:
            self._nexus_virtual_seg_ids = {}
            # Invalidate cached network properties
            self._reverse_network = None
            self._independent_networks = None
            self._reaches_by_tw = None
            return

        # Generate virtual segment IDs above existing ID space
        all_ids = list(self._links_df.index) + list(self._connections.keys())
        next_id = max(all_ids) + 1

        virtual_seg_ids = {}
        virtual_link_records = []

        for target_link_id in target_links:
            virtual_seg_id = next_id
            next_id += 1
            virtual_seg_ids[int(target_link_id)] = virtual_seg_id

            # Virtual segment flows into target link — this makes it
            # appear in rconn as upstream of the target link, which is
            # how compute.py detects offnetwork_upstreams.
            self._connections[virtual_seg_id] = [int(target_link_id)]

            # Build row with channel params inherited from target link
            target_row = self._links_df.loc[target_link_id]
            record = {
                'link_id': virtual_seg_id,
                'fp_id': target_row['fp_id'],
                'div_id': target_row.get('div_id', 0),
                'dn_node_id': target_row.get('dn_node_id', 0),
                'up_node_id': None,
                'length_km': target_row['length_km'],
            }
            # Copy channel params
            for col in self._links_df.columns:
                if col not in record:
                    record[col] = target_row[col]
            virtual_link_records.append(record)

        # Add virtual segment rows to links_df
        if virtual_link_records:
            virtual_df = pd.DataFrame(virtual_link_records).set_index('link_id')
            self._links_df = pd.concat([self._links_df, virtual_df])

        self._nexus_virtual_seg_ids = virtual_seg_ids

        # Add q0 entries for virtual segments (zeros; kernel pre-fills from FVD)
        if self._q0 is not None:
            vseg_q0 = pd.DataFrame(
                0.0,
                index=list(virtual_seg_ids.values()),
                columns=self._q0.columns,
                dtype="float32",
            )
            self._q0 = pd.concat([self._q0, vseg_q0])

        # Invalidate cached network properties so they recompute with virtual segments
        self._reverse_network = None
        self._independent_networks = None
        self._reaches_by_tw = None

    def build_flowveldepth_interorder(self, nts, qts_subdivisions):
        """
        Build flowveldepth_interorder dict for upstream inflow virtual segments.

        Each virtual segment gets a pre-filled flow-velocity-depth timeseries
        that the kernel injects as upstream boundary conditions (q_up).

        Parameters
        ----------
        nts : int
            Number of routing timesteps.
        qts_subdivisions : int
            Number of routing timesteps per qlat timestep.

        Returns
        -------
        dict
            {virtual_seg_id: {"results": [q0, 0.0, 0.0, q1, 0.0, 0.0, ...]}}
        """
        fvd_interorder = {}
        if self._upstream_inflow_df is None or not self._nexus_virtual_seg_ids:
            return fvd_interorder

        for target_link_id, virtual_seg_id in self._nexus_virtual_seg_ids.items():
            row = self._upstream_inflow_df.loc[target_link_id]
            n_qlat_steps = len(row)
            results = []
            for ts in range(nts):
                qlat_idx = min(ts // qts_subdivisions, n_qlat_steps - 1)
                q = float(row.iloc[qlat_idx])
                results.extend([q, 0.0, 0.0])  # flow, velocity=0, depth=0
            fvd_interorder[virtual_seg_id] = {"results": results}

        return fvd_interorder

    def build_qlateral_array(
        self,
        run,
    ):
        # TODO: set default/optional arguments
        qts_subdivisions = run.get("qts_subdivisions", 1)
        nts = run.get("nts", 1)
        qlat_input_folder = run.get("qlat_input_folder", None)

        if qlat_input_folder:
            qlat_input_folder = Path(qlat_input_folder)
            if "qlat_files" in run:
                qlat_files = run.get("qlat_files")
                qlat_files = [qlat_input_folder.joinpath(f) for f in qlat_files]
            elif "qlat_file_pattern_filter" in run:
                qlat_file_pattern_filter = run.get("qlat_file_pattern_filter", "*CHRT_OUT*")
                qlat_files = sorted(qlat_input_folder.glob(qlat_file_pattern_filter))

            dfs = []

            # FIXME Temporary solution to allow t-route to use ngen nex-* output files as forcing files
            # This capability should be here, but we need to think through how to handle all of this
            # data in memory for large domains and many timesteps... - shorvath, Feb 28, 2024
            qlat_file_pattern_filter = self.forcing_parameters.get("qlat_file_pattern_filter", None)
            if qlat_file_pattern_filter == "nex-*":
                raise NotImplementedError("Nex-output not implemented!")
            else:
                for f in qlat_files:
                    df = read_file(f)
                    df["feature_id"] = df["feature_id"].map(
                        lambda x: int(str(x).removeprefix("nex-")) if str(x).startswith("nex") else int(x)
                    )
                    assert df["feature_id"].is_unique, (
                        f"'feature_id's must be unique. '{f!s}' contains duplicate 'feature_id's: {pformat(df.loc[df['feature_id'].duplicated(), 'feature_id'].to_list())}"
                    )
                    df = df.set_index("feature_id")
                    dfs.append(df)

                # lateral flows [m^3/s] indexed by div_id (divide/catchment)
                div_lateralflows_df = pd.concat(dfs, axis=1)

                # Clean up virtual segments from previous loop before building new mappings
                if self._nexus_virtual_seg_ids:
                    old_vseg_ids = set(self._nexus_virtual_seg_ids.values())
                    self._links_df = self._links_df.drop(
                        index=[i for i in old_vseg_ids if i in self._links_df.index]
                    )
                    if self._q0 is not None:
                        self._q0 = self._q0.drop(
                            index=[i for i in old_vseg_ids if i in self._q0.index]
                        )
                    for vseg_id in old_vseg_ids:
                        self._connections.pop(vseg_id, None)
                    self._nexus_virtual_seg_ids = {}

                # Distribute catchment discharge to links and upstream inflow
                qlats_df, self._flow_scaling_segment_df, self._upstream_inflow_df = \
                    distribute_catchment_discharge(
                        div_lateralflows_df,
                        self._dataframe,
                        self._links_df,
                        self._nodes_df,
                        self._reference_flowpaths,
                        self._fp_to_dn_nex,
                        self._nex_to_dn_fp,
                    )

                # Create virtual segments for upstream inflow injection
                self._create_upstream_virtual_segments(self._upstream_inflow_df)
        else:
            raise ValueError("qlat_input_folder does not exist")
        all_df = pd.DataFrame(
            np.zeros((len(self.segment_index), len(qlats_df.columns))),
                index=self.segment_index,
                columns=qlats_df.columns,
        )
        all_df.loc[qlats_df.index] = qlats_df
        qlats_df = all_df.sort_index()

        # column filtering
        max_col = 1 + nts // qts_subdivisions
        if len(qlats_df.columns) > max_col:
            qlats_df.drop(qlats_df.columns[max_col:], axis=1, inplace=True)

        # final filter to segment_index
        if not self.segment_index.empty:
            qlats_df = qlats_df[qlats_df.index.isin(self.segment_index)]

        self._qlateral = qlats_df



    def build_et_array(
        self,
        run,
    ):
        col_idx = run.get("et_index_name", "divide_id")
        var_idx = run.get("et_var_name", "ACTUAL_ET")
        try:
            ds = run["et_forcing_ds"]
        except KeyError as e:
            raise KeyError("Cannot find et_forcing_ds in runs") from e
        ds_AET = ds[var_idx]

        # mapping catchments to flowpath IDs
        mapping_dict = dict(zip(
            self._dataframe['divide_id'].values,
            self._dataframe.index.values
        ))
        keys = np.array([mapping_dict[key] for key in ds_AET[col_idx].values])

        time_strings = pd.to_datetime(ds_AET.time.values).strftime('%Y%m%d%H%M')
        aet_df = pd.DataFrame(
            data=ds_AET.values,
            index=keys,
            columns=time_strings
        )

        aet_df.index.name = 'key'
        ordered_aet_df = aet_df.reindex(self._dataframe.index, fill_value=0) # ordering based on the existing

        # Convert ET into ELOSS
        try:
            A_w = self._dataframe["tw"] * self._dataframe["dx"]
            _E = ordered_aet_df * self.forcing_parameters["peadj"]
            TIMINT = 1 # Hardcoding for hourly
            # _E is in mm/hr. Thus, MM/HR × (1/1000) × (1/3600) -> m/s
            ELOSS_cms = (_E / 1000 / 3600 / TIMINT).mul(A_w.values, axis=0)
            ELOSS_cfs = ELOSS_cms * 35.3147  # since NGEN runs in cfs, converting from cms to cfs. Can make a config setting later.
        except KeyError as e:
            raise KeyError("Cannot find flowpath attributes to map PET. Can you ensure ") from e
        self._eloss = ELOSS_cfs

    def write_preprocessed_data(
        self,
    ):
        # LOG.debug("saving preprocessed network data to disk for future use")
        # todo: consider a better default than None
        destination_folder = self.preprocessing_parameters.get("preprocess_output_folder", None)
        if destination_folder:
            output_filename = self.preprocessing_parameters.get("preprocess_output_filename", "preprocess_output")

        outputs = {
            "dataframe": self.dataframe,
            "flowpath_dict": self._flowpath_dict,
            "terminal_codes": self._terminal_codes,
            "upstream_termincal": self._upstream_terminal,
            "connections": self._connections,
            "waterbody_df": self._waterbody_df,
            "waterbody_types_df": self._waterbody_types_df,
            "waterbody_connections": self._waterbody_connections,
            "waterbody_type_specified": self._waterbody_type_specified,
            "link_lake_crosswalk": self._link_lake_crosswalk,
            "gages": self._gages,
            "usgs_lake_gage_crosswalk": self._usgs_lake_gage_crosswalk,
            "usace_lake_gage_crosswalk": self._usace_lake_gage_crosswalk,
            "rfc_lake_gage_crosswalk": self._rfc_lake_gage_crosswalk,
        }
        np.save(Path(destination_folder).joinpath(output_filename), outputs)

    def read_preprocessed_data(
        self,
    ):
        preprocess_filepath = self.preprocessing_parameters.get("preprocess_source_file", None)
        if preprocess_filepath:
            try:
                inputs = np.load(Path(preprocess_filepath), allow_pickle="TRUE").item()
            except:
                # LOG.critical('Canonot find %s' % Path(preprocess_filepath))
                quit()

            self._dataframe = inputs.get("dataframe", None)
            self._flowpath_dict = inputs.get("flowpath_dict", None)
            self._terminal_codes = inputs.get("terminal_codes", None)
            self._upstream_terminal = inputs.get("upstream_termincal", None)
            self._connections = inputs.get("connections", None)
            self._waterbody_df = inputs.get("waterbody_df", None)
            self._waterbody_types_df = inputs.get("waterbody_types_df", None)
            self._waterbody_connections = inputs.get("waterbody_connections", None)
            self._waterbody_type_specified = inputs.get("waterbody_type_specified", None)
            self._link_lake_crosswalk = inputs.get("link_lake_crosswalk", None)
            self._gages = inputs.get("gages", None)
            self._usgs_lake_gage_crosswalk = inputs.get("usgs_lake_gage_crosswalk", None)
            self._usace_lake_gage_crosswalk = inputs.get("usace_lake_gage_crosswalk", None)
            self._usbr_lake_gage_crosswalk = inputs.get("usbr_lake_gage_crosswalk", None)
            self._rfc_lake_gage_crosswalk = inputs.get("rfc_lake_gage_crosswalk", None)


# Re-exports for backward compatibility
from troute.nhf_topology import (
    build_downstream_connections,
    build_upstream_terminal,
    find_headwaters,
    find_tailwaters,
    get_terminal_nexus_ids,
    validate_connections,
)
from troute.nhf_discretize import distribute_catchment_discharge
