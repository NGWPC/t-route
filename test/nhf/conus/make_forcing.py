"""Generate channel forcing files for a CONUS test case using retrospective data, and create a config YAML for running the test case."""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
import yaml


def create_forcing_dataset(t_start: str, t_end: str, out_dir: str, hydrofabric_path: str, retrospective_path: str, forcing_file_pattern: str = "CHRTOUT_DOMAIN1"):
    """Create a dataset of channel forcing files from retrospective data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    crosswalk = gpd.read_file(hydrofabric_path, layer="reference_flowpaths")
    fps = gpd.read_file(hydrofabric_path, layer="flowpaths", ignore_geometry=True)
    retro = xr.open_zarr(retrospective_path, storage_options={"anon": True},)

    # Post-process
    feature_ids_retro = crosswalk["ref_fp_id"].values
    crosswalk = pd.merge(crosswalk[["ref_fp_id", "div_id"]], fps[["fp_id", "div_id"]], left_on="div_id", right_on="div_id", how="left")

    # Generate dataset
    iterator = pd.date_range(
        start=t_start,
        end=t_end,
        freq="H"
    )
    for i in iterator:
        print(f"Processing time step {i}...")
        qlat = retro.sel(feature_id=feature_ids_retro, time=i)["q_lateral"].reset_coords(drop=True)
        t_str = i.strftime("%Y%m%d%H%M")
        df = qlat.to_dataframe()
        df = pd.merge(df, crosswalk[["ref_fp_id", "fp_id"]], left_index=True, right_on="ref_fp_id", how="left").rename(columns={"fp_id": "feature_id", "q_lateral": t_str})[["feature_id", t_str]]
        df = df.groupby("feature_id").sum().reset_index()
        df.to_csv(out_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)

def make_config_yaml(config_path: str, hydrofabric_path: str, qlat_input_folder: str, nts: int, restart_time: str, file_pattern_filter: str = "*.CHRTOUT_DOMAIN1.csv"):
    """Create a config YAML for running the test case."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "log_parameters": {
            "showtiming": True,
            "log_level": "DEBUG",
        },
        "network_topology_parameters": {
            "supernetwork_parameters": {
                "geo_file_path": hydrofabric_path,
            },
            "waterbody_parameters": {
                "break_network_at_waterbodies": True,
            },
        },
        "compute_parameters": {
            "parallel_compute_method": "by-subnetwork-jit-clustered",
            "compute_kernel": "V02-structured",
            "assume_short_ts": True,
            "subnetwork_target_size": 10000,
            "cpu_pool": 1,
            "restart_parameters": {
                "start_datetime": restart_time,
            },
            "forcing_parameters": {
                "qts_subdivisions": 12,
                "dt": 300,
                "qlat_input_folder": qlat_input_folder,
                "qlat_file_pattern_filter": file_pattern_filter,
                "nts": nts,
                "max_loop_size": 288,
            },
            "data_assimilation_parameters": {
                "streamflow_da": {
                    "streamflow_nudging": False,
                    "diffusive_streamflow_nudging": False,
                },
                "reservoir_da": {
                    "reservoir_persistence_da": {
                        "reservoir_persistence_usgs": False,
                    },
                    "reservoir_rfc_da": {
                        "reservoir_rfc_forecasts": False,
                    },
                },
            },
        },
        "output_parameters": {
            "stream_output": {
                "stream_output_directory": "output/",
                "stream_output_time": -1,
                "stream_output_type": ".nc",
                "stream_output_internal_frequency": 60,
            },
        },
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)


def main():
    """Generate forcing files and config yaml for July 2022 floods."""
    start_time = "2022-07-25"
    end_time = "2022-07-31"
    out_path = Path(__file__).parent / "channel_forcing"
    hf_file = "nhf_0.4.1.dev6+g0cf24dcd2.gpkg"
    hf_path = Path(__file__).parent / "domain" / hf_file
    retro_path = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"

    create_forcing_dataset(
        t_start=start_time,
        t_end=end_time,
        out_dir=out_path,
        hydrofabric_path=hf_path,
        retrospective_path=retro_path,
    )

    nts = int((pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds() / 300) + 1  # Assuming dt=300s
    make_config_yaml(
        config_path=Path(__file__).parent / "test_case.yaml",
        hydrofabric_path=f"domain/{hf_file}",
        qlat_input_folder="channel_forcing/",
        nts=nts,
        restart_time=start_time,
    )

if __name__ == "__main__":
    main()
