"""Generate channel forcing files for a CONUS test case using retrospective data, and create a config YAML for running the test case."""
import argparse
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import xarray as xr
import yaml
from dataretrieval import nwis

### NEED TO INSTALL
# pip install xarray[zarr] zarr fsspec s3fs

RETRO_PATH = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
RETROSPECTIVE_LATERAL_FIELD = "q_lateral"
RETROSPECTIVE_FLOW_FIELD = "streamflow"

def create_forcing_dataset(t_start: str, t_end: str, out_dir: str, hydrofabric_path: str, retrospective_path: str, forcing_file_pattern: str = "CHRTOUT_DOMAIN1", generate_reference_data: bool = False, reference_dir: Union[str, None] = None):
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
        qlat = retro.sel(feature_id=feature_ids_retro, time=i)[RETROSPECTIVE_LATERAL_FIELD].reset_coords(drop=True)
        t_str = i.strftime("%Y%m%d%H%M")
        df = qlat.to_dataframe()
        df = pd.merge(df, crosswalk[["ref_fp_id", "fp_id"]], left_index=True, right_on="ref_fp_id", how="left").rename(columns={"fp_id": "feature_id", RETROSPECTIVE_LATERAL_FIELD: t_str})[["feature_id", t_str]]
        df = df.groupby("feature_id").sum().reset_index()
        df.to_csv(out_dir / f"{t_str}.{forcing_file_pattern}.csv", index=False)

    # Generate reference outputs if requested
    if generate_reference_data and reference_dir is not None:
        gages = gpd.read_file(hydrofabric_path, sql="SELECT * FROM gages WHERE status = 'USGS-active'", ignore_geometry=True)
        for _, gage in gages.iterrows():

            # Load retrospective flow for the gage's reference flowpath
            fp_id = int(gage["fp_id"])
            ref_fp_id = crosswalk.loc[crosswalk["fp_id"] == fp_id, "ref_fp_id"].values[0]
            retro_q = retro.sel(feature_id=ref_fp_id, time=slice(t_start, t_end))[RETROSPECTIVE_FLOW_FIELD].reset_coords(drop=True).to_dataframe()
            retro_q.index = retro_q.index.tz_localize("UTC")

            # Load USGS data, if available
            gage_id = gage["site_no"]
            usgs_q = nwis.get_iv(site=gage_id, start=t_start, end=t_end, parameterCd="00060")[0].rename(columns={"00060": "usgs_q"})
            usgs_q.index = pd.to_datetime(usgs_q.index)
            usgs_q = usgs_q.reindex(retro_q.index, method="nearest", tolerance=pd.Timedelta("15min"))

            # Combine and save
            df = pd.DataFrame({
                "time": retro_q.index,
                "retrospective_q": retro_q[RETROSPECTIVE_FLOW_FIELD].values,
                "usgs_q": usgs_q["usgs_q"].values,
            }).set_index("time")
            df.to_xarray().to_netcdf(Path(reference_dir) / f"gage_at_{fp_id}_reference.nc")

def make_config_yaml(config_path: str, hydrofabric_path: str, qlat_input_folder: str, nts: int, restart_time: str, file_pattern_filter: str = "*.CHRTOUT_DOMAIN1.csv", max_loop_size: int = 288):
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
                "dt": 300,
                "qlat_input_folder": qlat_input_folder,
                "qlat_file_pattern_filter": file_pattern_filter,
                "nts": nts,
                "max_loop_size": max_loop_size,
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

def build_forcing_dataset(start_time: str, end_time: str, case_id: str, run_id: str, hf_file: str, generate_reference_data: bool = False):
    """Build the forcing dataset and config YAML for a given case and run."""
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    run_dir = Path(__file__).parent / case_id
    forcing_subdir = f"channel_forcing_{run_id}" if run_id else "channel_forcing"
    config_path = run_dir / f"test_case_{run_id}.yaml" if run_id else run_dir / "test_case.yaml"
    out_path = run_dir / forcing_subdir
    hf_path = run_dir / "domain" / hf_file
    if generate_reference_data:
        reference_dir = run_dir / f"reference_outputs_{run_id}" if run_id else run_dir / "reference_outputs"
        reference_dir.mkdir(parents=True, exist_ok=True)
    else:
        reference_dir = None

    create_forcing_dataset(
        t_start=start_dt,
        t_end=end_dt,
        out_dir=out_path,
        hydrofabric_path=hf_path,
        retrospective_path=RETRO_PATH,
        generate_reference_data=generate_reference_data,
        reference_dir=reference_dir,
    )

    nts = int((end_dt - start_dt).total_seconds() / 300) + 1  # Assuming dt=300s
    make_config_yaml(
        config_path=config_path,
        hydrofabric_path=f"domain/{hf_file}",
        qlat_input_folder=f"{forcing_subdir}/",
        nts=nts,
        restart_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
    )

def conecuh_retro():
    """Generate forcing files and config yaml for December 2009 floods."""
    start_time = "2009-12-12"
    end_time = "2009-12-29"
    case_id = "conecuh_case"
    run_id = "retro"
    hf_file =  "02374250.gpkg"

    build_forcing_dataset(
        start_time=start_time,
        end_time=end_time,
        case_id=case_id,
        run_id=run_id,
        hf_file=hf_file,
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate forcing dataset and config YAML for a case."
    )

    parser.add_argument(
        "--start-time",
        default="2009-12-12 00:00",
        help="Simulation start time (e.g. '2009-12-12' or '2009-12-12 06:00')",
    )

    parser.add_argument(
        "--end-time",
        default="2009-12-29 00:00",
        help="Simulation end time (e.g. '2009-12-29' or '2009-12-29 12:00')",
    )

    parser.add_argument(
        "--case-id",
        default="conecuh_case",
        help="Case directory name",
    )

    parser.add_argument(
        "--hf-file",
        default="02374250.gpkg",
        help="Hydrofabric file inside domain directory",
    )

    parser.add_argument(
        "--run-id",
        default="retro",
        help="Run identifier.  There can be multiple runs per case.",
    )

    parser.add_argument(
        "--generate-reference-data",
        default=True,
        help="Generate reference data (USGS and retrospective outputs) for testing.",
    )

    args = parser.parse_args()

    build_forcing_dataset(
        start_time=args.start_time,
        end_time=args.end_time,
        case_id=args.case_id,
        run_id=args.run_id,
        hf_file=args.hf_file,
        generate_reference_data=args.generate_reference_data,
    )


if __name__ == "__main__":
    main()
