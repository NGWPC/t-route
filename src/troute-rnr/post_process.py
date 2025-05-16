"""A function to format t-route outputs into the correct format for water.noaa.gov"""

from datetime import datetime

import xarray as xr
from icefabric_tools import table_to_geopandas
from troute_rnr.settings import Settings


def post_process(settings: Settings) -> None:
    """A function to post-process the T-Route outputs from RnR

    Parameter
    ---------
    settings: Settings
        The global RnR settings
    """
    files = []
    print("Opening all datasets...")
    for file in settings.output_files_path.glob("*"):
        for ds in file.glob("*"):
            files.append(xr.open_dataset(ds, engine="netcdf4"))
    ds = xr.concat(files, dim="feature_id")

    # Find max flows and their time indices
    max_flows = ds.flow.max(dim="time")
    max_flow_times = ds.flow.idxmax(dim="time")
    catchments = [f"wb-{_id}" for _id in ds.feature_id.values]

    flowpaths = table_to_geopandas(settings.catalog.load_table("hydrofabric.flowpaths"))
    flowpaths = flowpaths.set_index("id")

    filtered_flowpaths = flowpaths.loc[flowpaths.index.isin(catchments)].copy()
    flow_dict = dict(zip([f"wb-{id_}" for id_ in ds.feature_id.values], max_flows.values * 35.3147))  # to cfs

    time_dict = dict(zip([f"wb-{id_}" for id_ in ds.feature_id.values], max_flow_times.values.astype(str)))
    filtered_flowpaths["streamflow_cfs"] = filtered_flowpaths.index.map(flow_dict)
    filtered_flowpaths["max_flow_time"] = filtered_flowpaths.index.map(time_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_inundation_{timestamp}.csv"

    output_path = settings.output_files_path / output_filename
    filtered_flowpaths.to_csv(output_path)
    print(f"Processing complete! Results saved to: {output_path}")


if __name__ == "__main__":
    post_process(Settings())
