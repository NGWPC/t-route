"""A function to format t-route outputs into the correct format for water.noaa.gov"""

import re
from datetime import datetime, timedelta

import xarray as xr
from icefabric_tools import table_to_geopandas
from troute_rnr.settings import Settings


def extract_timestamp_from_filename(filename: str) -> datetime | None:
    """Extract timestamp from filename like 'troute_output_202505061230.nc'

    Parameters
    ----------
    filename : str
        The filename to extract timestamp from (format: troute_output_YYYYMMDDHHMM.nc)

    Returns
    -------
    datetime | None
        The extracted datetime, or None if parsing fails
    """
    # Pattern to match troute_output_*.nc and extract the timestamp
    pattern = r"troute_output_(\d{12})\.nc"

    match = re.search(pattern, filename)
    if not match:
        return None
    timestamp_str = match.group(1)

    try:
        # Parse YYYYMMDDHHMM format
        year = int(timestamp_str[:4])
        month = int(timestamp_str[4:6])
        day = int(timestamp_str[6:8])
        hour = int(timestamp_str[8:10])
        minute = int(timestamp_str[10:12])
        return datetime(year, month, day, hour, minute)
    except (ValueError, IndexError) as e:
        print(f"Error parsing timestamp from {filename}: {e}")
        return None


def post_process(settings: Settings) -> None:
    """A function to post-process the T-Route outputs from RnR

    Parameter
    ---------
    settings: Settings
        The global RnR settings
    """
    files = []
    current_time = datetime.now()
    twenty_four_hours_ago = current_time - timedelta(hours=24)
    print("Opening all forecasts for times after the current timestep")
    for folder in settings.output_files_path.glob("*"):
        if folder.is_dir():
            for nc_file in folder.glob("*.nc"):
                file_timestamp = extract_timestamp_from_filename(nc_file.name)
                # Filter files created within the last 24 hours
                if file_timestamp and twenty_four_hours_ago <= file_timestamp <= current_time:
                    _ds = xr.open_dataset(nc_file, engine="netcdf4")
                    files.append(_ds)

    if not files:
        print("No files found within the last 24 hours")
        return

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
