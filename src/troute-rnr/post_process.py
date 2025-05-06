"""A function to format t-route outputs into the correct format for OWP"""

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
    print("opening all datasets")
    for file in settings.output_files_path.glob("*"):
        for ds in file.glob("*"):
            files.append(xr.open_dataset(ds, engine="netcdf4"))
    ds = xr.concat(files, dim="feature_id")
    catchments = [f"wb-{_id}" for _id in ds.feature_id.values]
    flowpaths = table_to_geopandas(settings.catalog.load_table("hydrofabric.flowpaths"))
    flowpaths = flowpaths.set_index("id")
    filtered_flowpaths = flowpaths.loc[flowpaths.index.isin(catchments)]
    flow_dict = dict(
        zip([f"wb-{id_}" for id_ in ds.feature_id.values], ds.flow.isel(time=0).values * 35.3147)
    )  # to cfs
    filtered_flowpaths["streamflow_cfs"] = filtered_flowpaths.index.map(flow_dict)
    filtered_flowpaths.to_csv(settings.output_files_path / "output_inundation.csv")


if __name__ == "__main__":
    post_process(Settings())
