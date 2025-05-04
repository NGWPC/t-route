"""Module for handling NWM data processing and NWPS integrations."""

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import lxml.etree
import numpy as np
import pandas as pd
import yaml
from icefabric_tools import find_origin, rnr
from nwm_routing.__main__ import main_v04 as t_route

from troute_rnr import write
from troute_rnr.schemas.nwps import ProcessedData
from troute_rnr.schemas.weather import Site
from troute_rnr.settings import Settings


def edit_yaml(original_file: Path, params: dict[str, str], restart_file: Path) -> Path:
    """A function to dynamically edit the T-Route config

    Parameters
    ----------
    original_file: Path
        The path to the base yaml config file
    params: Dict[str, str]
        The parameters that will be added to the base config file
    restart_file: path
        The location to the restart_file path

    Returns
    -------
    Path:
        The path to the dynamically generated config
    """
    tmp_yaml = original_file.with_name(f"tmp_{params['lid']}_{original_file.suffix}")
    with open(original_file) as file:
        data = yaml.safe_load(file)

    output_dir = params["output_folder"] / params["lid"]
    output_dir.mkdir(exist_ok=True)

    data["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"] = str(
        params["geo_file_path"]
    )

    data["compute_parameters"]["restart_parameters"]["start_datetime"] = params["start_datetime"]
    data["compute_parameters"]["restart_parameters"]["lite_channel_restart_file"] = str(restart_file)
    data["compute_parameters"]["forcing_parameters"]["nts"] = params["nts"]
    data["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = str(params["qlat_input_folder"])

    data["output_parameters"]["stream_output"]["stream_output_directory"] = str(output_dir)

    with open(tmp_yaml, "w") as file:
        yaml.dump(data, file)

    return tmp_yaml


def create_initial_start_file(params: dict[str, str], settings: Settings) -> Path:
    """Creating the initial start/restart files

    Parmeters
    ---------
    params: Dict[str, str]
        The parameters from the API to be added to the t-route config file
    settings: Settings
        The T-route BaseSettings

    Returns
    -------
    Path:
        The path to the t-route restart file
    """
    start_datetime = datetime.strptime(params["start_datetime"], "%Y-%m-%d_%H:%M")
    formatted_datetime = start_datetime.strftime("%Y-%m-%d_%H:%M")

    gdf = gpd.read_file(params["geo_file_path"], layer="network")
    mask = gdf["divide_id"].isna()
    keys = [int(val.split("-")[1]) for val in set(gdf[~mask]["divide_id"].values.tolist())]

    discharge_upstream = np.zeros([len(keys)])
    discharge_downstream = np.zeros([len(keys)])
    height = np.zeros([len(keys)])
    idx = keys.index(int(params["hy_id"]))
    discharge_upstream[idx] = float(params["initial_start"])

    time_array = np.array([pd.to_datetime(formatted_datetime, format="%Y-%m-%d_%H:%M")] * len(keys))

    df = pd.DataFrame(
        {
            "time": time_array,
            "key": np.array(keys),
            "qu0": discharge_upstream,
            "qd0": discharge_downstream,
            "h0": height,
        }
    )
    df.set_index("key", inplace=True)
    restart_full_path = settings.restart_path / f"{params['lid']}_{formatted_datetime}.pkl"
    df.to_pickle(restart_full_path)
    return restart_full_path


def format_config(inputs: ProcessedData, settings: Settings) -> None:
    """
    Create the configuration required for T-Route.

    Parameters
    ----------
    inputs: SiteData
        The site information, and forecasts for each RnR Reach
    settings: Settings
        The site information, and forecasts for each RnR Reach

    Returns
    -------
    None
    """
    reach = inputs.reach
    rnr.get_rnr_segment(settings.catalog, reach.id, settings.tmp_geopackage)
    network = settings.catalog.load_table("hydrofabric.network")
    hy_id = (
        find_origin(network_table=network, identifier=reach.id, id_type="comid")["id"].values[0].split("-")[1]
    )
    tmp_flow_files_path = settings.tmp_flow_files_path / inputs.lid
    tmp_flow_files_path.mkdir(exist_ok=True)
    write.write_flow_files(hy_id, reach, tmp_flow_files_path)

    start_timestamp = reach.times[0].strftime("%Y-%m-%d_%H:%M")
    time_diff = reach.times[-1] - reach.times[0]

    # Get the number of hours
    num_hours = time_diff.total_seconds() / 3600
    nts = 288 * int(num_hours / 24)  # 288 = 24 hours × 12 (5-minute intervals per hour)

    if reach.latest_observation is not None:
        initial_start = reach.latest_observation
    else:
        initial_start = reach.secondary_forecast[0]  # Using t0 as initial start since no obs

    params = {
        "lid": inputs.lid,
        "hy_id": hy_id,
        "initial_start": initial_start,
        "start_datetime": start_timestamp,
        "geo_file_path": settings.tmp_geopackage,
        "nts": nts,
        "qlat_input_folder": tmp_flow_files_path,
        "output_folder": settings.output_files_path,
    }
    restart_file = create_initial_start_file(params, settings)
    yaml_file_path = edit_yaml(settings.base_config_path, params, restart_file)

    t_route(["-f", str(yaml_file_path)])

    yaml_file_path.unlink()
    settings.tmp_geopackage.unlink()
    tmp_flow_files_path.unlink()


def format_xml(product_text: str) -> list[Site]:
    """
    Format product text from HML into valid XML segments.

    Parameters
    ----------
    product_text : str
        Product text in HML format.
    settings : Settings
        Configuration settings.

    Returns
    -------
    List[Site]
        List of Site objects extracted from the XML.
    """
    xml_split = product_text.split("?xml")
    sites = []
    # Ignore the first idx since it's never valid XML
    for i in range(1, len(xml_split)):
        xml_segment = "<?xml" + xml_split[i][:-2]  # Adding removed XML tag, and removed trailing tags
        try:
            site = Site.from_xml(xml_segment)
        except lxml.etree.XMLSyntaxError:
            # Removing extra content at end of document
            xml_segment = xml_segment.split("</site>")[0] + "</site>"
            site = Site.from_xml(xml_segment)
        sites.append(site)
    return sites
