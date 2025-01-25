import json
from typing import Dict

import geopandas as gpd
import pandas as pd
import pika
from pyogrio.errors import DataLayerError, DataSourceError

import nwm_routing.__main_._main_v04 as t_route
from troute_rnr.settings import Settings

settings = Settings()

def read_remote_gpkg(lid: int) -> Dict[str, pd.DataFrame]:
    try:
        gdf = {
            "divides": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="divides"),
            "nexus": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="nexus"),
            "flowpaths": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="flowpaths"),
            "network": gpd.read_file(f"{settings.S3_DOMAIN_URL}/domain_gpkgs/{lid}.gpkg",layer="network"),
        }
        try:
            gdf["flowpath-attributes"] = gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="flowpath-attributes")
        except DataLayerError:
            gdf["flowpath-attributes"] = gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="flowpath_attributes")
    except DataSourceError as e:
        print(str(e(f"Cannot find S3 domain data for {lid}")))

def write_forcast_csvs(gdf: Dict[str, pd.DataFrame]):
    # TODO call the NWPS API for forecasts from the LID, assign to top catchments, get downstream reach forecasts from others
    pass

def write_config(gdf: Dict[str, pd.DataFrame]):
    # TODO create the config required for T-Route
    pass


def run(
   ch: pika.channel.Channel,
   method: pika.spec.Basic.Deliver, 
   properties: pika.spec.BasicProperties,
   body: bytes
):
    body = json.loads(body.decode())
    lid = body["lid"]
    # TODO create an object for the input of the RnR data
    gdf = read_remote_gpkg(lid)
    write_forcast_csvs(gdf)
    restart_file = create_initial_start_file(params, settings)
    yaml_file_path = write_config(base_config, params, restart_file)
    t_route(["-f", yaml_file_path.__str__()])
    yaml_file_path.unlink()
    
    ch.basic_ack(delivery_tag=method.delivery_tag)


# s3://fim-services-data/replace-and-route/v0.2.0/domain_gpkgs/
