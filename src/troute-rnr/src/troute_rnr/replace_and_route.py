import json
from typing import Dict, List

import geopandas as gpd
import httpx
import lxml
import pandas as pd
import pika
from pydantic.error_wrappers import ValidationError
from pyogrio.errors import DataLayerError, DataSourceError

from nwm_routing.__main__ import main_v04 as t_route
from troute_rnr.format import format_xml, pull_nwm_inputs, write_forcast_csvs, write_config
from troute_rnr.settings import Settings
from troute_rnr.utils import get

settings = Settings()

headers = {
    'Accept': 'application/ld+json',
    'User-Agent': '(water.noaa.gov, Tadd.N.Bindas@rtx.com)'
}

def read_remote_gpkg(lid: int) -> Dict[str, pd.DataFrame]:
    try:
        gdf = {
            "divides": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="divides"),
            "nexus": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="nexus"),
            "flowpaths": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="flowpaths"),
            "network": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="network"),
            "lakes": gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="lakes"),
        }
        try:
            gdf["flowpath-attributes"] = gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="flowpath-attributes")
        except DataLayerError:
            gdf["flowpath-attributes"] = gpd.read_file(f"{settings.S3_DOMAIN_URL}/{lid}.gpkg",layer="flowpath_attributes")
    except DataSourceError as e:
        print(str(e(f"Cannot find S3 domain data for {lid}")))
    return gdf


def run(
   ch: pika.channel.Channel,
   method: pika.spec.Basic.Deliver, 
   properties: pika.spec.BasicProperties,
   body: bytes
):
    hml = json.loads(body.decode())
    print(f"Reading forecast for {hml['rdf']}, issued at {hml['issuance_time']}")
    site_data = get(hml["rdf"], headers=headers).json()
    forecasts = format_xml(site_data["productText"])
    if len(forecasts) == 0:
        # There is no forecast present in this message. End the process
        ch.basic_ack(delivery_tag=method.delivery_tag)
    else:
        for forecast in forecasts:
            gdf = read_remote_gpkg(forecast.lid)
            inputs = pull_nwm_inputs(forecast, gdf)
            write_forcast_csvs(gdf, inputs)
            restart_file = create_initial_start_file(params, settings)
            yaml_file_path = write_config(base_config, params, restart_file)
            t_route(["-f", yaml_file_path.__str__()])
            yaml_file_path.unlink()
            
        ch.basic_ack(delivery_tag=method.delivery_tag)

# s3://fim-services-data/replace-and-route/v0.2.0/domain_gpkgs/
