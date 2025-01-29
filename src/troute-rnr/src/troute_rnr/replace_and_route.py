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
from troute_rnr.schemas.nwps import GaugeData
from troute_rnr.schemas.weather import Site
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

def format_xml(product_text: str) -> List[GaugeData]:
    """A function to format the product text from HML into valid XML segments
    """
    xml_split = product_text.split("?xml")
    forecasts = []

    # ignore the first one since it's not valid XML
    for i in range(1, len(xml_split)):
        xml_segment = "<?xml" + xml_split[i][:-2]  # adding removed XML tag, and removed trailing tags
        try:
            site = Site.from_xml(xml_segment)
        except lxml.etree.XMLSyntaxError:
            xml_segment = xml_segment.split("</site>")[0] + "</site>"  # Removing extra content at end of document
            site = Site.from_xml(xml_segment)
        endpoint = f"{settings.BASE_URL}/gauges/{site.properties['id']}"
        try:
            forecast = get(endpoint).json()
            try:
                gauge_data = GaugeData(**forecast)
            except ValidationError:
                # There was no forecast/record for the site given
                continue
            if gauge_data.ForecastFloodCategory in settings.STAGES:
                forecasts.append(gauge_data)
        except httpx.HTTPStatusError:
            # There was no forecast/record for the site given
            # print(f"{endpoint} hit 404 error: {str(e)}")
            continue
            
    return forecasts 


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
        gdf = read_remote_gpkg(lid)
        write_forcast_csvs(gdf)
        restart_file = create_initial_start_file(params, settings)
        yaml_file_path = write_config(base_config, params, restart_file)
        t_route(["-f", yaml_file_path.__str__()])
        yaml_file_path.unlink()
        
        ch.basic_ack(delivery_tag=method.delivery_tag)

# s3://fim-services-data/replace-and-route/v0.2.0/domain_gpkgs/
