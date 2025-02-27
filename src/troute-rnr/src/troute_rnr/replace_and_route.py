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


# s3://fim-services-data/replace-and-route/v0.2.0/domain_gpkgs/
