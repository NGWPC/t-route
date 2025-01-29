import json
from typing import Dict, List

import geopandas as gpd
import httpx
import lxml
import pandas as pd
from pydantic.error_wrappers import ValidationError

from troute_rnr.schemas.nwps import GaugeData
from troute_rnr.schemas.weather import Site
from troute_rnr.utils import get

def pull_nwm_inputs(gdf, inputs):
    pass

def write_forcast_csvs(gdf: Dict[str, pd.DataFrame]):
    # TODO call the NWPS API for forecasts from the LID, assign to top catchments, get downstream reach forecasts from others
    pass

def write_config(gdf: Dict[str, pd.DataFrame]):
    # TODO create the config required for T-Route
    pass

def format_xml(product_text: str, settings: Settings) -> List[GaugeData]:
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
