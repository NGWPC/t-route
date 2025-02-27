from datetime import datetime
from pathlib import Path
from typing import Dict, List
import os

import geopandas as gpd
import httpx
import lxml
import numpy as np
import pandas as pd
from pydantic.error_wrappers import ValidationError

from troute_rnr.schemas.nwps import GaugeData, Reach, ReachClassification, ProcessedData
from troute_rnr.schemas.weather import Site
from troute_rnr.settings import Settings
from troute_rnr.utils import get

def get_reach_flow(reach_id, settings):
    flow_endpoint = f"{settings.BASE_URL}/reaches/{reach_id}/streamflow?series=short_range"
    reach_flow = get(flow_endpoint).json()
    return Reach(
        reach_id=reach_id,
        downstream_reach_id=int(reach_flow["reach"]["route"]["downstream"][0]["reachId"]),
        reach_classification=ReachClassification.flowline,
        times=[data["validTime"] for data in reach_flow["shortRange"]["series"]["data"]],
        forecast=[data["flow"] for data in reach_flow["shortRange"]["series"]["data"]]
    )

def fetch_all_flows(processed_data: ProcessedData, gauge_data: GaugeData, settings: Settings) -> List[Reach]:
    endpoint = f"{settings.BASE_URL}/gauges/{gauge_data.downstreamLid}"
    forecast = get(endpoint).json()
    ending_reach_id = int(forecast["reachId"])
    output = []
    downstream_reach_id = processed_data.reaches[0].downstream_reach_id
    counter = 0
    print("Pulling input reach forecasts")
    while downstream_reach_id != ending_reach_id and counter <= settings.reach_limit:
        reach = get_reach_flow(downstream_reach_id, settings)
        output.append(reach)
        downstream_reach_id = reach.downstream_reach_id
        counter += 1
    end_reach = get_reach_flow(ending_reach_id, settings)
    output.append(end_reach)
    return output

def pull_nwm_inputs(forecast, settings: Settings) -> ProcessedData:
    forecast_endpoint = f"{settings.BASE_URL}/gauges/{forecast.lid}/stageflow/forecast"
    site_data = get(forecast_endpoint).json()
    if site_data["data"][0]["secondary"] == -999:
        return None
    else:
        metadata_endpoint = f"{settings.BASE_URL}/reaches/{forecast.reachId}"
        downstream_metadata = get(metadata_endpoint).json()
        downstream_reach_id = int(downstream_metadata["route"]["downstream"][0]["reachId"])
        processed_data = ProcessedData(
            lid = forecast.lid,
            downstream_lid = forecast.downstreamLid,
            reaches = [
                Reach(
                    reach_id=forecast.reachId,
                    downstream_reach_id=downstream_reach_id,
                    reach_classification=ReachClassification.rfc_point,
                    times = [val["validTime"] for val in site_data["data"]],
                    forecast=[val["secondary"] for val in site_data["data"]],
                )
            ]
        )
        flowline_data = fetch_all_flows(processed_data, forecast, settings)
        processed_data.reaches.extend(flowline_data)
    return processed_data


def write_forcast_csvs(gdf: Dict[str, pd.DataFrame], inputs: ProcessedData):
    output_path = Path(__file__).parents[0] / "base_files/tmp"
    upstream_reach: Reach = inputs.reaches[0]
    times = upstream_reach.times
    for idx, time in enumerate(times):
        try:
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
        formatted_time = dt.strftime("%Y%m%d%H%M")
        _df = pd.DataFrame(
            {
                "feature_id": [mapped_feature_id],
                formatted_time: [filtered_data[idx]],
            }
        )
        if not os.path.exists(os.path.join(output_path, lid)):
            os.makedirs(os.path.normpath(os.path.join(output_path, lid)))
        file_path = os.path.normpath(
            os.path.join(output_path, lid, formatted_time + ".CHRTOUT_DOMAIN1.csv")
        )
        _df.to_csv(file_path, index=False)
        domain_files.append(
            {
                "lid": lid,
                "formatted_time": formatted_time,
                "file_location": file_path,
                "secondary_forecast": filtered_data[idx],
            }
        )

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
