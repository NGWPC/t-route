"""Module for handling NWM data processing and NWPS integrations."""

import httpx
import lxml.etree
from pydantic.error_wrappers import ValidationError

from troute_rnr.schemas.nwps import GaugeData, ProcessedData, Reach, ReachClassification
from troute_rnr.schemas.weather import Site
from troute_rnr.settings import Settings
from troute_rnr.utils import get


def get_reach_flow(reach_id: int, settings: Settings) -> Reach:
    """
    Fetch flow data for a specific reach.

    Parameters
    ----------
    reach_id : int
        The identifier for the reach to fetch.
    settings : Settings
        Configuration settings containing base URL and other parameters.

    Returns
    -------
    Reach
        Object containing the reach flow data.
    """
    flow_endpoint = f"{settings.BASE_URL}/reaches/{reach_id}/streamflow?series=short_range"
    reach_flow = get(flow_endpoint).json()
    return Reach(
        reach_id=reach_id,
        downstream_reach_id=int(reach_flow["reach"]["route"]["downstream"][0]["reachId"]),
        reach_classification=ReachClassification.flowline,
        times=[data["validTime"] for data in reach_flow["shortRange"]["series"]["data"]],
        forecast=[data["flow"] for data in reach_flow["shortRange"]["series"]["data"]],
    )


def fetch_all_flows(processed_data: ProcessedData, gauge_data: GaugeData, settings: Settings) -> list[Reach]:
    """
    Fetch flow data for all reaches in the route.

    Parameters
    ----------
    processed_data : ProcessedData
        Already processed data containing initial reach information.
    gauge_data : GaugeData
        Gauge data containing the downstream LID.
    settings : Settings
        Configuration settings.

    Returns
    -------
    list[Reach]
        list of reach objects containing flow data.
    """
    endpoint = f"{settings.BASE_URL}/gauges/{gauge_data.downstreamLid}"
    forecast = get(endpoint).json()
    ending_reach_id = int(forecast["reachId"])
    output: list[Reach] = []
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


def pull_nwm_inputs(forecast: GaugeData, settings: Settings) -> ProcessedData | None:
    """
    Pull National Water Model inputs for a given forecast.

    Parameters
    ----------
    forecast : GaugeData
        Gauge data containing forecast information.
    settings : Settings
        Configuration settings.

    Returns
    -------
    ProcessedData | None
        Processed data containing reach information or None if data is invalid.
    """
    forecast_endpoint = f"{settings.BASE_URL}/gauges/{forecast.lid}/stageflow/forecast"
    site_data = get(forecast_endpoint).json()
    if site_data["data"][0]["secondary"] == -999:
        return None

    metadata_endpoint = f"{settings.BASE_URL}/reaches/{forecast.reachId}"
    downstream_metadata = get(metadata_endpoint).json()
    downstream_reach_id = int(downstream_metadata["route"]["downstream"][0]["reachId"])
    processed_data = ProcessedData(
        lid=forecast.lid,
        downstream_lid=forecast.downstreamLid,
        reaches=[
            Reach(
                reach_id=forecast.reachId,
                downstream_reach_id=downstream_reach_id,
                reach_classification=ReachClassification.rfc_point,
                times=[val["validTime"] for val in site_data["data"]],
                forecast=[val["secondary"] for val in site_data["data"]],
            )
        ],
    )
    flowline_data = fetch_all_flows(processed_data, forecast, settings)
    processed_data.reaches.extend(flowline_data)
    return processed_data


def write_forecast_csvs() -> None:
    """
    Write forecast data to CSV files.

    Parameters
    ----------
    gdf : Dict[str, pd.DataFrame]
        Dictionary of GeoDataFrames.
    inputs : ProcessedData
        Processed data containing reach information.

    Returns
    -------
    None
    """
    # TODO: create the csvs required for T-Route
    pass


def write_config() -> None:
    """
    Create the configuration required for T-Route.

    Parameters
    ----------
    gdf : Dict[str, pd.DataFrame]
        Dictionary of GeoDataFrames.

    Returns
    -------
    None
    """
    # TODO: create the config required for T-Route
    pass


def format_xml(product_text: str, settings: Settings) -> list[GaugeData]:
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
    List[GaugeData]
        List of gauge data objects extracted from the XML.
    """
    xml_split = product_text.split("?xml")
    forecasts = []

    # Ignore the first one since it's not valid XML
    for i in range(1, len(xml_split)):
        xml_segment = "<?xml" + xml_split[i][:-2]  # Adding removed XML tag, and removed trailing tags
        try:
            site = Site.from_xml(xml_segment)
        except lxml.etree.XMLSyntaxError:
            # Removing extra content at end of document
            xml_segment = xml_segment.split("</site>")[0] + "</site>"
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
            continue

    return forecasts
