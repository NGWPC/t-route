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


def get_site_data(site: Site, settings: Settings) -> GaugeData | None:
    """Retrieves gauge data from the NWPS API for a specific site and validates it meets flood criteria.

    Parameters
    ----------
    site : Site
        The site object containing properties with an 'id' field used to construct the API endpoint
    settings : Settings
        Configuration object containing BASE_URL for the API endpoint and STAGES for flood criteria validation

    Returns
    -------
    GaugeData | None
        A validated GaugeData object if the site data meets flood criteria requirements,
        None if the site's flood category does not meet the required criteria

    Raises
    ------
    ValidationError
        If the API response cannot be parsed into a valid GaugeData object
    httpx.HTTPStatusError
        If the API request fails or returns an error status
    """
    endpoint = f"{settings.BASE_URL}/gauges/{site.properties['id']}"
    try:
        forecast = get(endpoint).json()
        try:
            gauge_data = GaugeData(**forecast)
        except ValidationError as e:
            msg = f"ValidationError: Pydantic validation error for the endpoint given: {endpoint}"
            print(msg)
            raise e
        if gauge_data.ForecastFloodCategory in settings.STAGES:
            return gauge_data
        else:
            msg = f"This site does not meet the criteria for a flood: {gauge_data.lid}"
            print(msg)
            return None
    except httpx.HTTPStatusError as e:
        msg = f"HTTPStatusError: There was no forecast/record within NWPS for the site given: {endpoint}"
        print(msg)
        raise httpx.HTTPStatusError(msg) from e
