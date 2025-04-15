"""Schema definitions for NWPS data models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RFC(BaseModel):
    """
    River Forecast Center information.

    Parameters
    ----------
    abbreviation : str
        The abbreviated name of the RFC.
    name : str
        The full name of the RFC.
    """

    abbreviation: str
    name: str


class WFO(BaseModel):
    """
    Weather Forecast Office information.

    Parameters
    ----------
    abbreviation : str
        The abbreviated name of the WFO.
    name : str
        The full name of the WFO.
    """

    abbreviation: str
    name: str


class State(BaseModel):
    """
    State information.

    Parameters
    ----------
    abbreviation : str
        The state's two-letter abbreviation.
    name : str
        The full name of the state.
    """

    abbreviation: str
    name: str


class PEDTS(BaseModel):
    """
    Physical Element Data Type and Source information.

    Parameters
    ----------
    observed : str
        The observed PEDTS.
    forecast : str
        The forecast PEDTS.
    """

    observed: str
    forecast: str


class StatusData(BaseModel):
    """
    Status data for a gauge reading.

    Parameters
    ----------
    primary : float
        The primary measurement value.
    primaryUnit : str
        The unit of the primary measurement.
    secondary : float
        The secondary measurement value.
    secondaryUnit : str
        The unit of the secondary measurement.
    floodCategory : str
        The current flood category.
    validTime : datetime
        The timestamp when this status was recorded.
    """

    primary: float
    primaryUnit: str
    secondary: float
    secondaryUnit: str
    floodCategory: str
    validTime: datetime


class Status(BaseModel):
    """
    Overall status including observed and forecast data.

    Parameters
    ----------
    observed : StatusData
        The observed status data.
    forecast : StatusData
        The forecast status data.
    """

    observed: StatusData
    forecast: StatusData


class FloodCategory(BaseModel):
    """
    Flood category thresholds.

    Parameters
    ----------
    stage : float
        The stage (water level) threshold for this category.
    flow : float
        The flow rate threshold for this category.
    """

    stage: float
    flow: float


class FloodCategories(BaseModel):
    """
    Thresholds for different flood categories.

    Parameters
    ----------
    major : FloodCategory
        Thresholds for major flooding.
    moderate : FloodCategory
        Thresholds for moderate flooding.
    minor : FloodCategory
        Thresholds for minor flooding.
    action : FloodCategory
        Thresholds for flood action stage.
    """

    major: FloodCategory
    moderate: FloodCategory
    minor: FloodCategory
    action: FloodCategory


class LRO(BaseModel):
    """
    Long Range Outlook information.

    Parameters
    ----------
    minorCS : str
        Minor flood chance statement.
    moderateCS : str
        Moderate flood chance statement.
    majorCS : str
        Major flood chance statement.
    producedTime : datetime
        Time when the outlook was produced.
    interval : str
        Interval of the outlook.
    """

    minorCS: str
    moderateCS: str
    majorCS: str
    producedTime: datetime
    interval: str


class Crest(BaseModel):
    """
    Information about a flood crest.

    Parameters
    ----------
    occurredTime : datetime
        Time when the crest occurred.
    stage : float
        Water stage at crest.
    flow : float
        Flow rate at crest.
    preliminary : str
        Indicator if this is a preliminary crest.
    olddatum : bool
        Indicator if this uses an old datum.
    """

    occurredTime: datetime
    stage: float
    flow: float
    preliminary: str
    olddatum: bool


class LowWater(BaseModel):
    """
    Information about low water conditions.

    Parameters
    ----------
    occurredTime : datetime
        Time when the low water condition occurred.
    stage : float
        Water stage at low water.
    flow : float
        Flow rate at low water.
    statement : str
        Statement about the low water condition.
    """

    occurredTime: datetime
    stage: float
    flow: float
    statement: str


class Impact(BaseModel):
    """
    Information about flood impacts at a certain stage.

    Parameters
    ----------
    stage : float
        Water stage at which this impact occurs.
    statement : str
        Description of the impact.
    """

    stage: float
    statement: str


class Flood(BaseModel):
    """
    Comprehensive flood information.

    Parameters
    ----------
    stageUnits : str
        Units used for stage measurements.
    flowUnits : str
        Units used for flow measurements.
    categories : FloodCategories
        Thresholds for different flood categories.
    lro : LRO | None
        Long Range Outlook information, if available.
    crests : dict[str, list[Crest]]
        Historical and forecasted flood crests.
    lowWaters : dict[str, list[LowWater]]
        Historical and forecasted low water conditions.
    impacts : list[Impact]
        List of flood impacts at different stages.
    """

    stageUnits: str
    flowUnits: str
    categories: FloodCategories
    lro: Optional[LRO]
    crests: dict[str, list[Crest]]
    lowWaters: dict[str, list[LowWater]]
    impacts: list[Impact]


class ProbabilityImages(BaseModel):
    """
    Links to probability images.

    Parameters
    ----------
    stage : str
        Link to stage probability image.
    flow : str
        Link to flow probability image.
    volume : str
        Link to volume probability image.
    """

    stage: str
    flow: str
    volume: str


class Probability(BaseModel):
    """
    Probability information for different time ranges.

    Parameters
    ----------
    weekint : ProbabilityImages
        Week interval probability images.
    entperiod : ProbabilityImages
        Entire period probability images.
    shortrange : str
        Link to short range probability image.
    """

    weekint: ProbabilityImages
    entperiod: ProbabilityImages
    shortrange: str


class Hydrograph(BaseModel):
    """
    Links to hydrograph images.

    Parameters
    ----------
    default : str
        Link to default hydrograph image.
    floodcat : str
        Link to flood category hydrograph image.
    """

    default: str
    floodcat: str


class PhotoGeometry(BaseModel):
    """
    Geometry information for a photo.

    Parameters
    ----------
    type : str
        Type of geometry (e.g., "Point").
    coordinates : list[float]
        Coordinates of the photo location.
    """

    type: str
    coordinates: list[float]


class PhotoProperties(BaseModel):
    """
    Properties of a photo.

    Parameters
    ----------
    image : str
        Link to the image file.
    caption : str
        Caption for the photo.
    """

    image: str
    caption: str


class Photo(BaseModel):
    """
    Information about a photo.

    Parameters
    ----------
    id : str
        Unique identifier for the photo.
    type : str
        Type of the photo data.
    geometry : PhotoGeometry
        Geometry information for the photo.
    properties : PhotoProperties
        Properties of the photo.
    """

    id: str
    type: str
    geometry: PhotoGeometry
    properties: PhotoProperties


class Images(BaseModel):
    """
    Collection of various images related to the gauge.

    Parameters
    ----------
    probability : Probability
        Probability images for different time ranges.
    hydrograph : Hydrograph
        Hydrograph images.
    photos : list[Photo]
        List of photos related to the gauge location.
    """

    probability: Probability
    hydrograph: Hydrograph
    photos: list[Photo]


class DataAttribution(BaseModel):
    """
    Attribution information for data sources.

    Parameters
    ----------
    abbrev : str
        Abbreviation of the data source.
    text : str
        Full text of the attribution.
    title : str
        Title of the data source.
    url : str
        URL for more information about the data source.
    """

    abbrev: str
    text: str
    title: str
    url: str


class ImpactLowWater(BaseModel):
    """
    Information about low water impacts.

    Parameters
    ----------
    value : str
        Value at which the low water impact occurs.
    impact : str
        Description of the low water impact.
    """

    value: str
    impact: str


class NormalThreshold(BaseModel):
    """
    Normal water level threshold information.

    Parameters
    ----------
    value : float
        The value of the normal threshold.
    units : str
        Units of measurement for the threshold.
    """

    value: float
    units: str


class Hydronote(BaseModel):
    """
    Hydrologic note information.

    Parameters
    ----------
    statement : str
        The content of the hydrologic note.
    effective : str
        The time when the note becomes effective.
    expiration : str
        The time when the note expires.
    """

    statement: str
    effective: str
    expiration: str


class DatumValue(BaseModel):
    """
    Information about a specific datum value.

    Parameters
    ----------
    label : str
        Label for the datum value.
    abbrev : str
        Abbreviation for the datum value.
    description : str
        Description of the datum value.
    value : float
        The numerical value of the datum.
    """

    label: str
    abbrev: str
    description: str
    value: float


class Datums(BaseModel):
    """
    Collection of datum information.

    Parameters
    ----------
    vertical : dict[str, list[DatumValue]]
        Vertical datum information.
    horizontal : dict[str, list[DatumValue]]
        Horizontal datum information.
    notes : dict[str, list[str]]
        Additional notes about the datums.
    """

    vertical: dict[str, list[DatumValue]]
    horizontal: dict[str, list[DatumValue]]
    notes: dict[str, list[str]]


class ZeroDatum(BaseModel):
    """
    Information about the zero datum.

    Parameters
    ----------
    value : float
        The value of the zero datum.
    datum : str
        The type or name of the datum.
    """

    value: float
    datum: str


class Downloads(BaseModel):
    """
    Links to downloadable data.

    Parameters
    ----------
    depthGrids : str
        Link to depth grids data.
    images : str
        Link to image data.
    kmz : str
        Link to KMZ file.
    """

    depthGrids: str
    images: str
    kmz: str


class InundationDataAttribution(BaseModel):
    """
    Attribution information for inundation data.

    Parameters
    ----------
    text : str
        Attribution text.
    title : str
        Title of the data source.
    url : str
        URL for more information.
    image : str
        Link to an image related to the attribution.
    """

    text: str
    title: str
    url: str
    image: str


class Inundation(BaseModel):
    """
    Information about inundation data and services.

    Parameters
    ----------
    enabled : bool
        Whether inundation data is enabled.
    url : str
        URL for inundation data.
    zeroDatum : ZeroDatum | None
        Zero datum information, if available.
    downloads : Downloads | None
        Links to downloadable data, if available.
    siteSpecificInfo : str
        Site-specific inundation information.
    dataAttribution : list[InundationDataAttribution]
        List of data attributions for inundation data.
    """

    enabled: bool
    url: str
    zeroDatum: Optional[ZeroDatum]
    downloads: Optional[Downloads]
    siteSpecificInfo: str
    dataAttribution: list[InundationDataAttribution]


class InService(BaseModel):
    """
    Information about the service status of the gauge.

    Parameters
    ----------
    enabled : bool
        Whether the gauge is in service.
    message : str
        Any message related to the service status.
    """

    enabled: bool
    message: str


class LowThreshold(BaseModel):
    """
    Information about the low water threshold.

    Parameters
    ----------
    units : str
        Units of measurement for the threshold.
    value : float
        The value of the low threshold.
    """

    units: str
    value: float


class GaugeData(BaseModel):
    """
    Comprehensive data about a gauge.

    Parameters
    ----------
    lid : str
        Location ID of the gauge.
    usgsId : str
        USGS ID of the gauge.
    reachId : str
        Reach ID associated with the gauge.
    name : str
        Name of the gauge location.
    description : str
        Description of the gauge location.
    rfc : RFC
        River Forecast Center information.
    wfo : WFO
        Weather Forecast Office information.
    state : State
        State information.
    county : str
        County where the gauge is located.
    timeZone : str
        Time zone of the gauge location.
    latitude : float
        Latitude of the gauge location.
    longitude : float
        Longitude of the gauge location.
    pedts : PEDTS
        Physical Element Data Type and Source information.
    status : Status
        Current status of the gauge.
    flood : Flood
        Flood-related information.
    images : Images
        Collection of related images.
    dataAttribution : list[DataAttribution]
        List of data attributions.
    impactsLowWaters : list[ImpactLowWater]
        List of low water impacts.
    normalThreshold : NormalThreshold | None
        Normal water level threshold, if available.
    hydronotes : list[Hydronote]
        List of hydrologic notes.
    datums : Datums
        Datum information.
    inundation : Inundation
        Inundation data and services information.
    upstreamLid : str
        Location ID of the upstream gauge.
    downstreamLid : str
        Location ID of the downstream gauge.
    inService : InService
        Service status information.
    lowThreshold : LowThreshold | None
        Low water threshold information, if available.
    forecastReliability : str
        Information about the reliability of forecasts.
    TruncateObs : str
        Information about truncation of observations.
    TruncateFcst : str
        Information about truncation of forecasts.
    ObservedFloodCategory : str
        Observed flood category.
    ForecastFloodCategory : str
        Forecast flood category.
    """

    lid: str
    usgsId: str
    reachId: str
    name: str
    description: str
    rfc: RFC
    wfo: WFO
    state: State
    county: str
    timeZone: str
    latitude: float
    longitude: float
    pedts: PEDTS
    status: Status
    flood: Flood
    images: Images
    dataAttribution: list[DataAttribution]
    impactsLowWaters: list[ImpactLowWater]
    normalThreshold: Optional[NormalThreshold]
    hydronotes: list[Hydronote]
    datums: Datums
    inundation: Inundation
    upstreamLid: str
    downstreamLid: str
    inService: InService
    lowThreshold: Optional[LowThreshold]
    forecastReliability: str
    TruncateObs: str
    TruncateFcst: str
    ObservedFloodCategory: str
    ForecastFloodCategory: str


class GaugeForecast(BaseModel):
    """
    Forecast data for a gauge.

    Parameters
    ----------
    times : list[datetime]
        List of forecast times.
    primary_name : str
        Name of the primary forecast parameter.
    primary_forecast : list[float]
        List of primary forecast values.
    primary_unit : str
        Unit of measurement for primary forecast.
    secondary_name : str
        Name of the secondary forecast parameter.
    latest_observation: list[float]
        The latest observation from NWPS.
    latest_obs_units: str
        The latest observation units.
    secondary_forecast : list[float]
        List of secondary forecast values.
    secondary_unit : str
        Unit of measurement for secondary forecast.
    """

    times: list[datetime]
    primary_name: str
    primary_forecast: list[float]
    primary_unit: str
    latest_observation: list[float]
    latest_obs_units: str
    secondary_name: str
    secondary_forecast: list[float]
    secondary_unit: str


class ReachClassification(str, Enum):
    """
    Classification types for reaches.

    Attributes
    ----------
    rfc_point : str
        RFC point classification.
    flowline : str
        Flowline classification.
    """

    rfc_point = "rfc_point"
    flowline = "flowline"


class Reach(BaseModel):
    """
    Information about a river reach.

    Parameters
    ----------
    reach_id : int
        Identifier for the reach.
    downstream_reach_id : int
        Identifier for the downstream reach.
    reach_classification : ReachClassification
        Classification of the reach.
    times : list[datetime]
        List of times for the forecast data.
    forecast : list[float]
        List of forecast values.
    """

    reach_id: int
    downstream_reach_id: int
    reach_classification: ReachClassification
    times: list[datetime]
    forecast: list[float]


class ProcessedData(BaseModel):
    """
    Container for processed data about a location.

    Parameters
    ----------
    lid : str
        Location ID.
    downstream_lid : str
        Downstream location ID.
    reaches : list[Reach] | None
        List of reaches, if available.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    lid: str
    downstream_lid: str
    reaches: Optional[list[Reach]]


class ResultItem(BaseModel):
    """
    Represents the result of processing a single RFC entry.

    Parameters
    ----------
    status : str
        The status of the processing operation.
        Possible values: 'success', 'no_forecast', 'api_error', 'error'.
    lid : str
        The location ID (LID) of the processed RFC entry.
    error_type : str | None, optional
        The exception/error that was raised.
    error_message : str | None, optional
        The error message that was raised.
    status_code : str | None, optional
        The status code of the exception if applicable.
    """

    status: str
    lid: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    status_code: Optional[str] = None


class Summary(BaseModel):
    """
    Summarizes the results of processing multiple RFC entries.

    Parameters
    ----------
    total : int
        The total number of RFC entries processed.
    success : int
        The number of entries successfully processed.
    no_forecast : int
        The number of entries that had no forecast available.
    api_error : int
        The number of entries that encountered an API error.
    validation_error : int
        The number of entries that encountered a validation error.
    """

    total: int
    success: int
    no_forecast: int
    api_error: int
    validation_error: int


class PublishMessagesResponse(BaseModel):
    """
    Represents the full response of the publish_messages endpoint.

    Parameters
    ----------
    status : int
        The HTTP status code of the response.
    summary : Summary
        A summary of the processing results.
    results : list[ResultItem]
        Detailed results for each processed RFC entry.
    """

    status: int
    summary: Summary
    results: list[ResultItem]


class ConsumerStatus(BaseModel):
    """
    Status of a consumer service.

    Parameters
    ----------
    is_running : bool, default=False
        Whether the consumer is currently running.
    """

    is_running: bool = False
