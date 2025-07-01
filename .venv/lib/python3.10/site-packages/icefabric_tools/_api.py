"""Contains all api functions that can be called outside of the icefabric_tools package"""

import geopandas as gpd
import pandas as pd
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.expressions import BooleanExpression
from pyiceberg.table import ALWAYS_TRUE, Table
from shapely import wkb


def load_hydrofabric(catalog_settings: dict[str, str]) -> Catalog:
    """A function to read in the hydrofabric catalog

    Parameters
    ----------
    catalog_settings : dict[str, str]
        The settings to read the hydrofabric catalog

    Returns
    -------
    Catalog
        The Iceberg catalog
    """
    return load_catalog("hydrofabric", **catalog_settings)

def table_to_geopandas(
    table: Table,
    row_filter: str | BooleanExpression = ALWAYS_TRUE,
    case_sensitive: bool | None = True,
    snapshot_id: int | None = None,
    limit: int | None = None
) -> gpd.GeoDataFrame:
    """Converts a table to a geopandas dataframe

    Parameters
    ----------
    table : Table
        The iceberg table you are trying to read from
    row_filter : str | None, optional
        A string or BooleanExpression that describes the desired rows, by default ""
    case_sensitive : bool | None, optional
        If True column matching is case sensitive, by default True
    snapshot_id : int | None, optional
        Optional Snapshot ID to time travel to.
        If None, scans the table as of the current snapshot ID, by default None
    limit : int | None, optional
        An integer representing the number of rows to return in the scan result.
        If None, fetches all matching rows., by default None

    Returns
    -------
    gpd.DataFrame
        The resulting queried row, but in a geodataframe
    """
    df = table.scan(
        row_filter=row_filter,
        case_sensitive=case_sensitive,
        snapshot_id=snapshot_id,
        limit=limit,
    ).to_pandas()
    return to_geopandas(df)


def to_geopandas(
    df: pd.DataFrame,
    crs: str = "EPSG:5070"
) -> gpd.GeoDataFrame:
    """Converts the geometries in a pandas df to a geopandas dataframe

    Parameters
    ----------
    df: pd.DataFrame
        The iceberg table you are trying to read from
    crs: str, optional
        A string representing the CRS to set in the gdf, by default "EPSG:5070"

    Returns
    -------
    gpd.DataFrame
        The resulting queried row, but in a geodataframe

    Raises
    ------
    ValueError
        Raised if the table does not have a geometry column
    """
    if "geometry" not in df.columns:
        raise ValueError("The provided table does not have a geometry column.")

    geometry = df["geometry"].apply(lambda x: wkb.loads(x) if x is not None else None)
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


def find_origin(network_table: Table, identifier: str, id_type: str ="hl_uri") -> pd.DataFrame:
    """Find an origin point in the hydrofabric network.

    This function handles the case where multiple records match the identifier.
    It follows the R implementation to select a single origin point based on
    the minimum hydroseq value.

    Parameters
    ----------
    network_table : Table
        The HF network table from the hydrofabric catalog
    identifier : str
        The unique identifier you want to find the origin of
    id_type : str, optional
        The network table column you can query from, by default "hl_uri"

    Returns
    -------
    pd.DataFrame
        The origin row from the network table

    Raises
    ------
    ValueError
        The provided identifier is not supported
    ValueError
        No origin for the point is found
    ValueError
        Multiple origins for the point are found
    """
    # Filter network table by the identifier
    if id_type == "hl_uri":
        row_filter = f"{id_type} == '{identifier}'"
    elif id_type == "comid":
        row_filter = f"hf_id == {identifier}"
    elif id_type == "id":
        row_filter = f"id == '{identifier}'"
    elif id_type == "poi_id":
        row_filter = f"poi_id == '{identifier}'"
    else:
        raise ValueError(f"Identifier {id_type} not supported")

    # Get all matching records
    origin_candidates = network_table.scan(row_filter=row_filter).to_pandas()

    if len(origin_candidates) == 0:
        raise ValueError(f"No origin found for {id_type}='{identifier}'")

    # Select relevant columns for the origin
    origin_cols = ["id", "toid", "vpuid", "topo", "hydroseq"]
    available_cols = [col for col in origin_cols if col in origin_candidates.columns]

    # Select only the relevant columns and drop duplicates
    origin = origin_candidates[available_cols].drop_duplicates()

    # Find the record with minimum hydroseq
    if "hydroseq" in origin.columns:
        # For consistency with R, check if there are unique hydroseq values
        if len(origin["hydroseq"].unique()) > 1:
            # Sort by hydroseq and take the minimum
            origin = origin.sort_values("hydroseq").iloc[0:1]

    # If we still have multiple records, it's a problem
    if len(origin) > 1:
        raise ValueError(f"Multiple origins found: {origin['id'].tolist()}")

    return origin
