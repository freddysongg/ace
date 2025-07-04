from pathlib import Path
import numpy as np
import pandas as pd
import py3dep
import rioxarray as rxr
import rasterio
from rasterio import sample
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from tqdm.auto import tqdm
import io
import zipfile
import requests
from shapely.geometry import box


COAST_URLS = [
    "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip",
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip",
]
COAST_DIR = Path("data/geodata/ne_coast")
COAST_SHP = COAST_DIR / "ne_10m_coastline.shp"
DEM_TIF = Path("data/geodata/dem_ca_30m.tif")


def _ensure_coastline():
    """Download and cache the Natural Earth coastline shapefile."""
    if not COAST_SHP.exists():
        print("Downloading coastline data …")
        COAST_DIR.mkdir(parents=True, exist_ok=True)

        # Try each URL until one succeeds
        success = False
        for url in COAST_URLS:
            try:
                print(f"  • Attempting {url}")
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    zf.extractall(COAST_DIR)
                success = True
                break
            except Exception as exc:
                print(f"    ↳ Failed: {exc}")

        if not success:
            raise RuntimeError(
                "Unable to download Natural Earth coastline shapefile from any known source."
            )
    return gpd.read_file(COAST_SHP).to_crs("EPSG:4326")


def _ensure_dem(df: pd.DataFrame, resolution: int = 30) -> Path:
    """Download and cache a DEM covering the dataframe's extent."""
    dem_file = Path(f"data/geodata/dem_ca_{resolution}m.tif")
    if not dem_file.exists():
        print(f"Downloading DEM at {resolution}m resolution. This may take a moment...")
        pad = 0.05
        bounds = (
            df["lon"].min() - pad,
            df["lat"].min() - pad,
            df["lon"].max() + pad,
            df["lat"].max() + pad,
        )

        bbox_poly = box(*bounds)

        dem_file.parent.mkdir(parents=True, exist_ok=True)
        dem = py3dep.get_dem(bbox_poly, resolution=resolution, crs="EPSG:4326")
        dem.rio.to_raster(dem_file)
    return dem_file


def _sample_dem(df: pd.DataFrame, dem_path: Path) -> np.ndarray:
    """Sample elevation for each point in the dataframe."""
    with rasterio.open(dem_path) as dem:
        coords = [(lon, lat) for lon, lat in zip(df["lon"].values, df["lat"].values)]
        elev = list(sample.sample_gen(dem, coords))
    return np.array(elev).squeeze()


def _nearest_coast_distance_point(lat, lon, coast_gdf, sindex):
    """Helper to calculate distance for a single point."""
    point = Point(lon, lat)
    nearest_geom_index = sindex.nearest(point)[1, 0]
    nearest_geom = coast_gdf.iloc[nearest_geom_index].geometry
    nearest_point_on_coast = nearest_geom.interpolate(nearest_geom.project(point))
    return geodesic(
        (lat, lon), (nearest_point_on_coast.y, nearest_point_on_coast.x)
    ).kilometers


def _distance_to_coast(df: pd.DataFrame, coast_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Calculate distance to coast for all points in the dataframe."""
    sindex = coast_gdf.sindex
    tqdm.pandas(desc="Calculating distance to coast (km)")
    distances = df.progress_apply(
        lambda row: _nearest_coast_distance_point(row.lat, row.lon, coast_gdf, sindex),
        axis=1,
    )
    return distances.values


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'elevation' (m) and 'distance_to_coast_km' to a DataFrame.
    Assumes 'lat' and 'lon' columns exist.
    """
    df_out = df.copy()

    coast_gdf = _ensure_coastline()
    dem_path = _ensure_dem(df_out)

    print("Sampling DEM for elevation...")
    df_out["elevation"] = _sample_dem(df_out, dem_path)

    print("Calculating distance to coast...")
    df_out["distance_to_coast_km"] = _distance_to_coast(df_out, coast_gdf)

    print("Geospatial features added successfully.")
    return df_out
