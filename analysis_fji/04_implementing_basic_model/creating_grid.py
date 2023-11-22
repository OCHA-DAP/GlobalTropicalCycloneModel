import os
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

input_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/02_housing_damage/input/"
)

# Load Fiji
adm2_shp = gpd.read_file(input_dir / "adm2_shp_fixed.gpkg")
adm2_shp = adm2_shp.to_crs("EPSG:4326")
# %%


def create_fji_grid():
    # Grid creation
    xmin, xmax, ymin, ymax = 176, 182, -21, -12  # Fiji extremes coordintates

    cell_size = 0.1

    cols = list(np.arange(xmin, xmax + cell_size, cell_size))
    rows = list(np.arange(ymin, ymax + cell_size, cell_size))
    rows.reverse()

    polygons = [
        Polygon(
            [
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y - cell_size),
                (x, y - cell_size),
            ]
        )
        for x in cols
        for y in rows
    ]
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=adm2_shp.crs)
    grid["id"] = grid.index + 1

    # Intersection of grid and shapefile
    adm2_grid_intersection = gpd.overlay(adm2_shp, grid, how="identity")

    # Grid land overlap
    grid_land_overlap = grid.loc[grid["id"].isin(adm2_grid_intersection["id"])]

    # Grids per municipality
    grid_muni = gpd.sjoin(adm2_shp, grid_land_overlap, how="inner")
    grid_muni = grid_muni.drop_duplicates(subset=["id"])

    return grid, grid_land_overlap, grid_muni
