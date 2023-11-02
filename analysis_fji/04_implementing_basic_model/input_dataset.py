import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon
from climada.hazard import Centroids, TCTracks, TropCyclone, Hazard
from climada_petals.hazard import TCForecast
import warnings
# Filter out specific UserWarning by message
warnings.filterwarnings("ignore", message="Converting non-nanosecond precision datetime values to nanosecond precision")

from utils import get_stationary_data_fiji

def create_windfield_dataset():
    output_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/02_new_model_input/02_housing_damage/output/"
        )

    # Load grids
    grids = gpd.read_file(output_dir / "fji_0.1_degree_grid_land_overlap_new.gpkg")
    grids.geometry = grids.geometry.to_crs(grids.crs).centroid

    # Load ECMWF data
    tc_fcast = TCForecast()
    tc_fcast.fetch_ecmwf()

    # Predict Windfield on centroids
    cent = Centroids.from_geodataframe(grids)
    tc = TropCyclone.from_tracks(
        tc_fcast, centroids=cent, store_windfields=True, intensity_thres=0
    )

    # Create windfield dataset
    event_names = list(tc.event_name)

    # Define the boundaries for Fiji region
    xmin, xmax, ymin, ymax = 176, 182, -21, -12
    fiji_polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    df_windfield = pd.DataFrame()
    for i, intensity_sparse in enumerate(tc.intensity):
        # Get the windfield
        windfield = intensity_sparse.toarray().flatten()
        npoints = len(windfield)
        event_id = event_names[i]

        # Track distance
        DEG_TO_KM = 111.1
        tc_track = tc_fcast.get_track()[i]
        points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
        tc_track_line = LineString(points)
        tc_track_distance = grids["geometry"].apply(
            lambda point: point.distance(tc_track_line) * DEG_TO_KM
        )

        # Basin
        basin = np.unique(tc_track.basin)

        # Adquisition Period
        time0 = np.unique(tc_track.time)[0]
        time1 = np.unique(tc_track.time)[-1]

        # Does it touch Fiji borders?
        intersects_fiji = tc_track_line.intersects(fiji_polygon)

        # Add to DF
        df_to_add = pd.DataFrame(
            dict(
                event_id_ecmwf=[event_id] * npoints,
                unique_id = [i] * npoints,
                basins=[basin.tolist()] * npoints,
                time_init=[time0] * npoints,
                time_end=[time1] * npoints,
                in_fiji=[intersects_fiji] * npoints,
                grid_point_id=grids["id"],
                wind_speed=windfield,
                track_distance=tc_track_distance,
                geometry = grids.geometry
            )
        )
        df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)

    return df_windfield

def create_input_dataset(df_windfield):
    # Windfield dataset
    #df_windfield = create_windfield_dataset()

    # Stationaty data
    df_stationary = get_stationary_data_fiji()

    # Focus on Fiji
    fiji_forecast = df_windfield[df_windfield.in_fiji == True]

    # Merge everything
    input_df = df_stationary.merge(fiji_forecast, left_on='grid_point_id', right_on='grid_point_id')[
        ['grid_point_id',
        'IWI',
        'total_buildings',
        'with_coast',
        'coast_length',
        'mean_altitude',
        'mean_slope',
        'wind_speed',
        'track_distance',
        'event_id_ecmwf',
        'unique_id',
        'time_init',
        'time_end'
        ]].reset_index(drop=True)

    return input_df
