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
warnings.filterwarnings(
    "ignore",
    message="Converting non-nanosecond precision datetime values to nanosecond precision",
)

from datetime import datetime, timedelta
import time
import datetime as dt
from bs4 import BeautifulSoup
import requests

from shapely.geometry import Polygon
import rasterio
from rasterio.features import geometry_mask
from rasterstats import zonal_stats

from utils import get_stationary_data_fiji


def create_windfield_dataset(thres=120):
    output_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/02_new_model_input/02_housing_damage/output/"
    )

    # Load grids
    grids = gpd.read_file(
        output_dir / "fji_0.1_degree_grid_land_overlap_new.gpkg"
    )
    grids.geometry = grids.geometry.to_crs(grids.crs).centroid

    # Load ECMWF data
    tc_fcast = TCForecast()
    tc_fcast.fetch_ecmwf()

    # Modify each of the event based on threshold
    n_events = len(tc_fcast.data)
    # Threshold
    thres = 120  # h
    today = datetime.now()
    # Calculate the threshold datetime from the current date and time
    threshold_datetime = np.datetime64(today + timedelta(hours=thres))

    xarray_data_list = []
    for i in range(n_events):
        data_event = tc_fcast.data[i]
        # Elements to consider
        index_thres = len(
            np.where(np.array(data_event.time) < threshold_datetime)[0]
        )
        if index_thres > 4:  # Events with at least 4 datapoints
            data_event_thres = data_event.isel(time=slice(0, index_thres))
            xarray_data_list.append(data_event_thres)
        else:
            continue

    # Create TropCyclone class with modified data (Predict Windfield on centroids)
    cent = Centroids.from_geodataframe(grids)
    tc_fcast_mod = TCForecast(xarray_data_list)
    tc = TropCyclone.from_tracks(
        tc_fcast_mod, centroids=cent, store_windfields=True, intensity_thres=0
    )

    # Create windfield dataset
    event_names = list(tc.event_name)

    # Define the boundaries for Fiji region
    xmin, xmax, ymin, ymax = 176, 182, -21, -12
    fiji_polygon = Polygon(
        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    )

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
                unique_id=[i] * npoints,
                basins=[basin.tolist()] * npoints,
                time_init=[time0] * npoints,
                time_end=[time1] * npoints,
                in_fiji=[intersects_fiji] * npoints,
                grid_point_id=grids["id"],
                wind_speed=windfield,
                track_distance=tc_track_distance,
                geometry=grids.geometry,
            )
        )
        df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)

    return df_windfield


def download_gefs_data(download_folder):
    # Calculate the current date
    today = datetime.now().strftime("%Y%m%d")

    # Generate folder names for 18, 12, 06, and 00 h
    folder_names = [f"{today}/18", f"{today}/12", f"{today}/06", f"{today}/00"]

    for folder in folder_names:
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.{folder}/prcp_bc_gb2/"
        response = requests.get(url)

        if response.status_code == 200:
            print("Downloading ", url)
            soup = BeautifulSoup(response.text, "html.parser")
            links = [
                a["href"]
                for a in soup.find_all("a")
                if a["href"].startswith("geprcp.t")
                and a["href"][:-7].endswith("bc_")
            ]

            if links:
                for link in links:
                    link_url = url + link
                    file_name = link.split("/")[-1]
                    file_path = os.path.join(download_folder, file_name)

                    # print(f"Downloading {link_url}")
                    download_response = requests.get(link_url)

                    with open(file_path, "wb") as f:
                        f.write(download_response.content)
                return
            break
    print("prcp_gb2 not found for today.")


# Define a function to adjust the longitude of a single polygon
def adjust_longitude(polygon):
    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)

    # Adjust longitudes from [0, 360) to [-180, 180)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)

    # Create a new Polygon with adjusted coordinates
    return Polygon(coords)


def create_rainfall_dataset(df_windfield):
    # Setting directories
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/02_new_model_input/03_rainfall/input"
    )

    output_dir = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/02_new_model_input/03_rainfall/output"
    )

    grid_input = (
        Path(os.getenv("STORM_DATA_DIR"))
        / "analysis_fji/02_new_model_input/02_housing_damage/output/"
    )

    # Focus on Fiji
    fiji_forecast = df_windfield[df_windfield.in_fiji == True]

    # Group the DataFrame by the 'unique_id' column
    grouped = fiji_forecast.groupby("unique_id")

    # Create a list of Forecasts DataFrames, one for each unique_id
    list_forecast = [group for name, group in grouped]

    # Metadata of events
    event_metadata = pd.DataFrame(
        {
            "event": fiji_forecast.unique_id.unique(),
            "start_date": [df.time_init.iloc[0] for df in list_forecast],
            "end_date": [df.time_end.iloc[0] for df in list_forecast],
        }
    )

    # Download today's forecasts
    today = datetime.now().strftime("%Y%m%d")
    download_folder = Path(input_dir) / "NOMADS" / today / "input_files"
    os.makedirs(download_folder, exist_ok=True)
    download_gefs_data(download_folder)

    # Load grid
    grid_land_overlap = gpd.read_file(
        grid_input / "fji_0.1_degree_grid_land_overlap_new.gpkg"
    )
    grid_land_overlap["id"] = grid_land_overlap["id"].astype(int)
    grid = grid_land_overlap.copy()

    # Load files
    today = datetime.now().strftime("%Y%m%d")
    gefs_folder_path = Path(input_dir) / "NOMADS" / today / "input_files"

    # Output folder
    processed_output_dir_grid = (
        Path(input_dir) / "NOMADS" / today / "output_processed_bygrid"
    )
    os.makedirs(processed_output_dir_grid, exist_ok=True)

    processed_output_dir = (
        Path(input_dir) / "NOMADS" / today / "output_processed"
    )
    os.makedirs(processed_output_dir, exist_ok=True)

    # Apply the adjust_longitude function to each geometry in the DataFrame
    grid_transformed = grid.copy()
    grid_transformed["geometry"] = grid_transformed["geometry"].apply(
        adjust_longitude
    )
    grid_transformed = grid_transformed[
        ["id", "Latitude", "Longitude", "geometry"]
    ]

    # Max and mean rainfall by grid

    # List of statistics to compute
    stats_list = ["mean", "max"]
    grid_list = []
    # Create a loop for running through all GEFS files
    for gefs_file in sorted(os.listdir(gefs_folder_path)):
        if gefs_file.startswith("geprcp"):
            gefs_file_path = Path(gefs_folder_path) / gefs_file
            input_raster = rasterio.open(gefs_file_path)
            array = input_raster.read(1)

            # Compute statistics for grid cells
            summary_stats = zonal_stats(
                grid_transformed,
                array,
                stats=stats_list,
                nodata=-999,  # There's no specificaiton on this, so we invent a number
                all_touched=True,
                affine=input_raster.transform,
            )

            grid_stats = pd.DataFrame(summary_stats)

            # Change values to mm/hr
            if gefs_file[:-5].endswith("06"):
                grid_stats[stats_list] /= 6  # For 6h accumulation data
            else:
                grid_stats[stats_list] /= 24  # For 24h accumulation data

            # Set appropriate date and time information
            forecast_hours = int(
                gefs_file.split(".")[-1][-3:]
            )  # Extract forecast hours from the filename
            release_hour = int(
                gefs_file.split(".")[1][1:3]
            )  # Extract release hour from the filename
            release_date = datetime.now().replace(
                hour=release_hour, minute=0, second=0, microsecond=0
            )  # Set release date
            forecast_date = release_date + timedelta(
                hours=forecast_hours
            )  # Calculate forecast date

            # Merge grid statistics with grid data
            grid_merged = pd.concat([grid_transformed, grid_stats], axis=1)
            # Set appropriate date and time information (modify as per GEFS data format)
            grid_merged["date"] = forecast_date.strftime(
                "%Y%m%d%H"
            )  # Format date as string
            grid_merged = grid_merged[["id", "max", "mean", "date"]]
            grid_list.append(grid_merged)

            # Save the processed data. Name file by time and put in folder 06 or 24 regarding rainfall accumulation.
            grid_date = forecast_date.strftime("%Y%m%d%H")
            if gefs_file[:-5].endswith("06"):
                # Set out dir
                outdir = processed_output_dir_grid / "06"
                os.makedirs(outdir, exist_ok=True)
            else:
                # Set out dir
                outdir = processed_output_dir_grid / "24"
                os.makedirs(outdir, exist_ok=True)

            grid_merged.to_csv(
                outdir / f"{grid_date}_gridstats.csv",
                index=False,
            )

    # Create input rainfall dataset
    for folder in ["06", "24"]:
        list_df = []
        for file in sorted(os.listdir(processed_output_dir_grid / folder)):
            file_path = processed_output_dir_grid / folder / file
            df_aux = pd.read_csv(file_path)
            # Put 0 in nans values
            df_aux = df_aux.fillna(0)
            list_df.append(df_aux)

        final_df = pd.concat(list_df)
        # Convert the "date" column to datetime type
        final_df["date"] = pd.to_datetime(final_df["date"], format="%Y%m%d%H")

        # Separate DataFrames for "mean" and "max" statistics
        df_pivot_mean = final_df.pivot_table(
            index=["id"], columns="date", values="mean"
        )
        df_pivot_max = final_df.pivot_table(
            index=["id"], columns="date", values="max"
        )

        # Flatten the multi-level column index
        df_pivot_mean.columns = [
            f"{col.strftime('%Y-%m-%d %H:%M:%S')}"
            for col in df_pivot_mean.columns
        ]
        df_pivot_max.columns = [
            f"{col.strftime('%Y-%m-%d %H:%M:%S')}"
            for col in df_pivot_max.columns
        ]

        # Reset the index
        df_pivot_mean.reset_index(inplace=True)
        df_pivot_max.reset_index(inplace=True)

        # Save .csv
        outdir = processed_output_dir / folder
        os.makedirs(outdir, exist_ok=True)
        df_pivot_mean.to_csv(
            outdir / str("gridstats_" + "mean" + ".csv"),
            index=False,
        )
        df_pivot_max.to_csv(
            outdir / str("gridstats_" + "max" + ".csv"),
            index=False,
        )

    # Compute 6h and 24h max rainfall
    events = event_metadata.event.to_list()
    time_frame_24 = 4  # in 6 hour periods
    time_frame_6 = 1  # in 6 hour periods

    for stats in ["mean", "max"]:
        df_rainfall_final = pd.DataFrame()
        for event in events:
            # Getting event info
            df_info = event_metadata[event_metadata["event"] == event]

            # End date is the end date of the forecast
            end_date = df_info["end_date"]
            # Start date is starting day of the forecast
            start_date = df_info["start_date"]

            # Loading the data (6h and 24h accumulation)
            df_rainfall_06 = pd.read_csv(
                processed_output_dir
                / "06"
                / str("gridstats_" + stats + ".csv")
            )
            df_rainfall_24 = pd.read_csv(
                processed_output_dir
                / "24"
                / str("gridstats_" + stats + ".csv")
            )

            # Focus on event dates (df_rainfall_06.iloc[3:] and df_rainfall_24.iloc[1:] have the same columns)
            available_dates_t = [
                date
                for date in df_rainfall_24.columns[1:]
                if (pd.to_datetime(date) >= start_date.iloc[0])
                & (pd.to_datetime(date) < end_date.iloc[0])
            ]
            # Restrict to event dates
            df_rainfall_06 = df_rainfall_06[["id"] + available_dates_t]
            df_rainfall_24 = df_rainfall_24[["id"] + available_dates_t]

            # Create new dataframe
            df_mean_rainfall = pd.DataFrame({"id": df_rainfall_24["id"]})

            df_mean_rainfall["rainfall_max_6h"] = (
                df_rainfall_06.iloc[:, 1:]
                .T.rolling(time_frame_6)
                .mean()
                .max(axis=0)
            )

            df_mean_rainfall["rainfall_max_24h"] = (
                df_rainfall_24.iloc[:, 1:]
                .T.rolling(time_frame_24)
                .mean()
                .max(axis=0)
            )
            df_rainfall_single = df_mean_rainfall[
                [
                    "id",
                    "rainfall_max_6h",
                    "rainfall_max_24h",
                ]
            ]
            df_rainfall_single["event"] = event
            df_rainfall_final = pd.concat(
                [df_rainfall_final, df_rainfall_single]
            )
        # Saving everything
        outdir = output_dir / "NOMADS" / today
        os.makedirs(outdir, exist_ok=True)
        df_rainfall_final.to_csv(
            outdir / str("rainfall_data_rw_" + stats + ".csv"), index=False
        )


def create_input_dataset(df_windfield, df_rainfall):
    # Stationaty data
    df_stationary = get_stationary_data_fiji()

    # Focus on Fiji
    fiji_forecast = df_windfield[df_windfield.in_fiji == True]

    # drop nans (explained in the last part of the rainfall_data.ipynb)
    df_rainfall = df_rainfall.dropna()

    # Merge rainfall and windfield data
    variable_data = fiji_forecast.merge(
        df_rainfall,
        left_on=["unique_id", "grid_point_id"],
        right_on=["event", "id"],
    )[
        [
            "grid_point_id",
            "event_id_ecmwf",
            "unique_id",
            "time_init",
            "time_end",
            "wind_speed",
            "track_distance",
            "rainfall_max_6h",
            "rainfall_max_24h",
        ]
    ].sort_values(
        "rainfall_max_6h", ascending=False
    )

    # Merge everything
    input_df = df_stationary.merge(
        variable_data, left_on="grid_point_id", right_on="grid_point_id"
    )[
        [
            "grid_point_id",
            "IWI",
            "total_buildings",
            "with_coast",
            "coast_length",
            "mean_altitude",
            "mean_slope",
            "wind_speed",
            "track_distance",
            "rainfall_max_6h",
            "rainfall_max_24h",
            "event_id_ecmwf",
            "unique_id",
            "time_init",
            "time_end",
        ]
    ].reset_index(
        drop=True
    )

    return input_df
