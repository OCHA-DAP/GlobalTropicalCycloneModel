```python
import getpass
import os
from pathlib import Path

import pandas as pd
import datetime as dt
from datetime import timedelta
from bs4 import BeautifulSoup
import requests
import time
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterstats import zonal_stats

from utils import get_combined_dataset, get_municipality_grids
from input_dataset import create_input_dataset, create_windfield_dataset
```

```python
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
```

## Events of interest: windfield data

```python
# Download real-time data
df_windfield = create_windfield_dataset()
```

```python
# Focus on Fiji
fiji_forecast = df_windfield[df_windfield.in_fiji == True]

# Group the DataFrame by the 'unique_id' column
grouped = fiji_forecast.groupby('unique_id')

# Create a list of Forecasts DataFrames, one for each unique_id
list_forecast = [group for name, group in grouped]
```

I don't know the landfall. All I have are some windforecasts. So the time span for the adquisition of the rainfall data is just the time span of the forecast.

```python
event_metadata = pd.DataFrame({
    'event':fiji_forecast.unique_id.unique(),
    'start_date':[df.time_init.iloc[0] for df in list_forecast],
    'end_date': [df.time_end.iloc[0] for df in list_forecast]
})
```

## Forecast data dowloading


We want to access every precipitation (prcp) file inside GEFS Bias-Corrected ensemble

Check https://www.nco.ncep.noaa.gov/pmb/products/gens/ for more information of what the name of every file means.

Also, there are 2 type of files:

- 06h ...
- 24h ...

that corresponds to xxh (06h,24h) rainfall accumulation.

Some info:

- Resolution: 0.5 degrees (v!=12.0) https://www.nco.ncep.noaa.gov/pmb/products/gens/
- Units:
- Forecast reliability: avg 40% in 72h https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs/faq.php

```python
import requests
from datetime import datetime
import os
from bs4 import BeautifulSoup

def download_gefs_data(download_folder):
    # Calculate the current date
    today = datetime.now().strftime("%Y%m%d")

    # Generate folder names for 18, 12, 06, and 00 h
    folder_names = [f"{today}/18",f"{today}/12", f"{today}/06", f"{today}/00"]

    for folder in folder_names:
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.{folder}/prcp_bc_gb2/"
        response = requests.get(url)

        if response.status_code == 200:
            print(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links  = [a['href'] for a in soup.find_all('a') if a['href'].startswith("geprcp.t") and a['href'][:-7].endswith("bc_")]

            if links:
                for link in links:
                    link_url = url + link
                    file_name = link.split("/")[-1]
                    file_path = os.path.join(download_folder, file_name)

                    print(f"Downloading {link_url}")
                    download_response = requests.get(link_url)

                    with open(file_path, 'wb') as f:
                        f.write(download_response.content)
                return
            break
    print("prcp_gb2 not found for today.")


# Example usage:
today = datetime.now().strftime("%Y%m%d")
download_folder = Path(input_dir) / "NOMADS" / today / "input_files"
os.makedirs(download_folder, exist_ok=True)
download_gefs_data(download_folder)
```

## Forecast data analysis

```python
# Load grid
grid_land_overlap = gpd.read_file(grid_input / "fji_0.1_degree_grid_land_overlap_new.gpkg")
grid_land_overlap["id"] = grid_land_overlap["id"].astype(int)
grid = grid_land_overlap.copy()

# Load files
today = datetime.now().strftime("%Y%m%d")
gefs_folder_path = Path(input_dir) / "NOMADS" / today / "input_files"

# Output folder
processed_output_dir_grid = Path(input_dir) / "NOMADS" / today / "output_processed_bygrid"
os.makedirs(processed_output_dir_grid, exist_ok=True)

processed_output_dir = Path(input_dir) / "NOMADS" / today / "output_processed"
os.makedirs(processed_output_dir, exist_ok=True)
```

Some checks

```python
# Let's see what are the values for nodata
gefs_file = os.listdir(gefs_folder_path)[0]
gefs_file_path = Path(gefs_folder_path) / gefs_file
with rasterio.open(gefs_file_path) as src:
    nodata_value = src.nodatavals[0]  # Get the NoData value for the first band (assuming a single band)
    print(f"NoData value: {nodata_value}")
```

```python
# Let's see the crs.. in a intuitive way, this are the limits
input_raster = rasterio.open(gefs_file_path)
input_raster.bounds
```

Since we are in Fiji and we want information for >180 degree longitudes and the CRS of our grid is one with that goes from 0 to 360 in longitude.. we have to modify the grid crs. But it's not that simple. Let's do it by modifing manually the geometry

```python
grid_transformed = grid.copy()

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

# Apply the adjust_longitude function to each geometry in the DataFrame
grid_transformed["geometry"] = grid_transformed["geometry"].apply(adjust_longitude)
grid_transformed = grid_transformed[['id', 'Latitude', 'Longitude','geometry']]
grid_transformed
```

Now lets get the max and mean rainfall by grid

```python
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
                    nodata=-999, # There's no specificaiton on this, so we invent a number
                    all_touched=True,
                    affine=input_raster.transform,
                )

        grid_stats = pd.DataFrame(summary_stats)

        # Change values to mm/hr
        if gefs_file[:-5].endswith('06'):
            grid_stats[stats_list] /= 6 # For 6h accumulation data
        else:
            grid_stats[stats_list] /= 24 # For 24h accumulation data

        # Set appropriate date and time information
        forecast_hours = int(gefs_file.split(".")[-1][-3:])  # Extract forecast hours from the filename
        release_hour = int(gefs_file.split(".")[1][1:3])  # Extract release hour from the filename
        release_date = datetime.now().replace(hour=release_hour, minute=0, second=0, microsecond=0)  # Set release date
        forecast_date = release_date + timedelta(hours=forecast_hours)  # Calculate forecast date

        # Merge grid statistics with grid data
        grid_merged = pd.concat([grid_transformed, grid_stats], axis=1)
        # Set appropriate date and time information (modify as per GEFS data format)
        grid_merged["date"] = forecast_date.strftime("%Y%m%d%H")  # Format date as string
        grid_merged = grid_merged[['id', 'max', 'mean', 'date']]
        grid_list.append(grid_merged)

        # Save the processed data. Name file by time and put in folder 06 or 24 regarding rainfall accumulation.
        grid_date = forecast_date.strftime("%Y%m%d%H")
        if gefs_file[:-5].endswith('06'):
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
```

Obs: we have nans values! Thats ok, we dont have data for every cell. As we did with the model, we put 0s in here.

```python
grid_list[1][grid_list[0].isna().any(axis=1)].head()
```

Let's create an output more similar to the one of the Philippines.

```python
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
    df_pivot_mean = final_df.pivot_table(index=["id"], columns="date", values="mean")
    df_pivot_max = final_df.pivot_table(index=["id"], columns="date", values="max")

    # Flatten the multi-level column index
    df_pivot_mean.columns = [f"{col.strftime('%Y-%m-%d %H:%M:%S')}" for col in df_pivot_mean.columns]
    df_pivot_max.columns = [f"{col.strftime('%Y-%m-%d %H:%M:%S')}" for col in df_pivot_max.columns]

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
```

## Compute max Rainfall in 6H and in 24H time windows

```python
events = event_metadata.event.to_list()
time_frame_24 = 4  # in 6 hour periods
time_frame_6 = 1  # in 6 hour periods

for stats in ["mean", "max"]:
    df_rainfall_final = pd.DataFrame()
    for event in events:
        # Getting event info
        df_info = event_metadata[event_metadata["event"] == event]

        # End date is the end date of the forecast
        end_date = df_info['end_date']
        # Start date is starting day of the forecast
        start_date = df_info['start_date']

        # Loading the data (6h and 24h accumulation)
        df_rainfall_06 = pd.read_csv(
            processed_output_dir / "06" /str("gridstats_" + stats + ".csv")
        )
        df_rainfall_24 = pd.read_csv(
            processed_output_dir / "24" /str("gridstats_" + stats + ".csv")
        )

        # Focus on event dates (df_rainfall_06.iloc[3:] and df_rainfall_24.iloc[1:] have the same columns)
        available_dates_t = [
            date for date in df_rainfall_24.columns[1:]
            if (pd.to_datetime(date) >= start_date.iloc[0]) & (pd.to_datetime(date) < end_date.iloc[0])
        ]
        # Restrict to event dates
        df_rainfall_06 = df_rainfall_06[['id']+available_dates_t]
        df_rainfall_24 = df_rainfall_24[['id']+available_dates_t]

        # Create new dataframe
        df_mean_rainfall = pd.DataFrame(
                {"id": df_rainfall_24["id"]}
        )

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
        df_rainfall_final = pd.concat([df_rainfall_final, df_rainfall_single])
    # Saving everything
    outdir = output_dir / "NOMADS" / today
    os.makedirs(outdir, exist_ok=True)
    df_rainfall_final.to_csv(
        outdir / str("rainfall_data_rw_" + stats + ".csv"), index=False
    )
```

```python
import matplotlib.pyplot as plt
# Loading the data
df_rainfall_24 = pd.read_csv(
    processed_output_dir / "24" /str("gridstats_" + stats + ".csv")
)
example_mean = df_rainfall_24.iloc[:, 1:].T.rolling(time_frame_24).mean().max(axis=0)
example_median = df_rainfall_24.iloc[:, 1:].T.rolling(time_frame_24).median().max(axis=0)

plt.plot(example_mean, example_median, 'o')
plt.title('Max rainfall value over multiple windows')
plt.xlabel('Mean rainfall value of 24h window')
plt.ylabel('Median rainfall value of 24h window')
plt.grid()
plt.show()
```

## Merge rainfall data and windfield data

```python
filename = output_dir / "NOMADS"/ today /"rainfall_data_rw_mean.csv"
df_rainfall = pd.read_csv(filename)
fiji_forecast.merge(df_rainfall, left_on=['unique_id','grid_point_id'], right_on=['event', 'id'])[
    ['grid_point_id',
    'event',
    'wind_speed',
    'track_distance',
    'rainfall_max_6h',
    'rainfall_max_24h',
    'time_init',
    'time_end',
    'in_fiji',
    'geometry']].sort_values('rainfall_max_6h', ascending=False)

```

There are some nans: this is because for the last event (usually), since the forecast is up to 16 days and the last event is on the day 16, we have just 1 datetime to compute the rolling. This is depricable because events 16 days from now are not significative and we are going to keep just the ones in the near future (up tp 120h aprox)

```python
df_rainfall[df_rainfall.isna().any(axis=1)] # Its just the last dataset...
```

To compare, let's see the values of rainfall for the historical data.

```python
from utils import get_training_dataset_complete
df_complete = get_training_dataset_complete()
df_complete[['rainfall_max_6h','rainfall_max_24h']].sort_values('rainfall_max_6h', ascending=False)
```
