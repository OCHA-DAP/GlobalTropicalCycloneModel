---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: env1
    language: python
    name: python3
---

```python
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from shapely.geometry import LineString, Point
from climada.hazard import Centroids, TCTracks, TropCyclone, Hazard
import xarray as xr
import datetime
from pykrige.ok import OrdinaryKriging
from utils import get_stationary_data_fiji
```

```python
output_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/02_housing_damage/output/"
)

# Load grid and stationary data
df = get_stationary_data_fiji()
grids = gpd.read_file(output_dir / "fji_0.1_degree_grid_land_overlap_new.gpkg")
grids.geometry = grids.geometry.to_crs(grids.crs).centroid
df_stationary = df.merge(grids, right_on='id', left_on='grid_point_id').drop(['index', 'id'], axis=1)
```

### Load forecasts

```python
# Folder path
folder_path = '/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/Forecasts_Fiji/forecasts/'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Get the full file paths
csv_file_paths = [os.path.join(folder_path, filename) for filename in csv_files]
```

```python
csv_files
```

```python
# Evan typhoon 1
df_forecast = pd.read_csv(csv_file_paths[0], header=6, engine='python').iloc[1:][
    ["Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']",'Latitude', 'Longitude', 'MeanWind', 'Pressure', 'PressureOCI', 'Category','RadiusOCI', 'RadiusMaxWinds']].rename(
        {"Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']":'forecast_time'},
        axis=1
    )

# Create a GeoDataFrame from the DataFrame with latitude and longitude
geometry = [Point(xy) for xy in zip(df_forecast['Longitude'], df_forecast['Latitude'])]
df_forecast = gpd.GeoDataFrame(df_forecast, geometry=geometry)

# Convert 'forecast_time' to datetime format
df_forecast['forecast_time'] = pd.to_datetime(df_forecast['forecast_time'])

# Sort the DataFrame by 'forecast_time' in ascending order
df_forecast = df_forecast.sort_values(by='forecast_time')

# Calculate the time step by subtracting consecutive 'forecast_time' values
df_forecast['time_step'] = df_forecast['forecast_time'].diff().dt.total_seconds() / 3600
df_forecast['time_step'].fillna(0, inplace=True)

geometry2 = [Point(xy) for xy in zip(df_forecast2['Longitude'], df_forecast2['Latitude'])]
df_forecast2 = gpd.GeoDataFrame(df_forecast2, geometry=geometry2)

df_forecast
```

### Calculate complete windfield

```python
# The idea is to compute the windfield in every grid cell. Not just in the locations where we have points.
fig, ax = plt.subplots(1,1)
df_forecast.plot(column='MeanWind', cmap='coolwarm', markersize=20, legend=True, ax=ax)
#df_forecast2.plot(column='MeanWind', cmap='coolwarm', markersize=20, legend=True, ax=ax)
grids.plot(ax=ax)

plt.show()
```

Let's load a custom track in CLIMADA

```python
def adjust_tracks(forcast_df):
    track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', 0.514444*forcast_df.MeanWind.values),
            'environmental_pressure': ('time', forcast_df.Pressure.values),
            'central_pressure': ('time',forcast_df.Pressure.values),
            'lat': ('time',forcast_df.Latitude.values),
            'lon': ('time', forcast_df.Longitude.values),
            'radius_max_wind':('time', forcast_df.RadiusMaxWinds.values),
            'radius_oci':('time',forcast_df.RadiusOCI.values),
            'time_step':('time',np.full_like(forcast_df.time_step.values, 3, dtype=float)),
        },
        coords={
            'time': forcast_df.forecast_time.values,
        },
        attrs={
            'max_sustained_wind_unit': 'm/s',
            'central_pressure_unit': 'mb',
            #'name': forcast_df.name,
            #'sid': forcast_df.sid,#+str(forcast_df.ensemble_number),
            #'orig_event_flag': forcast_df.orig_event_flag,
            #'data_provider': forcast_df.data_provider,
            #'id_no': forcast_df.id_no,
            #'basin': forcast_df.basin,
            'category': forcast_df.Category.iloc[0],
        }
    )
    track = track.set_coords(['lat', 'lon'])
    return track

tracks=TCTracks()
tracks.data=[adjust_tracks(df_forecast)]
```

```python
cent = Centroids.from_geodataframe(grids)

tc = TropCyclone.from_tracks(
    tracks, centroids=cent, store_windfields=True, intensity_thres=0
)
```

Let's compute the track distance of every point of the grid cell.

```python
#Track distance
points = gpd.points_from_xy(df_forecast.Longitude, df_forecast.Latitude)
tc_track_line = LineString(points)
# Get the track distance
DEG_TO_KM = 111.1
tc_track_distance = grids["geometry"].apply(
    lambda point: point.distance(tc_track_line) * DEG_TO_KM
)
```
