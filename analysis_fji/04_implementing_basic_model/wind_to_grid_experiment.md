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

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_36495/386341765.py:9: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      grids.geometry = grids.geometry.to_crs(grids.crs).centroid


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




    ['20121213T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201217T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121215T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201216T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121214T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201214T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121216T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200405T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200404T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201215T060000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121217T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200408T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200407T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121215T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20121215T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200407T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201211T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121214T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200406T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200406T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121214T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20121216T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200406T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201212T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121216T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200407T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121217T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201214T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201214T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201215T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201215T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201217T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121213T000000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20200406T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121213T180000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201217T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201216T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200407T180000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201216T000000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121216T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201214T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200408T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200404T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200408T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200405T120000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200405T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121217T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201215T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200409T000000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20200405T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20121215T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201217T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20201219T180000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20121213T120000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20121214T060000Z_Official_Forecast_Track_1213_04F_EVAN.csv',
     '20201216T120000Z_Official_Forecast_Track_2021_02F_YASA.csv',
     '20200408T060000Z_Official_Forecast_Track_1920_12F_Harold.csv',
     '20201218T000000Z_Official_Forecast_Track_2021_02F_YASA.csv']




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>forecast_time</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MeanWind</th>
      <th>Pressure</th>
      <th>PressureOCI</th>
      <th>Category</th>
      <th>RadiusOCI</th>
      <th>RadiusMaxWinds</th>
      <th>geometry</th>
      <th>time_step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2012-12-13 06:00:00+00:00</td>
      <td>-14.00000</td>
      <td>188.80000</td>
      <td>65.0</td>
      <td>975.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>200.0</td>
      <td>20.0</td>
      <td>POINT (188.800 -14.000)</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-12-13 12:00:00+00:00</td>
      <td>-14.00498</td>
      <td>189.09286</td>
      <td>70.0</td>
      <td>967.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (189.093 -14.005)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-12-13 18:00:00+00:00</td>
      <td>-14.01747</td>
      <td>188.93646</td>
      <td>70.0</td>
      <td>967.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (188.936 -14.017)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-12-14 00:00:00+00:00</td>
      <td>-14.02999</td>
      <td>188.77873</td>
      <td>75.0</td>
      <td>962.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (188.779 -14.030)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012-12-14 06:00:00+00:00</td>
      <td>-14.08758</td>
      <td>188.26548</td>
      <td>75.0</td>
      <td>964.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (188.265 -14.088)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2012-12-14 12:00:00+00:00</td>
      <td>-14.14563</td>
      <td>187.75121</td>
      <td>75.0</td>
      <td>964.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (187.751 -14.146)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012-12-14 18:00:00+00:00</td>
      <td>-14.35967</td>
      <td>186.90250</td>
      <td>80.0</td>
      <td>961.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (186.903 -14.360)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2012-12-15 00:00:00+00:00</td>
      <td>-14.57499</td>
      <td>186.04873</td>
      <td>80.0</td>
      <td>961.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (186.049 -14.575)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2012-12-15 06:00:00+00:00</td>
      <td>-14.90298</td>
      <td>185.07229</td>
      <td>85.0</td>
      <td>961.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (185.072 -14.903)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2012-12-15 12:00:00+00:00</td>
      <td>-15.22902</td>
      <td>184.10162</td>
      <td>85.0</td>
      <td>961.0</td>
      <td>1000.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (184.102 -15.229)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2012-12-15 18:00:00+00:00</td>
      <td>-15.56450</td>
      <td>183.10271</td>
      <td>90.0</td>
      <td>957.0</td>
      <td>1000.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (183.103 -15.565)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2012-12-16 00:00:00+00:00</td>
      <td>-15.89999</td>
      <td>182.10373</td>
      <td>90.0</td>
      <td>957.0</td>
      <td>1000.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (182.104 -15.900)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2012-12-16 06:00:00+00:00</td>
      <td>-16.26998</td>
      <td>181.25248</td>
      <td>90.0</td>
      <td>956.0</td>
      <td>1000.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (181.252 -16.270)</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



### Calculate complete windfield


```python
# The idea is to compute the windfield in every grid cell. Not just in the locations where we have points.
fig, ax = plt.subplots(1,1)
df_forecast.plot(column='MeanWind', cmap='coolwarm', markersize=20, legend=True, ax=ax)
#df_forecast2.plot(column='MeanWind', cmap='coolwarm', markersize=20, legend=True, ax=ax)
grids.plot(ax=ax)

plt.show()
```



![png](wind_to_grid_experiment_files/wind_to_grid_experiment_7_0.png)



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

    2023-10-30 17:17:22,779 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb Cell 13 line 3
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#Y103sZmlsZQ%3D%3D?line=0'>1</a> cent = Centroids.from_geodataframe(grids)
    ----> <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#Y103sZmlsZQ%3D%3D?line=2'>3</a> tc = TropCyclone.from_tracks(
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#Y103sZmlsZQ%3D%3D?line=3'>4</a>     tracks, centroids=cent, store_windfields=True, intensity_thres=0
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#Y103sZmlsZQ%3D%3D?line=4'>5</a> )


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/climada/hazard/trop_cyclone.py:321, in TropCyclone.from_tracks(cls, tracks, centroids, pool, model, ignore_distance_to_coast, store_windfields, metric, intensity_thres, max_latitude, max_dist_inland_km, max_dist_eye_km, max_memory_gb)
        318         LOGGER.info("Progress: %d%%", perc)
        319         last_perc = perc
        320     tc_haz_list.append(
    --> 321         cls.from_single_track(track, centroids, coastal_idx,
        322                               model=model, store_windfields=store_windfields,
        323                               metric=metric, intensity_thres=intensity_thres,
        324                               max_dist_eye_km=max_dist_eye_km,
        325                               max_memory_gb=max_memory_gb))
        326 if last_perc < 100:
        327     LOGGER.info("Progress: 100%")


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/climada/hazard/trop_cyclone.py:578, in TropCyclone.from_single_track(cls, track, centroids, coastal_idx, model, store_windfields, metric, intensity_thres, max_dist_eye_km, max_memory_gb)
        576 new_haz.event_id = np.array([1])
        577 new_haz.frequency = np.array([1])
    --> 578 new_haz.event_name = [track.sid]
        579 new_haz.fraction = sparse.csr_matrix(new_haz.intensity.shape)
        580 # store first day of track as date


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/common.py:278, in AttrAccessMixin.__getattr__(self, name)
        276         with suppress(KeyError):
        277             return source[name]
    --> 278 raise AttributeError(
        279     f"{type(self).__name__!r} object has no attribute {name!r}"
        280 )


    AttributeError: 'Dataset' object has no attribute 'sid'


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
