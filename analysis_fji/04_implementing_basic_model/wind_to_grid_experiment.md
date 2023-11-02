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
#from climada.entity import PointData, ImpactModel
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

# Evan typhoon 2
df_forecast2 = pd.read_csv(csv_file_paths[2], header=6, engine='python').iloc[1:][
    ["Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']",'Latitude', 'Longitude', 'MeanWind', 'Pressure', 'PressureOCI','Category', 'RadiusOCI']].rename(
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




```python
pd.read_csv(csv_file_paths[1], header=6, engine='python').iloc[2:]

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
      <th>Time[fmt=yyyy-MM-dd'T'HH:mm:ss'Z']</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Symbol</th>
      <th>Category</th>
      <th>Pressure</th>
      <th>PressureOCI</th>
      <th>RadiusOCI</th>
      <th>Radius1000hPa</th>
      <th>RadiusMaxWinds</th>
      <th>...</th>
      <th>HowMaxWindRadius</th>
      <th>HowGaleRadius</th>
      <th>HowStormRadius</th>
      <th>HowHurricaneRadius</th>
      <th>UncMaxWindSpeed</th>
      <th>HowMaxWindSpeed</th>
      <th>HowGust</th>
      <th>EyeRadius</th>
      <th>UncEyeRadius</th>
      <th>HowEyeRadius</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2020-12-17T08:00:00Z</td>
      <td>-16.80238</td>
      <td>179.07420</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>909.3</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-12-17T12:00:00Z</td>
      <td>-17.20714</td>
      <td>179.62262</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>914.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-12-17T14:00:00Z</td>
      <td>-17.44048</td>
      <td>179.84406</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>916.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-12-17T18:00:00Z</td>
      <td>-17.90714</td>
      <td>180.28690</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>920.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-12-17T20:00:00Z</td>
      <td>-18.14048</td>
      <td>180.50833</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>921.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-12-18T00:00:00Z</td>
      <td>-18.60714</td>
      <td>180.95119</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>923.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-12-18T06:00:00Z</td>
      <td>-19.32500</td>
      <td>181.33690</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>928.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-12-18T12:00:00Z</td>
      <td>-20.04286</td>
      <td>181.72262</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>927.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020-12-18T18:00:00Z</td>
      <td>-20.68929</td>
      <td>181.84404</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>932.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-12-19T00:00:00Z</td>
      <td>-21.33571</td>
      <td>181.96548</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>932.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2020-12-19T06:00:00Z</td>
      <td>-21.89821</td>
      <td>181.82976</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>936.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2020-12-19T12:00:00Z</td>
      <td>-22.46071</td>
      <td>181.69405</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>941.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2020-12-19T18:00:00Z</td>
      <td>-22.95893</td>
      <td>181.42976</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>951.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2020-12-20T00:00:00Z</td>
      <td>-23.45714</td>
      <td>181.16548</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>955.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2020-12-20T06:00:00Z</td>
      <td>-23.89176</td>
      <td>180.85202</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>963.0</td>
      <td>1004.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 57 columns</p>
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



![png](wind_to_grid_experiment_files/wind_to_grid_experiment_8_0.png)



Let's load a custom track in CLIMADA


```python
import datetime
dset = xr.Dataset(
    dict(
        intensity=(
            ["time", "latitude", "longitude"],
            [[[0, 1, 2], [3, 4, 5]]],
        )
    ),
    dict(
        time=[datetime.datetime(2000, 1, 1)],
        latitude=[0, 1],
        longitude=[0, 1, 2],
    ),
)
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:    (time: 13, latitude: 13, longitude: 13)
Coordinates:
  * time       (time) object &#x27;2012-12-13T06:00:00Z&#x27; ... &#x27;2012-12-16T06:00:00Z&#x27;
  * latitude   (latitude) float64 -16.27 -15.9 -15.56 ... -14.02 -14.0 -14.0
  * longitude  (longitude) float64 181.3 182.1 183.1 184.1 ... 188.8 188.9 189.1
Data variables:
    intensity  (time, latitude, longitude) float64 nan nan nan ... nan nan nan
    Pressure   (time, latitude, longitude) float64 nan nan nan ... nan nan nan
    Category   (time, latitude, longitude) float64 nan nan nan ... nan nan nan
    geometry   (time, latitude, longitude) object nan nan nan ... nan nan nan</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-26c94744-f41c-4475-b81f-c483fb65bfbd' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-26c94744-f41c-4475-b81f-c483fb65bfbd' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 13</li><li><span class='xr-has-index'>latitude</span>: 13</li><li><span class='xr-has-index'>longitude</span>: 13</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-aed33ddf-3c1a-428c-b7e1-9f82469a8914' class='xr-section-summary-in' type='checkbox'  checked><label for='section-aed33ddf-3c1a-428c-b7e1-9f82469a8914' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;2012-12-13T06:00:00Z&#x27; ... &#x27;2012...</div><input id='attrs-844ce505-7162-4e91-b747-e57e71425799' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-844ce505-7162-4e91-b747-e57e71425799' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-24ae16e2-4de9-42a2-a12c-b15a6c8a9760' class='xr-var-data-in' type='checkbox'><label for='data-24ae16e2-4de9-42a2-a12c-b15a6c8a9760' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2012-12-13T06:00:00Z&#x27;, &#x27;2012-12-13T12:00:00Z&#x27;, &#x27;2012-12-13T18:00:00Z&#x27;,
       &#x27;2012-12-14T00:00:00Z&#x27;, &#x27;2012-12-14T06:00:00Z&#x27;, &#x27;2012-12-14T12:00:00Z&#x27;,
       &#x27;2012-12-14T18:00:00Z&#x27;, &#x27;2012-12-15T00:00:00Z&#x27;, &#x27;2012-12-15T06:00:00Z&#x27;,
       &#x27;2012-12-15T12:00:00Z&#x27;, &#x27;2012-12-15T18:00:00Z&#x27;, &#x27;2012-12-16T00:00:00Z&#x27;,
       &#x27;2012-12-16T06:00:00Z&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>latitude</span></div><div class='xr-var-dims'>(latitude)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-16.27 -15.9 -15.56 ... -14.0 -14.0</div><input id='attrs-ae71b936-a0a3-4496-ba7c-23962d954a10' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ae71b936-a0a3-4496-ba7c-23962d954a10' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4012b546-cd1d-43c7-80f9-6cfe81e0c871' class='xr-var-data-in' type='checkbox'><label for='data-4012b546-cd1d-43c7-80f9-6cfe81e0c871' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-16.26998, -15.89999, -15.5645 , -15.22902, -14.90298, -14.57499,
       -14.35967, -14.14563, -14.08758, -14.02999, -14.01747, -14.00498,
       -14.     ])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>longitude</span></div><div class='xr-var-dims'>(longitude)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>181.3 182.1 183.1 ... 188.9 189.1</div><input id='attrs-d05704e6-a92a-49e5-b2ab-9fa55081aa75' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d05704e6-a92a-49e5-b2ab-9fa55081aa75' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f700fa2d-43ce-44f6-a8d3-243dcf28a7bb' class='xr-var-data-in' type='checkbox'><label for='data-f700fa2d-43ce-44f6-a8d3-243dcf28a7bb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([181.25248, 182.10373, 183.10271, 184.10162, 185.07229, 186.04873,
       186.9025 , 187.75121, 188.26548, 188.77873, 188.8    , 188.93646,
       189.09286])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-32c8f542-4643-48aa-a580-e83bf9e4690a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-32c8f542-4643-48aa-a580-e83bf9e4690a' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>intensity</span></div><div class='xr-var-dims'>(time, latitude, longitude)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>nan nan nan nan ... nan nan nan nan</div><input id='attrs-e77396a3-421e-4622-ac51-9746285bf5ad' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e77396a3-421e-4622-ac51-9746285bf5ad' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-82bb7e29-c959-4731-9b2d-af451793ed8d' class='xr-var-data-in' type='checkbox'><label for='data-82bb7e29-c959-4731-9b2d-af451793ed8d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., 65., nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, 70.],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
...
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, 90., nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[90., nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Pressure</span></div><div class='xr-var-dims'>(time, latitude, longitude)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>nan nan nan nan ... nan nan nan nan</div><input id='attrs-3780475e-9ef9-4f8a-b3e4-09e42e23fcc9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3780475e-9ef9-4f8a-b3e4-09e42e23fcc9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cda818a8-df75-402c-8fbc-16728eaa2418' class='xr-var-data-in' type='checkbox'><label for='data-cda818a8-df75-402c-8fbc-16728eaa2418' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        ...,
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ..., 975.,  nan,  nan]],

       [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        ...,
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan, 967.],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan]],

       [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        ...,
...
        ...,
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan]],

       [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan, 957.,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        ...,
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan]],

       [[956.,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        ...,
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan],
        [ nan,  nan,  nan, ...,  nan,  nan,  nan]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Category</span></div><div class='xr-var-dims'>(time, latitude, longitude)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>nan nan nan nan ... nan nan nan nan</div><input id='attrs-e239c031-1fcf-4440-9a21-609096a6126f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e239c031-1fcf-4440-9a21-609096a6126f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-39fa0bb1-ea63-4127-870a-d257522347bc' class='xr-var-data-in' type='checkbox'><label for='data-39fa0bb1-ea63-4127-870a-d257522347bc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ...,  3., nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan,  3.],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
...
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan,  4., nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[ 4., nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>geometry</span></div><div class='xr-var-dims'>(time, latitude, longitude)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>nan nan nan nan ... nan nan nan nan</div><input id='attrs-b5c7844f-6de6-4ab8-85c2-22b86519e5c6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b5c7844f-6de6-4ab8-85c2-22b86519e5c6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-edb07d3a-412c-4c0e-94d6-e717ddc3022b' class='xr-var-data-in' type='checkbox'><label for='data-edb07d3a-412c-4c0e-94d6-e717ddc3022b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., &lt;POINT (188.8 -14)&gt;, nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, &lt;POINT (189.093 -14.005)&gt;],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
...
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[nan, nan, nan, ..., nan, nan, nan],
        [nan, &lt;POINT (182.104 -15.9)&gt;, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]],

       [[&lt;POINT (181.252 -16.27)&gt;, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        ...,
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan],
        [nan, nan, nan, ..., nan, nan, nan]]], dtype=object)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1d49b7a2-b6bb-4d21-b70d-5e420ae16b9d' class='xr-section-summary-in' type='checkbox'  ><label for='section-1d49b7a2-b6bb-4d21-b70d-5e420ae16b9d' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-faf70108-7626-42e6-bdcc-b3e53738d796' class='xr-index-data-in' type='checkbox'/><label for='index-faf70108-7626-42e6-bdcc-b3e53738d796' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;2012-12-13T06:00:00Z&#x27;, &#x27;2012-12-13T12:00:00Z&#x27;, &#x27;2012-12-13T18:00:00Z&#x27;,
       &#x27;2012-12-14T00:00:00Z&#x27;, &#x27;2012-12-14T06:00:00Z&#x27;, &#x27;2012-12-14T12:00:00Z&#x27;,
       &#x27;2012-12-14T18:00:00Z&#x27;, &#x27;2012-12-15T00:00:00Z&#x27;, &#x27;2012-12-15T06:00:00Z&#x27;,
       &#x27;2012-12-15T12:00:00Z&#x27;, &#x27;2012-12-15T18:00:00Z&#x27;, &#x27;2012-12-16T00:00:00Z&#x27;,
       &#x27;2012-12-16T06:00:00Z&#x27;],
      dtype=&#x27;object&#x27;, name=&#x27;time&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>latitude</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-3de41bbb-beaa-4db1-814c-9ba8c7cdd84d' class='xr-index-data-in' type='checkbox'/><label for='index-3de41bbb-beaa-4db1-814c-9ba8c7cdd84d' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([-16.26998, -15.89999,  -15.5645, -15.22902, -14.90298, -14.57499,
       -14.35967, -14.14563, -14.08758, -14.02999, -14.01747, -14.00498,
           -14.0],
      dtype=&#x27;float64&#x27;, name=&#x27;latitude&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>longitude</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-92134636-59a9-4d5f-bc79-32fff39a699c' class='xr-index-data-in' type='checkbox'/><label for='index-92134636-59a9-4d5f-bc79-32fff39a699c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([181.25248, 182.10373, 183.10271, 184.10162, 185.07229, 186.04873,
        186.9025, 187.75121, 188.26548, 188.77873,     188.8, 188.93646,
       189.09286],
      dtype=&#x27;float64&#x27;, name=&#x27;longitude&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c11d0111-f726-47f7-bfb0-ddd1ea0da768' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-c11d0111-f726-47f7-bfb0-ddd1ea0da768' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




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



```python
# Convert to xarray Dataset
windspeed = df_forecast['MeanWind'].values
xarr_data = xr.Dataset(
    {'intensity': (['time', 'latitude', 'longitude'], windspeed)},
    coords={
        'time': [pd.to_datetime(t) for t in df_forecast['forecast_time']],
        'latitude': df_forecast['Latitude'],
        'longitude': df_forecast['Longitude'],
    },
)

# Set metadata
hazard_type = "Tropical Cyclone"
intensity_unit = "km/h"
crs = "EPSG:4326"  # You can specify the correct CRS if different

# Create CLIMADA Hazard object
hazard = Hazard.from_xarray_raster(
    xarr_data,
    hazard_type=hazard_type,
    intensity_unit=intensity_unit,
    crs=crs
)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/variable.py:129, in as_variable(obj, name)
        128 try:
    --> 129     obj = Variable(*obj)
        130 except (TypeError, ValueError) as error:


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/variable.py:364, in Variable.__init__(self, dims, data, attrs, encoding, fastpath)
        345 """
        346 Parameters
        347 ----------
       (...)
        362     unrecognized encoding items.
        363 """
    --> 364 super().__init__(
        365     dims=dims, data=as_compatible_data(data, fastpath=fastpath), attrs=attrs
        366 )
        368 self._encoding = None


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/namedarray/core.py:252, in NamedArray.__init__(self, dims, data, attrs)
        251 self._data = data
    --> 252 self._dims = self._parse_dimensions(dims)
        253 self._attrs = dict(attrs) if attrs else None


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/namedarray/core.py:480, in NamedArray._parse_dimensions(self, dims)
        479 if len(dims) != self.ndim:
    --> 480     raise ValueError(
        481         f"dimensions {dims} must have the same length as the "
        482         f"number of data dimensions, ndim={self.ndim}"
        483     )
        484 return dims


    ValueError: dimensions ('time', 'latitude', 'longitude') must have the same length as the number of data dimensions, ndim=1


    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)

    /Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb Cell 10 line 3
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=0'>1</a> # Convert to xarray Dataset
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=1'>2</a> windspeed = df_forecast['MeanWind'].values
    ----> <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=2'>3</a> xarr_data = xr.Dataset(
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=3'>4</a>     {'intensity': (['time', 'latitude', 'longitude'], windspeed)},
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=4'>5</a>     coords={
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=5'>6</a>         'time': [pd.to_datetime(t) for t in df_forecast['forecast_time']],
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=6'>7</a>         'latitude': df_forecast['Latitude'],
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=7'>8</a>         'longitude': df_forecast['Longitude'],
          <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=8'>9</a>     },
         <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=9'>10</a> )
         <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=11'>12</a> # Set metadata
         <a href='vscode-notebook-cell:/Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI%20clean/analysis_fji/04_implementing_basic_model/wind_to_grid_experiment.ipynb#X53sZmlsZQ%3D%3D?line=12'>13</a> hazard_type = "Tropical Cyclone"


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/dataset.py:696, in Dataset.__init__(self, data_vars, coords, attrs)
        693 if isinstance(coords, Dataset):
        694     coords = coords._variables
    --> 696 variables, coord_names, dims, indexes, _ = merge_data_and_coords(
        697     data_vars, coords
        698 )
        700 self._attrs = dict(attrs) if attrs is not None else None
        701 self._close = None


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/dataset.py:425, in merge_data_and_coords(data_vars, coords)
        421     coords = create_coords_with_default_indexes(coords, data_vars)
        423 # exclude coords from alignment (all variables in a Coordinates object should
        424 # already be aligned together) and use coordinates' indexes to align data_vars
    --> 425 return merge_core(
        426     [data_vars, coords],
        427     compat="broadcast_equals",
        428     join="outer",
        429     explicit_coords=tuple(coords),
        430     indexes=coords.xindexes,
        431     priority_arg=1,
        432     skip_align_args=[1],
        433 )


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/merge.py:718, in merge_core(objects, compat, join, combine_attrs, priority_arg, explicit_coords, indexes, fill_value, skip_align_args)
        715 for pos, obj in skip_align_objs:
        716     aligned.insert(pos, obj)
    --> 718 collected = collect_variables_and_indexes(aligned, indexes=indexes)
        719 prioritized = _get_priority_vars_and_indexes(aligned, priority_arg, compat=compat)
        720 variables, out_indexes = merge_collected(
        721     collected, prioritized, compat=compat, combine_attrs=combine_attrs
        722 )


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/merge.py:358, in collect_variables_and_indexes(list_of_mappings, indexes)
        355     indexes_.pop(name, None)
        356     append_all(coords_, indexes_)
    --> 358 variable = as_variable(variable, name=name)
        359 if name in indexes:
        360     append(name, variable, indexes[name])


    File ~/anaconda3/envs/env6/lib/python3.11/site-packages/xarray/core/variable.py:131, in as_variable(obj, name)
        129         obj = Variable(*obj)
        130     except (TypeError, ValueError) as error:
    --> 131         raise error.__class__(
        132             f"Variable {name!r}: Could not convert tuple of form "
        133             f"(dims, data[, attrs, encoding]): {obj} to Variable."
        134         )
        135 elif utils.is_scalar(obj):
        136     obj = Variable([], obj)


    ValueError: Variable 'intensity': Could not convert tuple of form (dims, data[, attrs, encoding]): (['time', 'latitude', 'longitude'], array([110., 110., 110., 110., 110., 100., 100., 100.,  95.,  85.,  85.,
            80.,  75.])) to Variable.


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


```python
tc_track_distance
```




    0      207.520151
    1      211.503372
    2      205.115188
    3      184.372377
    4      174.000972
              ...
    407    184.116723
    408    173.663997
    409    208.786829
    410    187.881376
    411    118.476946
    Length: 412, dtype: float64




```python
# Define your kriging model, e.g., Ordinary Kriging
ok = OrdinaryKriging(
    df_forecast.geometry.centroid.x,
    df_forecast.geometry.centroid.y,
    df_forecast['MeanWind'],  # Your wind speed data
    variogram_model='spherical',  # Choose an appropriate model
    verbose=False
)

# Define the points where you want to predict wind speed
prediction_points = grids.copy()  # Copy your centroids GeoDataFrame
prediction_points['predicted_wind_speed'] = 0  # Initialize the predicted wind speed

# Loop through the centroids and predict wind speed for each one
for idx, row in prediction_points.iterrows():
    x, y = row['geometry'].x, row['geometry'].y
    prediction_points.at[idx, 'predicted_wind_speed'] = ok.execute('grid', x, y)[0]

```

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_36495/1586986366.py:17: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas. Value '[88.01922945193941]' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
      prediction_points.at[idx, 'predicted_wind_speed'] = ok.execute('grid', x, y)[0]



```python
fig, ax = plt.subplots(1,1)
prediction_points.plot(column='predicted_wind_speed', cmap='coolwarm', markersize=20, legend=True, ax=ax)
df_forecast.plot(ax=ax, color='green')
plt.show()
```



![png](wind_to_grid_experiment_files/wind_to_grid_experiment_18_0.png)




```python
#gpd.sjoin(grids, df_forecast, how='left')

# Create PointData object
point_data = climada.entity.PointData()
point_data.set_lat_lon(wind_data["Latitude"], wind_data["Longitude"])
point_data.set_values(wind_data["MeanWind"])

# Load the grid data from the GeoPackage
grid_gdf = gpd.read_file("grid_file.gpkg")

# Create an ImpactModel
impact_model = climada.entity.ImpactModel()
impact_model.set_resolution(resol=0.01)  # Adjust the resolution as needed

# Interpolate wind speeds on the grid
impact_model.calc_intensity(point_data, resol=0.01)  # Use the same resolution as specified for the model

# Access the interpolated wind speeds on the grid
interpolated_wind_speeds = impact_model.intensity
```

### Create input dataframe


```python
# Get the windfield
windfield = df_forecast.MeanWind

#Track distance
points = gpd.points_from_xy(df_forecast.Longitude, df_forecast.Latitude)
tc_track_line = LineString(points)
# Get the track distance
DEG_TO_KM = 111.1
tc_track_distance = grids["geometry"].apply(
    lambda point: point.distance(tc_track_line) * DEG_TO_KM
)

# df_windfield = pd.DataFrame({
#     grid_point_id=grids["id"],
#     wind_speed=windfield,
#     track_distance=tc_track_distance,
#     geometry = grids.geometry
# })


```
