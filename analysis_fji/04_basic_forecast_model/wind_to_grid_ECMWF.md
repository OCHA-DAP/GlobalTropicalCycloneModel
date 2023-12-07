---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: env6
    language: python
    name: python3
---

```python
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon
from climada.hazard import Centroids, TCTracks, TropCyclone, Hazard
from climada_petals.hazard import TCForecast
import warnings
import xarray as xr
from datetime import datetime, timedelta

# Filter out specific UserWarning by message
warnings.filterwarnings("ignore", message="Converting non-nanosecond precision datetime values to nanosecond precision")

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

```python
tc_fcast = TCForecast()
tc_fcast.fetch_ecmwf()
```

```python
tc_fcast.data[0]
```

```python
tc_fcast.plot()
plt.show()
```

```python
cent = Centroids.from_geodataframe(grids)
tc = TropCyclone.from_tracks(
    tc_fcast, centroids=cent, store_windfields=True, intensity_thres=0
)
```

Put a threshold in the forcast time. This make paths shorter in some cases but this is just because the more we move in time, the less precisse the wind forecast is.

```python
# Modify each of the event
n_events = len(tc_fcast.data)

# Threshold
thres = 72 #h
today = datetime.now()
# Calculate the threshold datetime from the current date and time
threshold_datetime = np.datetime64(today + timedelta(hours=thres))

xarray_data_list = []
for i in range(n_events):
    data_event = tc_fcast.data[i]
    # Elements to consider
    index_thres = len(np.where(np.array(data_event.time) < threshold_datetime)[0])
    if index_thres > 4: # Events with at least 4 datapoints
        data_event_thres = data_event.isel(time=slice(0, index_thres))
        xarray_data_list.append(data_event_thres)
    else:
        continue

# Create TropCyclone class with modified data
tc_fcast_mod = TCForecast(xarray_data_list)
tc = TropCyclone.from_tracks(tc_fcast_mod, centroids=cent, store_windfields=True, intensity_thres=0)
```

```python
# Create windfield dataset
event_names = list(tc.event_name)

# Define the boundaries for Fiji region + 3 degrees in each direction
xmin, xmax, ymin, ymax = 173, 185, -24, -9
fiji_polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
```

```python
df_windfield = pd.DataFrame()
for i, intensity_sparse in enumerate(tc.intensity):
    # Get the windfield
    windfield = intensity_sparse.toarray().flatten()
    npoints = len(windfield)
    event_id = event_names[i]

    # Track distance
    DEG_TO_KM = 111.1
    tc_track = tc_fcast_mod.get_track()[i]
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
```

```python
df_windfield.sort_values('time_init')
```

```python
df_windfield[df_windfield.in_fiji == True]
```

```python
fiji_forecast = df_windfield[df_windfield.in_fiji == True]
events_fiji = fiji_forecast.unique_id.unique()

fiji_forecast[fiji_forecast.unique_id == events_fiji[0]]
```

```python
event1 = fiji_forecast[fiji_forecast.unique_id == events_fiji[0]]
gdf_aux = gpd.GeoDataFrame(event1)

# Plot
fig, ax = plt.subplots(1,1)
gdf_aux.plot(ax=ax, column='wind_speed', cmap='coolwarm', markersize=20, legend=True, label= 'Wind Speed [m/s]')
```

```python
# Input dataset
input_df = df.merge(fiji_forecast, left_on='grid_point_id', right_on='grid_point_id')[
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
    ]].reset_index(drop=True)

input_df
```
