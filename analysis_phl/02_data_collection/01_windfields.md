# Windfields

This notebook is for downloading typhoon tracks from
IBTrACS and generating the windfields.


```python
from pathlib import Path
import os

from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
```


```python
DEG_TO_KM = 111.1  # Convert 1 degree to km
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_phl/02_model_features"
```

## Get typhoon data

Typhoon IDs from IBTrACS are taken from
[here](https://ncics.org/ibtracs/index.php?name=browse-name)


```python
# Import list of typhoons to a dataframe
typhoons_df = pd.read_csv(input_dir / "01_windfield/typhoons.csv")
intersection = typhoons_df.copy()
```


```python
# Download NECESARY tracks
sel_ibtracs = [];i=0
for track in intersection.typhoon_id:
    sel_ibtracs.append(TCTracks.from_ibtracs_netcdf(storm_id=track))
    print('Track {} de {}'.format(i, len(intersection)-1))
    i+=1
```

    2023-12-13 16:43:36,847 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 0 de 38
    2023-12-13 16:43:39,938 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 1 de 38
    2023-12-13 16:43:42,845 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 2 de 38
    2023-12-13 16:43:45,982 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 3 de 38
    2023-12-13 16:43:49,047 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 4 de 38
    2023-12-13 16:43:52,031 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 5 de 38
    2023-12-13 16:43:55,003 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 6 de 38
    2023-12-13 16:43:57,989 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 7 de 38
    2023-12-13 16:44:00,924 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 8 de 38
    2023-12-13 16:44:03,915 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 9 de 38
    2023-12-13 16:44:06,882 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 10 de 38
    2023-12-13 16:44:09,899 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 11 de 38
    2023-12-13 16:44:12,823 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 12 de 38
    2023-12-13 16:44:15,818 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 13 de 38
    2023-12-13 16:44:18,907 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 14 de 38
    2023-12-13 16:44:21,897 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 15 de 38
    2023-12-13 16:44:24,822 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 16 de 38
    2023-12-13 16:44:27,808 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 17 de 38
    2023-12-13 16:44:30,739 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 18 de 38
    2023-12-13 16:44:33,733 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 19 de 38
    2023-12-13 16:44:36,660 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 20 de 38
    2023-12-13 16:44:39,662 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 21 de 38
    2023-12-13 16:44:42,622 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 22 de 38
    2023-12-13 16:44:45,636 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 23 de 38
    2023-12-13 16:44:48,826 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 24 de 38
    2023-12-13 16:44:51,939 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 25 de 38
    2023-12-13 16:44:54,942 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 26 de 38
    2023-12-13 16:44:57,926 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 27 de 38
    2023-12-13 16:45:00,923 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 28 de 38
    2023-12-13 16:45:03,907 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 29 de 38
    2023-12-13 16:45:06,907 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 30 de 38
    2023-12-13 16:45:09,884 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 31 de 38
    2023-12-13 16:45:12,872 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 32 de 38
    2023-12-13 16:45:15,849 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 33 de 38
    2023-12-13 16:45:18,875 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 34 de 38
    2023-12-13 16:45:21,851 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 35 de 38
    2023-12-13 16:45:24,871 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 36 de 38
    2023-12-13 16:45:27,979 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 37 de 38
    2023-12-13 16:45:30,958 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 38 de 38



```python
tc_tracks = TCTracks()
for track in sel_ibtracs:
    tc_track = track.get_track()
    # Interpolation
    tc_track.interp(
        time = pd.date_range(tc_track.time.values[0], tc_track.time.values[-1], freq="30T")
    )
    tc_tracks.append(tc_track)
```


```python
# Plot the tracks
# Takes a while, especially after the interpolation.
ax = tc_tracks.plot()
ax.set_title('PHL and surroundings Typhoon Tracks')
plt.show()
```



![png](01_windfields_files/01_windfields_7_0.png)



## Construct the windfield

The typhoon tracks will be used to construct the windfield.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.


```python
# Just grid-land overlap
filepath = (
    input_dir
    / "02_housing_damage/output/phl_0.1_degree_grid_centroids_land_overlap.gpkg"
)
gdf = gpd.read_file(filepath)

# Include oceans
filepath_complete = (
    input_dir
    / "02_housing_damage/output/phl_0.1_degree_grid_centroids.gpkg"
)
gdf_all = gpd.read_file(filepath_complete)

# Centroids
cent = Centroids.from_geodataframe(gdf) # grid-land overlap
cent_all = Centroids.from_geodataframe(gdf_all) # include oceans
```

    2023-12-13 16:51:09,458 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.
    2023-12-13 16:51:09,461 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



```python
cent.check()
cent.plot()
plt.show()
```



![png](01_windfields_files/01_windfields_10_0.png)




```python
# Trop cyclone
tc = TropCyclone.from_tracks(
    tc_tracks, centroids=cent, store_windfields=True, intensity_thres=0
)

tc_all = TropCyclone.from_tracks(
    tc_tracks, centroids=cent_all, store_windfields=True, intensity_thres=0
)
```

## Examples


```python
# Let's look at a specific typhoon as an example.
name = 'DURIAN'
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['typhoon_id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name)
plt.show()
```



![png](01_windfields_files/01_windfields_13_0.png)




```python
# Let's look at a specific typhoon as an example.
name = 'MELOR'
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['typhoon_id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name)
plt.show()
```



![png](01_windfields_files/01_windfields_14_0.png)




```python
# Let's look at a specific typhoon as an example.
name = 'MOLAVE'
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['typhoon_id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name)
plt.show()
```



![png](01_windfields_files/01_windfields_15_0.png)



## Save the windfields to df

Need to extract the windfield per typhoon, and
save it in a dataframe along with the grid points


```python
df_windfield = pd.DataFrame()

for intensity_sparse, event_id in zip(tc.intensity, tc.event_name):
    # Get the windfield
    windfield = intensity_sparse.toarray().flatten()
    npoints = len(windfield)
    typhoon_info = typhoons_df[typhoons_df["typhoon_id"] == event_id]
    # Get the track distance
    tc_track = tc_tracks.get_track(track_name=event_id)
    points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
    tc_track_line = LineString(points)
    # TODO: Not sure that curvature is taken into account in this
    # calculation, i.e. might be missing cos(lat) term. Since we're
    # close to the equator in this case it doesn't matter.
    tc_track_distance = gdf["geometry"].apply(
        lambda point: point.distance(tc_track_line) * DEG_TO_KM
    )
    # Add to DF
    df_to_add = pd.DataFrame(
        dict(
            typhoon_id=[event_id] * npoints,
            typhoon_name=[typhoon_info["typhoon_name"].values[0]] * npoints,
            typhoon_year=[typhoon_info["typhoon_year"].values[0]] * npoints,
            grid_point_id=gdf["id"],
            wind_speed=windfield,
            track_distance=tc_track_distance,
            geometry = gdf.geometry
        )
    )
    df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)
df_windfield
```

    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/shapely/measurement.py:72: RuntimeWarning: invalid value encountered in distance
      return lib.distance(a, b, **kwargs)





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
      <th>typhoon_id</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>101</td>
      <td>0.0</td>
      <td>308.690020</td>
      <td>POINT (114.30000 11.10000)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4475</td>
      <td>0.0</td>
      <td>623.151133</td>
      <td>POINT (116.90000 7.90000)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4639</td>
      <td>0.0</td>
      <td>588.305668</td>
      <td>POINT (117.00000 8.20000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4640</td>
      <td>0.0</td>
      <td>599.219433</td>
      <td>POINT (117.00000 8.10000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4641</td>
      <td>0.0</td>
      <td>610.140281</td>
      <td>POINT (117.00000 8.00000)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145309</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20677</td>
      <td>0.0</td>
      <td>644.615067</td>
      <td>POINT (126.60000 7.60000)</td>
    </tr>
    <tr>
      <th>145310</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20678</td>
      <td>0.0</td>
      <td>655.724121</td>
      <td>POINT (126.60000 7.50000)</td>
    </tr>
    <tr>
      <th>145311</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20679</td>
      <td>0.0</td>
      <td>666.833174</td>
      <td>POINT (126.60000 7.40000)</td>
    </tr>
    <tr>
      <th>145312</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20680</td>
      <td>0.0</td>
      <td>677.942228</td>
      <td>POINT (126.60000 7.30000)</td>
    </tr>
    <tr>
      <th>145313</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>689.051282</td>
      <td>POINT (126.60000 7.20000)</td>
    </tr>
  </tbody>
</table>
<p>145314 rows × 7 columns</p>
</div>



## Sanity checks


```python
# Check nans
df_windfield.isna().sum()
```




    typhoon_id        0
    typhoon_name      0
    typhoon_year      0
    grid_point_id     0
    wind_speed        0
    track_distance    0
    geometry          0
    dtype: int64




```python
# Check number of cells
print(len(df_windfield[df_windfield.typhoon_name=='MELOR']))
```

    3726



```python
# Plot wind speed against track distance
df_windfield.plot.scatter("track_distance", "wind_speed")
plt.xlabel('Track Distance [Km]')
plt.ylabel('Wind Speed [m/s]')

plt.show()
```



![png](01_windfields_files/01_windfields_21_0.png)



## Save everything


```python
# Save df as a csv file
df_windfield.to_csv(input_dir / "01_windfield/windfield_data_phl.csv")
```
