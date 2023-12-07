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
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/02_new_model_input"
```

## Get typhoon data

Typhoon IDs from IBTrACS are taken from
[here](https://ncics.org/ibtracs/index.php?name=browse-name)


```python
# Import list of typhoons to a dataframe
typhoons_df = pd.read_csv(input_dir / "01_windfield/typhoons.csv")
```


```python
housing_path_in = input_dir / '02_housing_damage/input'
df_housing = pd.read_csv(housing_path_in / 'fji_impact_data/processed_house_impact.csv')
```


```python
cyclones = df_housing['Cyclone Name'].unique()
intersection = typhoons_df[typhoons_df['typhoon_name'].isin(cyclones)].drop_duplicates(keep='last', subset = ['typhoon_name'])
intersection
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
      <th>Unnamed: 0</th>
      <th>typhoon_id</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>659</th>
      <td>659</td>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>678</th>
      <td>678</td>
      <td>2012346S14180</td>
      <td>Evan</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>715</th>
      <td>715</td>
      <td>2016041S14170</td>
      <td>Winston</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>732</th>
      <td>732</td>
      <td>2018038S15172</td>
      <td>Gita</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>749</th>
      <td>749</td>
      <td>2019359S08175</td>
      <td>Sarai</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>751</th>
      <td>751</td>
      <td>2020015S12170</td>
      <td>Tino</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>759</th>
      <td>759</td>
      <td>2020092S09155</td>
      <td>Harold</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>761</th>
      <td>761</td>
      <td>2020346S13168</td>
      <td>Yasa</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>769</th>
      <td>769</td>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Download NECESARY tracks
sel_ibtracs = [];i=0
for track in intersection.typhoon_id:
    sel_ibtracs.append(TCTracks.from_ibtracs_netcdf(storm_id=track))
    print('Track {} de {}'.format(i, len(intersection)-1))
    i+=1

```

    Track 0 de 8
    Track 1 de 8
    Track 2 de 8
    Track 3 de 8
    Track 4 de 8
    Track 5 de 8
    Track 6 de 8
    Track 7 de 8
    Track 8 de 8



```python
# Fix Tomas
#obs: .interp(x0,x,f(x)) gives the position of x0 in the fitting of (x,f(x))
#obs: daterange consider the track between certain intervals as discrete points instead of a continuous
tc_tracks = TCTracks()
for track in sel_ibtracs:
    tc_track = track.get_track()
    if tc_track.sid == '2010069S12188': #Tomas
        tc_track['lon'] *= -1
    tc_track.interp(
        time = pd.date_range(tc_track.time.values[0], tc_track.time.values[-1], freq="30T")
    )
    tc_tracks.append(tc_track)

```


```python
# Plot the tracks
# Takes a while, especially after the interpolation.
ax = tc_tracks.plot()
ax.set_title('Fiji and surroundings Typhoon Tracks')
ax.legend(loc='upper left')
plt.show()
```

    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.




![png](01_windfields_files/01_windfields_9_1.png)



## Construct the windfield

The typhoon tracks will be used to construct the wind field.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.


```python
filepath = (
    input_dir
    / "02_housing_damage/output/fji_0.1_degree_grid_centroids_land_overlap_new.gpkg"
)


gdf = gpd.read_file(filepath)
gdf["id"] = gdf["id"].astype(int)
```

Lets check the CRS of our GeoPandas DataFrame


```python
gdf.crs
```




    <Geographic 2D CRS: EPSG:4326>
    Name: WGS 84
    Axis Info [ellipsoidal]:
    - Lat[north]: Geodetic latitude (degree)
    - Lon[east]: Geodetic longitude (degree)
    Area of Use:
    - name: World.
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984 ensemble
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich




```python
# multipolygon data to centroids
# Read the GeoDataFrame from the file
gdf_aux = gpd.read_file(filepath)

# Set the CRS to EPSG 4326 (WGS 84)
gdf_aux = gdf.set_crs("EPSG:4326",allow_override=True)
cent = Centroids.from_geodataframe(gdf_aux)

cent.check()
cent.plot()
plt.show()
```

    2023-11-01 23:09:46,731 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.




![png](01_windfields_files/01_windfields_14_1.png)




```python
# Trop cyclone
tc = TropCyclone.from_tracks(
    tc_tracks, centroids=cent, store_windfields=True, intensity_thres=0
)
```


```python
# Let's look at a specific typhoon as an example.
example_typhoon_id = "2010069S12188"  # Tomas
ax = tc.plot_intensity(example_typhoon_id)
ax.set_title('Tomas')
plt.show()
```



![png](01_windfields_files/01_windfields_16_0.png)



## Save the windfields to df

Need to extract the windfield per typhoon, and
save it in a dataframe along with the grid points


```python
df_windfield = pd.DataFrame()
# cent = Centroids.from_geodataframe(gdf4)
# tc = TropCyclone.from_tracks(
#     tc_tracks, centroids=cent, store_windfields=True, intensity_thres=0
# )

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
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>355</td>
      <td>11.489542</td>
      <td>297.755905</td>
      <td>POINT (176.85000 -17.15000)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>409</td>
      <td>16.290230</td>
      <td>150.425391</td>
      <td>POINT (176.95000 -12.45000)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>456</td>
      <td>12.035214</td>
      <td>286.692573</td>
      <td>POINT (176.95000 -17.15000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>510</td>
      <td>16.976736</td>
      <td>148.319961</td>
      <td>POINT (177.05000 -12.45000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>511</td>
      <td>16.959047</td>
      <td>159.228640</td>
      <td>POINT (177.05000 -12.55000)</td>
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
      <th>3784</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5122</td>
      <td>0.000000</td>
      <td>340.122623</td>
      <td>POINT (181.55000 -19.15000)</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5123</td>
      <td>0.000000</td>
      <td>335.597712</td>
      <td>POINT (181.55000 -19.25000)</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5221</td>
      <td>0.000000</td>
      <td>350.361565</td>
      <td>POINT (181.65000 -18.95000)</td>
    </tr>
    <tr>
      <th>3787</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5223</td>
      <td>0.000000</td>
      <td>350.269410</td>
      <td>POINT (181.65000 -19.15000)</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5331</td>
      <td>0.000000</td>
      <td>335.268400</td>
      <td>POINT (181.75000 -19.85000)</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 7 columns</p>
</div>




```python
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



## Sanity checks


```python
# Check that that the grid points match for the example typhoon.
# Looks good to me!
df_example = df_windfield[df_windfield["typhoon_id"] == example_typhoon_id]
gdf_example = gpd.GeoDataFrame(gdf.merge(df_example, left_on="id", right_on="grid_point_id"))
gdf_example = gdf_example.set_geometry("geometry_x")
ax = gdf_example.plot(c=gdf_example["wind_speed"])
ax.set_title('Tomas Typhoon')
plt.show()
```



![png](01_windfields_files/01_windfields_21_0.png)




```python
# Plot wind speed against track distance
df_windfield.plot.scatter("track_distance", "wind_speed")
plt.xlabel('Track Distance [Km]')
plt.ylabel('Wind Speed [m/s]')

plt.show()
```



![png](01_windfields_files/01_windfields_22_0.png)



## Save everything


```python
df_windfield[df_windfield.typhoon_name=='Tomas']
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
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>355</td>
      <td>11.489542</td>
      <td>297.755905</td>
      <td>POINT (176.85000 -17.15000)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>409</td>
      <td>16.290230</td>
      <td>150.425391</td>
      <td>POINT (176.95000 -12.45000)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>456</td>
      <td>12.035214</td>
      <td>286.692573</td>
      <td>POINT (176.95000 -17.15000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>510</td>
      <td>16.976736</td>
      <td>148.319961</td>
      <td>POINT (177.05000 -12.45000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>511</td>
      <td>16.959047</td>
      <td>159.228640</td>
      <td>POINT (177.05000 -12.55000)</td>
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
      <th>416</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5122</td>
      <td>14.164562</td>
      <td>235.017644</td>
      <td>POINT (181.55000 -19.15000)</td>
    </tr>
    <tr>
      <th>417</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5123</td>
      <td>14.161645</td>
      <td>237.008207</td>
      <td>POINT (181.55000 -19.25000)</td>
    </tr>
    <tr>
      <th>418</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5221</td>
      <td>13.308208</td>
      <td>242.009356</td>
      <td>POINT (181.65000 -18.95000)</td>
    </tr>
    <tr>
      <th>419</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5223</td>
      <td>13.415320</td>
      <td>245.947867</td>
      <td>POINT (181.65000 -19.15000)</td>
    </tr>
    <tr>
      <th>420</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5331</td>
      <td>12.830786</td>
      <td>272.118149</td>
      <td>POINT (181.75000 -19.85000)</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 7 columns</p>
</div>




```python
# Save df as a csv file
df_windfield.to_csv(input_dir / "01_windfield/windfield_data_fji_new.csv")
```
