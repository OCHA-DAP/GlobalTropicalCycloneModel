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
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis_fji/02_model_features"
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

    2023-12-15 16:52:05,933 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 0 de 8
    2023-12-15 16:52:09,156 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 1 de 8
    2023-12-15 16:52:12,029 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 2 de 8
    2023-12-15 16:52:14,990 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 3 de 8
    2023-12-15 16:52:17,872 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 4 de 8
    2023-12-15 16:52:20,834 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 5 de 8
    2023-12-15 16:52:23,776 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 6 de 8
    2023-12-15 16:52:26,773 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
    Track 7 de 8
    2023-12-15 16:52:29,785 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.
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

The typhoon tracks will be used to construct the windfield.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.


```python
# Just grid-land overlap
filepath = (
    input_dir
    / "02_housing_damage/output/fji_0.1_degree_grid_centroids_land_overlap_new.gpkg"
)
gdf = gpd.read_file(filepath)
# Include oceans
filepath_complete = (
    input_dir
    / "02_housing_damage/output/fji_0.1_degree_grid_centroids_new.gpkg"
)
gdf_all = gpd.read_file(filepath_complete)

# Centroids
cent = Centroids.from_geodataframe(gdf) # grid-land overlap
cent_all = Centroids.from_geodataframe(gdf_all) # include oceans
```

    2023-12-15 16:56:31,138 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.
    2023-12-15 16:56:31,140 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



```python
# Change CRS
from shapely.geometry import Point
grid_transformed = gdf.copy()

# Define a function to adjust the longitude of a single polygon
def adjust_longitude(centroid):
    # Extract the coordinates of the polygon
    lon, lat = centroid.x, centroid.y

    # Adjust longitudes from [0, 360) to [-180, 180)
    if lon > 180:
        lon -= 360

    # Create a new Polygon with adjusted coordinates
    return Point(lon, lat)

# Apply the adjust_longitude function to each geometry in the DataFrame
grid_transformed["geometry"] = grid_transformed["geometry"].centroid.apply(adjust_longitude)
grid_transformed = grid_transformed[['id', 'Latitude', 'Longitude','geometry']]
cent_transformed = Centroids.from_geodataframe(grid_transformed)
```

    2023-12-15 17:16:09,423 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.


    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_46046/3669709669.py:18: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      grid_transformed["geometry"] = grid_transformed["geometry"].centroid.apply(adjust_longitude)



```python
cent.check()
cent.plot()
plt.show()
```



![png](01_windfields_files/01_windfields_13_0.png)




```python
# Trop cyclone
tc = TropCyclone.from_tracks(
    tc_tracks, centroids=cent, store_windfields=True, intensity_thres=0
)

tc_all = TropCyclone.from_tracks(
    tc_tracks, centroids=cent_all, store_windfields=True, intensity_thres=0
)

tc_transformed = TropCyclone.from_tracks(
    tc_tracks, centroids=cent_transformed, store_windfields=True, intensity_thres=0
)

```


```python
# Playing with different models
tc_aux = TropCyclone.from_tracks(
    tc_tracks, centroids=cent_all, store_windfields=True, intensity_thres=0, model='ER11'
)
```

## Examples


```python
# Let's look at a specific typhoon as an example.
name = 'Yasa'
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['typhoon_id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name)
plt.show()
```



![png](01_windfields_files/01_windfields_17_0.png)




```python
# Let's look at a specific typhoon as an example.
name = 'Tomas'
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['typhoon_id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name)
plt.show()
```



![png](01_windfields_files/01_windfields_18_0.png)




```python
# Let's look at a specific typhoon as an example.
name = 'Evan'
example_typhoon_id = intersection[intersection['typhoon_name'] == name]['typhoon_id'].iloc[0]
ax = tc_all.plot_intensity(example_typhoon_id)
ax.set_title(name)
plt.show()
```



![png](01_windfields_files/01_windfields_19_0.png)



## Save the windfields to df

Need to extract the windfield per typhoon, and
save it in a dataframe along with the grid points


```python
df_windfield = pd.DataFrame()

for intensity_sparse, event_id in zip(tc_all.intensity, tc_all.event_name):
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
    tc_track_distance = gdf_all["geometry"].apply(
        lambda point: point.distance(tc_track_line) * DEG_TO_KM
    )
    # Add to DF
    df_to_add = pd.DataFrame(
        dict(
            typhoon_id=[event_id] * npoints,
            typhoon_name=[typhoon_info["typhoon_name"].values[0]] * npoints,
            typhoon_year=[typhoon_info["typhoon_year"].values[0]] * npoints,
            grid_point_id=gdf_all["id"],
            wind_speed=windfield,
            track_distance=tc_track_distance,
            geometry = gdf_all.geometry
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
      <td>1</td>
      <td>16.490247</td>
      <td>120.487517</td>
      <td>POINT (176.55000 -12.05000)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>2</td>
      <td>15.763962</td>
      <td>130.848164</td>
      <td>POINT (176.55000 -12.15000)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>3</td>
      <td>15.067161</td>
      <td>141.208812</td>
      <td>POINT (176.55000 -12.25000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>4</td>
      <td>14.401776</td>
      <td>151.569459</td>
      <td>POINT (176.55000 -12.35000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5</td>
      <td>13.898706</td>
      <td>161.930106</td>
      <td>POINT (176.55000 -12.45000)</td>
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
      <th>50899</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5652</td>
      <td>21.675704</td>
      <td>137.978097</td>
      <td>POINT (182.05000 -21.65000)</td>
    </tr>
    <tr>
      <th>50900</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5653</td>
      <td>22.758225</td>
      <td>127.132655</td>
      <td>POINT (182.05000 -21.75000)</td>
    </tr>
    <tr>
      <th>50901</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5654</td>
      <td>23.902929</td>
      <td>116.287214</td>
      <td>POINT (182.05000 -21.85000)</td>
    </tr>
    <tr>
      <th>50902</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5655</td>
      <td>25.112219</td>
      <td>105.441772</td>
      <td>POINT (182.05000 -21.95000)</td>
    </tr>
    <tr>
      <th>50903</th>
      <td>2021029S16171</td>
      <td>Ana</td>
      <td>2021</td>
      <td>5656</td>
      <td>26.387468</td>
      <td>94.596331</td>
      <td>POINT (182.05000 -22.05000)</td>
    </tr>
  </tbody>
</table>
<p>50904 rows × 7 columns</p>
</div>



## Fix 0 windspeed values


```python
# Load grids
filepath_grids = (
    input_dir
    / "02_housing_damage/output/fji_0.1_degree_grid_new.gpkg"
)
grid_all = gpd.read_file(filepath_grids)
```


```python
# YASA example
yasa_gpd = gpd.GeoDataFrame(df_windfield[df_windfield.typhoon_name=='Yasa'].merge(
    grid_all,
    left_on='grid_point_id',
    right_on='id'), geometry='geometry_y')

# Load track for Yasa
id = intersection[intersection['typhoon_name'] == 'Yasa'].typhoon_id.to_list()
track = TCTracks.from_ibtracs_netcdf(storm_id=id)
tc_track = track.get_track()
points_ib = gpd.points_from_xy(tc_track.lon, tc_track.lat)
tc_track_line_ib = LineString(points_ib)
geometries_ib = gpd.GeoSeries([tc_track_line_ib])

line_gdf_ib_yasa = gpd.GeoDataFrame(geometry=geometries_ib)
```

    2023-12-11 18:53:37,799 - climada.hazard.tc_tracks - WARNING - The cached IBTrACS data set dates from 2023-06-07 23:07:38 (older than 180 days). Very likely, a more recent version is available. Consider manually removing the file /Users/federico/climada/data/IBTrACS.ALL.v04r00.nc and re-running this function, which will download the most recent version of the IBTrACS data set from the official URL.



```python
# Define a function to calculate mean values for neighboring cells
def calculate_mean_for_neighbors(idx, gdf, buffer_size):
    row = gdf.iloc[idx]
    if row['wind_speed'] == 0:  # Check if wind_speed is 0
        buffered = row['geometry'].buffer(buffer_size)  # Adjust buffer size as needed

        # Find neighboring geometries that intersect with the buffer, excluding the current geometry
        neighbors = gdf[~gdf.geometry.equals(row['geometry']) & gdf.geometry.intersects(buffered)]

        if not neighbors.empty:
            # drop rows with 0 windspeed vals (we dont want to compute the mean while considering these cells)
            neighbors = neighbors[neighbors['wind_speed'] !=0]
            if len(neighbors) !=0:
                mean_val = neighbors['wind_speed'].mean()
            else:
                mean_val = 0
            return mean_val
    return row['wind_speed']  # Return the original value if no neighbors or wind_speed != 0

# Apply the function row-wise to calculate mean values for wind_speed == 0
buffer = 0.1
yasa_mod_gpd = yasa_gpd.copy() # Create a copy
yasa_mod_gpd = yasa_mod_gpd.rename({
    'geometry_y':'geometry'
    }, axis=1)
yasa_mod_gpd = gpd.GeoDataFrame(yasa_mod_gpd, geometry='geometry')

yasa_mod_gpd['wind_speed'] = yasa_mod_gpd.apply(lambda row: calculate_mean_for_neighbors(row.name, yasa_mod_gpd, buffer_size=buffer), axis=1)
```


```python
fig, ax = plt.subplots(1,2, figsize=(10,10))
cmap = 'Reds'

yasa_gpd.plot(column='wind_speed', cmap=cmap, linewidth=0.2, ax=ax[0], edgecolor='0.3', legend=True)
line_gdf_ib_yasa.plot(ax=ax[0], color='k', linewidth=1, label='Typhoon track')

yasa_mod_gpd.plot(column='wind_speed', cmap=cmap, linewidth=0.2, ax=ax[1], edgecolor='0.3', legend=True)
line_gdf_ib_yasa.plot(ax=ax[1], color='k', linewidth=1, label='Typhoon track')

ax[0].axis('off')
ax[1].axis('off')

ax[0].set_xlim(176, 182)
ax[0].set_ylim(-20, -12)
ax[1].set_xlim(176, 182)
ax[1].set_ylim(-20, -12)

ax[0].set_title('Windspeed [m/s], YASA', size=10)
ax[1].set_title('Windspeed [m/s], YASA modified \nBuffer size={} degrees'.format(buffer), size=10)

plt.tight_layout()
plt.show()
```



![png](01_windfields_files/01_windfields_26_0.png)



Fix windfeld


```python
# For every typhoon
df_windfield_mod = df_windfield.copy()
df_windfield_mod = df_windfield_mod.drop('geometry', axis=1)
typhoons = df_windfield_mod.typhoon_name.unique()

buffer = 0.1
df_windfield_fix = pd.DataFrame()
for typhoon in typhoons:
    df_aux = gpd.GeoDataFrame(df_windfield_mod[df_windfield_mod.typhoon_name==typhoon].merge(
        grid_all,
        left_on='grid_point_id',
        right_on='id'), geometry='geometry')
    # compute mean
    df_aux['wind_speed'] = df_aux.apply(lambda row: calculate_mean_for_neighbors(row.name, df_aux, buffer_size=buffer), axis=1)
    # back to df
    df_aux = pd.DataFrame(df_aux)
    df_windfield_fix = pd.concat([df_windfield_fix, df_aux])

# Just keep grid-land overlap cells
df_windfield_fix_overlap = df_windfield_fix[df_windfield_fix.grid_point_id.isin(gdf.id)]
```

## Sanity checks


```python
# Check nans
df_windfield_fix_overlap.isna().sum()
```




    typhoon_id        0
    typhoon_name      0
    typhoon_year      0
    grid_point_id     0
    wind_speed        0
    track_distance    0
    id                0
    geometry          0
    dtype: int64




```python
# Check number of cells
print(len(df_windfield_fix_overlap[df_windfield_fix_overlap.typhoon_name=='Winston']))
```

    421



```python
# Check that that the grid points match for the example typhoon.
# Looks good to me!
df_example = df_windfield_fix_overlap[df_windfield_fix_overlap["typhoon_name"]== 'Yasa']
gdf_example = gpd.GeoDataFrame(df_example, geometry='geometry')

fig, ax = plt.subplots(1,1)
gdf_example.plot(column='wind_speed', cmap='Reds', linewidth=0.2, edgecolor='0.3', ax=ax, legend=True)
line_gdf_ib_yasa.plot(ax=ax, color='k', linewidth=1, label='Typhoon track')

ax.axis('off')
ax.set_xlim(176, 182)
ax.set_ylim(-20, -12)
ax.set_title('Yasa Typhoon')
plt.show()
```



![png](01_windfields_files/01_windfields_32_0.png)




```python
# Plot wind speed against track distance
df_windfield_fix_overlap.plot.scatter("track_distance", "wind_speed")
plt.xlabel('Track Distance [Km]')
plt.ylabel('Wind Speed [m/s]')

plt.show()
```



![png](01_windfields_files/01_windfields_33_0.png)



## Save everything


```python
df_windfield_fix_overlap[df_windfield_fix_overlap.typhoon_name=='Tomas']
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
      <th>id</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>354</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>355</td>
      <td>11.489542</td>
      <td>297.755905</td>
      <td>355</td>
      <td>POLYGON ((176.80000 -17.10000, 176.90000 -17.1...</td>
    </tr>
    <tr>
      <th>408</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>409</td>
      <td>16.290230</td>
      <td>150.425391</td>
      <td>409</td>
      <td>POLYGON ((176.90000 -12.40000, 177.00000 -12.4...</td>
    </tr>
    <tr>
      <th>455</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>456</td>
      <td>12.035214</td>
      <td>286.692573</td>
      <td>456</td>
      <td>POLYGON ((176.90000 -17.10000, 177.00000 -17.1...</td>
    </tr>
    <tr>
      <th>509</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>510</td>
      <td>16.976736</td>
      <td>148.319961</td>
      <td>510</td>
      <td>POLYGON ((177.00000 -12.40000, 177.10000 -12.4...</td>
    </tr>
    <tr>
      <th>510</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>511</td>
      <td>16.959047</td>
      <td>159.228640</td>
      <td>511</td>
      <td>POLYGON ((177.00000 -12.50000, 177.10000 -12.5...</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>5121</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5122</td>
      <td>14.164562</td>
      <td>235.017644</td>
      <td>5122</td>
      <td>POLYGON ((181.50000 -19.10000, 181.60000 -19.1...</td>
    </tr>
    <tr>
      <th>5122</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5123</td>
      <td>14.161645</td>
      <td>237.008207</td>
      <td>5123</td>
      <td>POLYGON ((181.50000 -19.20000, 181.60000 -19.2...</td>
    </tr>
    <tr>
      <th>5220</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5221</td>
      <td>13.308208</td>
      <td>242.009356</td>
      <td>5221</td>
      <td>POLYGON ((181.60000 -18.90000, 181.70000 -18.9...</td>
    </tr>
    <tr>
      <th>5222</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5223</td>
      <td>13.331042</td>
      <td>245.947867</td>
      <td>5223</td>
      <td>POLYGON ((181.60000 -19.10000, 181.70000 -19.1...</td>
    </tr>
    <tr>
      <th>5330</th>
      <td>2010069S12188</td>
      <td>Tomas</td>
      <td>2010</td>
      <td>5331</td>
      <td>12.687658</td>
      <td>272.118149</td>
      <td>5331</td>
      <td>POLYGON ((181.70000 -19.80000, 181.80000 -19.8...</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 8 columns</p>
</div>




```python
# Save df as a csv file
df_windfield_fix_overlap.to_csv(input_dir / "01_windfield/windfield_data_fji_new_fixed.csv")
```
