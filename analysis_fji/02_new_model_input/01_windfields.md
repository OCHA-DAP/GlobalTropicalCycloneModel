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
```python
# Download NECESARY tracks
sel_ibtracs = [];i=0
for track in intersection.typhoon_id:
    sel_ibtracs.append(TCTracks.from_ibtracs_netcdf(storm_id=track))
    print('Track {} de {}'.format(i, len(intersection)-1))
    i+=1

```

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

## Construct the windfield

The typhoon tracks will be used to construct the wind field.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.


```python
# note that the below input file is only generated in
# 02.0_grid_definition.ipynb

filepath = (
    input_dir
    / "02_housing_damage/output/fji_0.1_degree_grid_centroids_land_overlap_new.gpkg"
)

# filepath = (
#     input_dir
#     / "02_housing_damage/output/fji_0.1_degree_grid_centroids_land_overlap.gpkg"
# )


gdf = gpd.read_file(filepath)
gdf["id"] = gdf["id"].astype(int)
```

Lets check the CRS of our GeoPandas DataFrame


```python
gdf.crs
```
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
```python
df_windfield.isna().sum()
```
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
```python
# Plot wind speed against track distance
df_windfield.plot.scatter("track_distance", "wind_speed")
plt.xlabel('Track Distance [Km]')
plt.ylabel('Wind Speed [m/s]')

plt.show()
```
## Save everything


```python
df_windfield[df_windfield.typhoon_name=='Tomas']
```
```python
# Save df as a csv file
df_windfield.to_csv(input_dir / "01_windfield/windfield_data_fji_new.csv")
```

```python

```
