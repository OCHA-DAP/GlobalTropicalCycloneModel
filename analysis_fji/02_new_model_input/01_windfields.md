# Windfields

This notebook is for downloading typhoon tracks from
IBTrACS and generating the windfields.

```python
%load_ext jupyter_black
```

```python
from pathlib import Path
import os
from dotenv import load_dotenv

from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
```

```python
load_dotenv()
```

```python
DEG_TO_KM = 111.1  # Convert 1 degree to km
FJI_CRS = "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
input_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis/02_new_model_input"
)
```

## Get typhoon data

Typhoon IDs from IBTrACS are taken from
[here](https://ncics.org/ibtracs/index.php?name=browse-name)

```python
# Import list of typhoons to a dataframe
WINSTON_ID = "2016041S14170"
YASA_ID = "2020346S13168"
typhoons_df = pd.read_csv(input_dir / "01_windfield/typhoons.csv")
typhoons_df
```

```python
# Download all tracks from the south pacific basin
sel_ibtracs = TCTracks.from_ibtracs_netcdf(
    year_range=(1970, 2022), basin="SP", provider="usa"
)
sel_ibtracs.size
```

```python
# Load impact data
df_impact = pd.read_csv(
    input_dir / "02_housing_damage/input/fji_impact_data.csv"
)
print(df_impact)
```

```python
df_tracks = pd.DataFrame()
typhoon_ids = []

for track in sel_ibtracs.data:
    start_date = track.time.min()
    eff_date = start_date - pd.Timedelta(days=200)
    season = f"{eff_date.dt.year.values}/{eff_date.dt.year.values + 1}"
    name_season = track.name.capitalize() + " " + season
    if name_season in df_impact["Name Season"]:
        typhoon_ids.append(track.sid)
    track.attrs[
        "season"
    ] = f"{eff_date.dt.year.values}/{eff_date.dt.year.values + 1}"
```

```python
# Get_track returns the first matching track based
# on the track ID. Interpolate from 3 hours to
# 30 minute intervals to create a smooth intensity field.
tc_tracks = TCTracks()
cyclone_ids = [WINSTON_ID, YASA_ID]
# typhoon_ids = typhoons_df["typhoon_id"]
for typhoon_id in typhoon_ids:
    tc_track = sel_ibtracs.get_track(typhoon_id)
    tc_track = tc_track.interp(
        time=pd.date_range(
            tc_track.time.values[0], tc_track.time.values[-1], freq="30T"
        )
    )
    tc_tracks.append(tc_track)

display(tc_tracks.data[:1])
```

```python
# Plot the tracks
# Takes a while, especially after the interpolation.
tc_tracks.plot()
```

## Construct the windfield

The typhoon tracks will be used to construct the wind field.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.

```python
# input_dir = (Path(os.getenv("STORM_DATA_DIR"))/ "analysis/02_new_model_input/input")
# filepath = input_dir / "phl_0.1_degree_grid_centroids.gpkg"
# gdf = gpd.read_file(filepath)

filepath = (
    input_dir
    / "02_housing_damage/output/fji_0.1_degree_grid_centroids_land_overlap.gpkg"
)
gdf = gpd.read_file(filepath)
gdf["id"] = gdf["id"].astype(int)

gdf.to_crs(epsg=4326)
```

```python
# multipolygon data to centroids

cent = Centroids.from_geodataframe(gdf.to_crs(epsg=4326))

cent.check()
cent.plot();
```

```python
cent.crs
```

```python
# Consider wind speeds lower than 17.5, which is the
# division between a TD and a TS.
tc = TropCyclone.from_tracks(
    tc_tracks, centroids=cent, store_windfields=True, intensity_thres=0
)
```

```python
tc.plot_fraction(YASA_ID)
```

```python
# Let's look at a specific typhoon as an example.
# It looks weird but I think it's just a
# projection issue, as the parts on land look fine
example_typhoon_id = "2020299N11144"  # Goni 2020
# example_typhoon_id = "2019354N05151"  # Phanfone
tc.plot_intensity(YASA_ID)
```

## Save the windfields

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
    tc_track_line = LineString(gpd.points_from_xy(tc_track.lon, tc_track.lat))
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
        )
    )
    df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)
df_windfield
```

## Sanity checks

```python
# Check that that the grid points match for the example typhoon.
# Looks good to me!
df_example = df_windfield[df_windfield["typhoon_id"] == WINSTON_ID]
gdf_example = gdf.merge(df_example, left_on="id", right_on="grid_point_id")
gdf_example.plot(c=gdf_example["wind_speed"])
```

```python
# Plot wind speed against track distance
df_windfield.plot.scatter("track_distance", "wind_speed")
```

## Save everything

```python
# Save df as a csv file
df_windfield.to_csv(input_dir / "01_windfield/windfield_data.csv")
```

```python

```
