# Data acquisition and handling

Here we acquire the wind, rainfall, topology and wealth distribution data and convert the information obtained to grid level.

## Report: codes


### Code 01: Windfields

In 01_windfield.ipynb we download the typhoon tracks from IbTracs, fix the coordinates from a specific typhoon (Tomas) from their repository and compute the 1-min max sustained windspeed and the distance to the track for each grid defined 02.0_grid_definition.ipynb.

- Output: windspeed and distance to track at grid level.


### Code 02: Grids and damage

In 02.0_grid_definition.ipynb we create the grid cells as squares with 0.1 degree width sides (polygons). Each grid has a unique grid_id.  We also compute the centroids of each each grid and study the distribution of grid cells by municipality. Additionally, we compute the grid-land intersection.

- Output: datasets of grids_id by municipality and grid_id, polygons and centroids geometries for
	- all the grids.
	- grids that intercept with land.

In 02.1_damage_agg_by_building.ipynb we use the east and west buildings dataset (see [data_fji](https://drive.google.com/drive/folders/15e5BPkhECGeKTObdJIuixICMqhPhVyPK)). We locate every building in the map and compute the quantity of buildings per grid cell. Next, since we have the number of buildings per grid and we know the number of houses destroyed at municipality level, we compute the percentage of houses destroyed in each grid. For that:

- We compute the density of buildings per grid.
- We compute the percentage of buildings destroyed per municipality.
- We multiply these 2 values to get the the percentage of buildings destroyed at grid level.

Output: datasets of damage at grid level and number of buildings by grid.


### Code 03: Rainfall

In 03.0_download_rainfall_dataset.ipynb we basically download the rainfall data from [GPM](https://arthurhouhttps.pps.eosdis.nasa.gov/pub/gpmdata). We need to register for this. It's everything explained at the code.

Output: GPM files for each typhoon.

In 03.1_create_rainfall_dataset.ipynb we compute the mean and max rainfall measured, at grid level.

Output: .csv files for each typhoon containing stats information at grid level.

In 03.2_computing_stats.ipynb we compute for every typhoon the max rainfall (mm/hr) at grid level using 2 time intervals: 6h and 24h.

Output: a dataset containing the grid_id, max_6h and max_24h measures for every grid cell and for every typhoon.


### Code 04: Topography variables

We get terrain altitude data from [SRTM](https://dwtkns.com/srtm30m/). Here we need to register, select the desired location and download data. The data consists of a group of .hdg files.

In 04.0_topography_variables.iynb we import each .hdg file, merge it into a single .tif file and compute mean altitude and mean slope measures at grid level.  Additionally, using the Fiji shapefile, we compute the length of the coastline for each grid cell and we created a binary variable *with_grid* that is 1 if the grid has a coastline or 0 if not.

Output: dataset of mean altitude, mean slope, coast length and with_coast at grid level.


### Code 05: IWI

Here we get the Mean International wealth index (IWI) from [globaldatalab](https://globaldatalab.org/areadata/table/iwi/FJI/). We use this data because Fiji is not in the list of countries with RWI information.

In 05.0_relative_wealth_index.ipynb we calculate the IWI at grid level.

Output: dataset of IWI at grid level.

### Code 06: Population data

We explore population dataset and urban-rural dataset. The idea is to get an accurate estimation of people per grid. Finally, we managed to get our hands on some Census data. Here, we not only got information about population but also about households by grid.

### Code 07: Light-index indicator

We tried to create a vulnerability index based on brightness levels of night satellite images.
