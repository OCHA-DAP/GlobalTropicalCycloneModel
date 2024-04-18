# Gathering features and creating the input dataset for the model training

## Codes

### Code 01: Windspeed

In 01.0 we download track information from IbTracks and we compute the wind_speed and track_distance features at grid level.
Some considerations:

-  We create a 'balanced' dataset: equal number of impactful events and non-impactful events. For this, we choose the closest events to the country (using the ```hti_distances.csv``` dataset) and downloaded the track information for each one of them.
-  We perform interpolation between each coordinate (lat, lom, time) and weather related variable (central pressure, atmospheric pressure, wind_speed) in order to 'smooth' the tracks.
-  The output is at grid level (grids defined in code 02.0).
-  We also gathered the landfall information (if available) and created the ```typhoons.csv``` table with information about start_time, end_time and landfall_time of each event. If the event didn't make landfall, we define the landfall as the closest point (lan, lat, time) to the land.

Output: windspeed and distance to track at grid level.


### Code 02: Grid definition and building damage

In 02.0 we define the grid. As usual, we created 0.1 degree grid cells.

In 02.1.0 we download the distribution of buildings using Google Open Building dataset. We end up with a csv with information on number of buildings per grid cell.

In 02.1.1, having defined the building damage in 06.0, we disaggregated to grid level in the usual way (the weight for the disaggregation is the number of buildings per grid).

Output: datasets of damage at grid level and number of buildings by grid.

### Code 03: Rainfall information

In 03.0_download_rainfall_dataset.ipynb we basically download the rainfall data from [GPM](https://arthurhouhttps.pps.eosdis.nasa.gov/pub/gpmdata). We need to register for this. It's everything explained at the code.

Output: GPM files for each typhoon.

In 03.1_create_rainfall_dataset.ipynb we compute the mean and max rainfall measured, at grid level.

Output: .csv files for each typhoon containing stats information at grid level.

In 03.2_computing_stats.ipynb we compute for every typhoon the max rainfall (mm/hr) at grid level using 2 time intervals: 6h and 24h.

Output: a dataset containing the grid_id, max_6h and max_24h measures for every grid cell and for every typhoon.


### Code 04: Topography variables

We get terrain altitude data from [SRTM](https://dwtkns.com/srtm30m/). Here we need to register, select the desired location and download data. The data consists of a group of .hdg files.

In 04.0.1_topography_variables.iynb we import each .hdg file, merge it into a single .tif file and compute mean altitude and mean slope measures at grid level.  Additionally, using the Fiji shapefile, we compute the length of the coastline for each grid cell and we created a binary variable *with_grid* that is 1 if the grid has a coastline or 0 if not.

Output: dataset of mean altitude, mean slope, coast length and with_coast at grid level.


### Code 05: IWI

Here we get the Mean International wealth index (IWI) from [globaldatalab](https://globaldatalab.org/areadata/table/iwi/FJI/). We use this data because Fiji is not in the list of countries with RWI information.

In 05.0_relative_wealth_index.ipynb we calculate the IWI at grid level.

Output: dataset of IWI at grid level.


### Code 06: Population data

In 06.0 we:

-  Create the impact data for buildings (based on people affected) that we later use in 02.1.1
-  Disaggregate the affected population to grid level using the number of people by grid.

To infer the building damage from the affected population dataset, we use 2 approaches:

-  A dummy approach consisting on transforming people affected --> bld damaged based on population/buildings density
-  An approach consisting on using the same ralation of bld damaged / people affected found for the Philippines.

Output: dataset of affected population at grid level and another dataset of impact data for buildings.

### Code 08: Merging the dataset

We basically merge every feature that we gathered to create the input dataset for training our model. For that, we use the grid_id number.

Input: damage data, population_data, wind data, rainfall data, IWI data and topographical data.

Output: dataset of all the data mentioned at grid level.
