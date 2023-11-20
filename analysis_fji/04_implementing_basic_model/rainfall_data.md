```python
import getpass
import os
from pathlib import Path

import pandas as pd
import datetime as dt
from datetime import timedelta
from bs4 import BeautifulSoup
import requests
import time
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterstats import zonal_stats

from utils import get_combined_dataset, get_municipality_grids
from input_dataset import create_input_dataset, create_windfield_dataset
```


```python
# Setting directories
input_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/03_rainfall/input"
)

output_dir = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/03_rainfall/output"
)

grid_input = (
    Path(os.getenv("STORM_DATA_DIR"))
    / "analysis_fji/02_new_model_input/02_housing_damage/output/"
)
```

## Events of interest: windfield data


```python
# Download real-time data
df_windfield = create_windfield_dataset()
```

    /Users/federico/Library/CloudStorage/OneDrive-Personal/Documentos/ISI_Project/FIJI clean/analysis_fji/04_implementing_basic_model/input_dataset.py:24: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      grids.geometry = grids.geometry.to_crs(grids.crs).centroid
    Download: 100%|██████████| 38/38 [00:03<00:00, 12.18 files/s]
    Processing:  66%|██████▌   | 25/38 [00:02<00:01, 11.22 files/s]

    2023-11-10 11:59:11,218 - climada_petals.hazard.tc_tracks_forecast - WARNING - Pressure at time 17: only 1 variable value for 2 ensemble members, duplicating value to all members. This is only acceptable for lat and lon data at time 0.
    2023-11-10 11:59:11,218 - climada_petals.hazard.tc_tracks_forecast - WARNING - Maximum 10m wind at time 17: only 1 variable value for 2 ensemble members, duplicating value to all members. This is only acceptable for lat and lon data at time 0.
    2023-11-10 11:59:11,221 - climada_petals.hazard.tc_tracks_forecast - WARNING - Pressure at time 20: only 1 variable value for 2 ensemble members, duplicating value to all members. This is only acceptable for lat and lon data at time 0.
    2023-11-10 11:59:11,224 - climada_petals.hazard.tc_tracks_forecast - WARNING - Maximum 10m wind at time 27: only 1 variable value for 2 ensemble members, duplicating value to all members. This is only acceptable for lat and lon data at time 0.


    Processing: 100%|██████████| 38/38 [00:04<00:00,  8.91 files/s]

    2023-11-10 11:59:12,544 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/shapely/measurement.py:72: RuntimeWarning: invalid value encountered in distance
      return lib.distance(a, b, **kwargs)
    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/shapely/measurement.py:72: RuntimeWarning: invalid value encountered in distance
      return lib.distance(a, b, **kwargs)
    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/shapely/measurement.py:72: RuntimeWarning: invalid value encountered in distance
      return lib.distance(a, b, **kwargs)



```python
# Focus on Fiji
fiji_forecast = df_windfield[df_windfield.in_fiji == True]

# Group the DataFrame by the 'unique_id' column
grouped = fiji_forecast.groupby('unique_id')

# Create a list of Forecasts DataFrames, one for each unique_id
list_forecast = [group for name, group in grouped]
```

I don't know the landfall. All I have are some windforecasts. So the time span for the adquisition of the rainfall data is just the time span of the forecast.


```python
event_metadata = pd.DataFrame({
    'event':fiji_forecast.unique_id.unique(),
    'start_date':[df.time_init.iloc[0] for df in list_forecast],
    'end_date': [df.time_end.iloc[0] for df in list_forecast]
})
```

## Forecast data dowloading

We want to access every precipitation (prcp) file inside GEFS Bias-Corrected ensemble

Check https://www.nco.ncep.noaa.gov/pmb/products/gens/ for more information of what the name of every file means.

Also, there are 2 type of files:

- 06h ...
- 24h ...

that corresponds to xxh (06h,24h) rainfall accumulation.

Some info:

- Resolution: 0.5 degrees (v!=12.0) https://www.nco.ncep.noaa.gov/pmb/products/gens/
- Units:
- Forecast reliability: avg 40% in 72h https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs/faq.php


```python
import requests
from datetime import datetime
import os
from bs4 import BeautifulSoup

def download_gefs_data(download_folder):
    # Calculate the current date
    today = datetime.now().strftime("%Y%m%d")

    # Generate folder names for 18, 12, 06, and 00 h
    folder_names = [f"{today}/18",f"{today}/12", f"{today}/06", f"{today}/00"]

    for folder in folder_names:
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.{folder}/prcp_bc_gb2/"
        response = requests.get(url)

        if response.status_code == 200:
            print(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links  = [a['href'] for a in soup.find_all('a') if a['href'].startswith("geprcp.t") and a['href'][:-7].endswith("bc_")]

            if links:
                for link in links:
                    link_url = url + link
                    file_name = link.split("/")[-1]
                    file_path = os.path.join(download_folder, file_name)

                    print(f"Downloading {link_url}")
                    download_response = requests.get(link_url)

                    with open(file_path, 'wb') as f:
                        f.write(download_response.content)
                return
            break
    print("prcp_gb2 not found for today.")


# Example usage:
today = datetime.now().strftime("%Y%m%d")
download_folder = Path(input_dir) / "NOMADS" / today / "input_files"
os.makedirs(download_folder, exist_ok=True)
download_gefs_data(download_folder)
```

    https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf006
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf012
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf018
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf024
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf030
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf036
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf042
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf048
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf054
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf060
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf066
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf072
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf078
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf084
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf090
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf096
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf102
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf108
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf114
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf120
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf126
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf132
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf138
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf144
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf150
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf156
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf162
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf168
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf174
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf180
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf186
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf192
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf198
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf204
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf210
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf216
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf222
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf228
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf234
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf240
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf246
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf252
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf258
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf264
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf270
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf276
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf282
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf288
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf294
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf300
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf306
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf312
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf318
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf324
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf330
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf336
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf342
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf348
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf354
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf360
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf366
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf372
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf378
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_06hf384
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf024
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf030
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf036
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf042
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf048
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf054
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf060
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf066
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf072
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf078
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf084
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf090
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf096
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf102
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf108
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf114
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf120
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf126
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf132
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf138
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf144
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf150
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf156
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf162
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf168
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf174
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf180
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf186
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf192
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf198
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf204
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf210
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf216
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf222
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf228
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf234
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf240
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf246
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf252
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf258
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf264
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf270
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf276
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf282
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf288
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf294
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf300
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf306
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf312
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf318
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf324
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf330
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf336
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf342
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf348
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf354
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf360
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf366
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf372
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf378
    Downloading https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.20231110/06/prcp_bc_gb2/geprcp.t06z.pgrb2a.0p50.bc_24hf384


## Forecast data analysis


```python
# Load grid
grid_land_overlap = gpd.read_file(grid_input / "fji_0.1_degree_grid_land_overlap_new.gpkg")
grid_land_overlap["id"] = grid_land_overlap["id"].astype(int)
grid = grid_land_overlap.copy()

# Load files
today = datetime.now().strftime("%Y%m%d")
gefs_folder_path = Path(input_dir) / "NOMADS" / today / "input_files"

# Output folder
processed_output_dir_grid = Path(input_dir) / "NOMADS" / today / "output_processed_bygrid"
os.makedirs(processed_output_dir_grid, exist_ok=True)

processed_output_dir = Path(input_dir) / "NOMADS" / today / "output_processed"
os.makedirs(processed_output_dir, exist_ok=True)
```

Some checks


```python
# Let's see what are the values for nodata
gefs_file = os.listdir(gefs_folder_path)[0]
gefs_file_path = Path(gefs_folder_path) / gefs_file
with rasterio.open(gefs_file_path) as src:
    nodata_value = src.nodatavals[0]  # Get the NoData value for the first band (assuming a single band)
    print(f"NoData value: {nodata_value}")
```

    NoData value: None



```python
# Let's see the crs.. in a intuitive way, this are the limits
input_raster = rasterio.open(gefs_file_path)
input_raster.bounds
```




    BoundingBox(left=-180.25, bottom=-90.25, right=179.75, top=90.25)



Since we are in Fiji and we want information for >180 degree longitudes and the CRS of our grid is one with that goes from 0 to 360 in longitude.. we have to modify the grid crs. But it's not that simple. Let's do it by modifing manually the geometry


```python
grid_transformed = grid.copy()

# Define a function to adjust the longitude of a single polygon
def adjust_longitude(polygon):
    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)

    # Adjust longitudes from [0, 360) to [-180, 180)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)

    # Create a new Polygon with adjusted coordinates
    return Polygon(coords)

# Apply the adjust_longitude function to each geometry in the DataFrame
grid_transformed["geometry"] = grid_transformed["geometry"].apply(adjust_longitude)
grid_transformed = grid_transformed[['id', 'Latitude', 'Longitude','geometry']]
grid_transformed
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
      <th>id</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>355</td>
      <td>-17.15</td>
      <td>176.85</td>
      <td>POLYGON ((176.80000 -17.10000, 176.90000 -17.1...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>409</td>
      <td>-12.45</td>
      <td>176.95</td>
      <td>POLYGON ((176.90000 -12.40000, 177.00000 -12.4...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>456</td>
      <td>-17.15</td>
      <td>176.95</td>
      <td>POLYGON ((176.90000 -17.10000, 177.00000 -17.1...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>510</td>
      <td>-12.45</td>
      <td>177.05</td>
      <td>POLYGON ((177.00000 -12.40000, 177.10000 -12.4...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>511</td>
      <td>-12.55</td>
      <td>177.05</td>
      <td>POLYGON ((177.00000 -12.50000, 177.10000 -12.5...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>416</th>
      <td>5122</td>
      <td>-19.15</td>
      <td>181.55</td>
      <td>POLYGON ((-178.50000 -19.10000, -178.40000 -19...</td>
    </tr>
    <tr>
      <th>417</th>
      <td>5123</td>
      <td>-19.25</td>
      <td>181.55</td>
      <td>POLYGON ((-178.50000 -19.20000, -178.40000 -19...</td>
    </tr>
    <tr>
      <th>418</th>
      <td>5221</td>
      <td>-18.95</td>
      <td>181.65</td>
      <td>POLYGON ((-178.40000 -18.90000, -178.30000 -18...</td>
    </tr>
    <tr>
      <th>419</th>
      <td>5223</td>
      <td>-19.15</td>
      <td>181.65</td>
      <td>POLYGON ((-178.40000 -19.10000, -178.30000 -19...</td>
    </tr>
    <tr>
      <th>420</th>
      <td>5331</td>
      <td>-19.85</td>
      <td>181.75</td>
      <td>POLYGON ((-178.30000 -19.80000, -178.20000 -19...</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 4 columns</p>
</div>



Now lets get the max and mean rainfall by grid


```python
# List of statistics to compute
stats_list = ["mean", "max"]
grid_list = []
# Create a loop for running through all GEFS files
for gefs_file in sorted(os.listdir(gefs_folder_path)):
    if gefs_file.startswith("geprcp"):
        gefs_file_path = Path(gefs_folder_path) / gefs_file
        input_raster = rasterio.open(gefs_file_path)
        array = input_raster.read(1)

        # Compute statistics for grid cells
        summary_stats = zonal_stats(
                    grid_transformed,
                    array,
                    stats=stats_list,
                    nodata=-999, # There's no specificaiton on this, so we invent a number
                    all_touched=True,
                    affine=input_raster.transform,
                )

        grid_stats = pd.DataFrame(summary_stats)

        # Change values to mm/hr
        if gefs_file[:-5].endswith('06'):
            grid_stats[stats_list] /= 6 # For 6h accumulation data
        else:
            grid_stats[stats_list] /= 24 # For 24h accumulation data

        # Set appropriate date and time information
        forecast_hours = int(gefs_file.split(".")[-1][-3:])  # Extract forecast hours from the filename
        release_hour = int(gefs_file.split(".")[1][1:3])  # Extract release hour from the filename
        release_date = datetime.now().replace(hour=release_hour, minute=0, second=0, microsecond=0)  # Set release date
        forecast_date = release_date + timedelta(hours=forecast_hours)  # Calculate forecast date

        # Merge grid statistics with grid data
        grid_merged = pd.concat([grid_transformed, grid_stats], axis=1)
        # Set appropriate date and time information (modify as per GEFS data format)
        grid_merged["date"] = forecast_date.strftime("%Y%m%d%H")  # Format date as string
        grid_merged = grid_merged[['id', 'max', 'mean', 'date']]
        grid_list.append(grid_merged)

        # Save the processed data. Name file by time and put in folder 06 or 24 regarding rainfall accumulation.
        grid_date = forecast_date.strftime("%Y%m%d%H")
        if gefs_file[:-5].endswith('06'):
            # Set out dir
            outdir = processed_output_dir_grid / "06"
            os.makedirs(outdir, exist_ok=True)
        else:
            # Set out dir
            outdir = processed_output_dir_grid / "24"
            os.makedirs(outdir, exist_ok=True)

        grid_merged.to_csv(
            outdir / f"{grid_date}_gridstats.csv",
            index=False,
        )
```

Obs: we have nans values! Thats ok, we dont have data for every cell. As we did with the model, we put 0s in here.


```python
grid_list[1][grid_list[0].isna().any(axis=1)].head()
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
      <th>id</th>
      <th>max</th>
      <th>mean</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>304</th>
      <td>3375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023111318</td>
    </tr>
    <tr>
      <th>305</th>
      <td>3376</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023111318</td>
    </tr>
    <tr>
      <th>306</th>
      <td>3377</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023111318</td>
    </tr>
    <tr>
      <th>307</th>
      <td>3378</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023111318</td>
    </tr>
    <tr>
      <th>308</th>
      <td>3379</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023111318</td>
    </tr>
  </tbody>
</table>
</div>



Let's create an output more similar to the one of the Philippines.


```python
for folder in ["06", "24"]:
    list_df = []
    for file in sorted(os.listdir(processed_output_dir_grid / folder)):
        file_path = processed_output_dir_grid / folder / file
        df_aux = pd.read_csv(file_path)
        # Put 0 in nans values
        df_aux = df_aux.fillna(0)
        list_df.append(df_aux)

    final_df = pd.concat(list_df)
    # Convert the "date" column to datetime type
    final_df["date"] = pd.to_datetime(final_df["date"], format="%Y%m%d%H")

    # Separate DataFrames for "mean" and "max" statistics
    df_pivot_mean = final_df.pivot_table(index=["id"], columns="date", values="mean")
    df_pivot_max = final_df.pivot_table(index=["id"], columns="date", values="max")

    # Flatten the multi-level column index
    df_pivot_mean.columns = [f"{col.strftime('%Y-%m-%d %H:%M:%S')}" for col in df_pivot_mean.columns]
    df_pivot_max.columns = [f"{col.strftime('%Y-%m-%d %H:%M:%S')}" for col in df_pivot_max.columns]

    # Reset the index
    df_pivot_mean.reset_index(inplace=True)
    df_pivot_max.reset_index(inplace=True)

    # Save .csv
    outdir = processed_output_dir / folder
    os.makedirs(outdir, exist_ok=True)
    df_pivot_mean.to_csv(
                outdir / str("gridstats_" + "mean" + ".csv"),
                index=False,
            )
    df_pivot_max.to_csv(
                outdir / str("gridstats_" + "max" + ".csv"),
                index=False,
            )
```

## Compute max Rainfall in 6H and in 24H time windows


```python
events = event_metadata.event.to_list()
time_frame_24 = 4  # in 6 hour periods
time_frame_6 = 1  # in 6 hour periods

for stats in ["mean", "max"]:
    df_rainfall_final = pd.DataFrame()
    for event in events:
        # Getting event info
        df_info = event_metadata[event_metadata["event"] == event]

        # End date is the end date of the forecast
        end_date = df_info['end_date']
        # Start date is starting day of the forecast
        start_date = df_info['start_date']

        # Loading the data (6h and 24h accumulation)
        df_rainfall_06 = pd.read_csv(
            processed_output_dir / "06" /str("gridstats_" + stats + ".csv")
        )
        df_rainfall_24 = pd.read_csv(
            processed_output_dir / "24" /str("gridstats_" + stats + ".csv")
        )

        # Focus on event dates (df_rainfall_06.iloc[3:] and df_rainfall_24.iloc[1:] have the same columns)
        available_dates_t = [
            date for date in df_rainfall_24.columns[1:]
            if (pd.to_datetime(date) >= start_date.iloc[0]) & (pd.to_datetime(date) < end_date.iloc[0])
        ]
        # Restrict to event dates
        df_rainfall_06 = df_rainfall_06[['id']+available_dates_t]
        df_rainfall_24 = df_rainfall_24[['id']+available_dates_t]

        # Create new dataframe
        df_mean_rainfall = pd.DataFrame(
                {"id": df_rainfall_24["id"]}
        )

        df_mean_rainfall["rainfall_max_6h"] = (
            df_rainfall_06.iloc[:, 1:]
                .T.rolling(time_frame_6)
                .mean()
                .max(axis=0)
        )

        df_mean_rainfall["rainfall_max_24h"] = (
            df_rainfall_24.iloc[:, 1:]
                .T.rolling(time_frame_24)
                .mean()
                .max(axis=0)
        )
        df_rainfall_single = df_mean_rainfall[
            [
                "id",
                "rainfall_max_6h",
                "rainfall_max_24h",
            ]
        ]
        df_rainfall_single["event"] = event
        df_rainfall_final = pd.concat([df_rainfall_final, df_rainfall_single])
    # Saving everything
    outdir = output_dir / "NOMADS" / today
    os.makedirs(outdir, exist_ok=True)
    df_rainfall_final.to_csv(
        outdir / str("rainfall_data_rw_" + stats + ".csv"), index=False
    )
```


```python
import matplotlib.pyplot as plt
# Loading the data
df_rainfall_24 = pd.read_csv(
    processed_output_dir / "24" /str("gridstats_" + stats + ".csv")
)
example_mean = df_rainfall_24.iloc[:, 1:].T.rolling(time_frame_24).mean().max(axis=0)
example_median = df_rainfall_24.iloc[:, 1:].T.rolling(time_frame_24).median().max(axis=0)

plt.plot(example_mean, example_median, 'o')
plt.title('Max rainfall value over multiple windows')
plt.xlabel('Mean rainfall value of 24h window')
plt.ylabel('Median rainfall value of 24h window')
plt.grid()
plt.show()
```



![png](rainfall_data_files/rainfall_data_25_0.png)



## Merge rainfall data and windfield data


```python
filename = output_dir / "NOMADS"/ today /"rainfall_data_rw_mean.csv"
df_rainfall = pd.read_csv(filename)
fiji_forecast.merge(df_rainfall, left_on=['unique_id','grid_point_id'], right_on=['event', 'id'])[
    ['grid_point_id',
    'event',
    'wind_speed',
    'track_distance',
    'rainfall_max_6h',
    'rainfall_max_24h',
    'time_init',
    'time_end',
    'in_fiji',
    'geometry']].sort_values('rainfall_max_6h', ascending=False)

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
      <th>grid_point_id</th>
      <th>event</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>time_init</th>
      <th>time_end</th>
      <th>in_fiji</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5036</th>
      <td>4935</td>
      <td>679</td>
      <td>8.009166</td>
      <td>186.733950</td>
      <td>11.500000</td>
      <td>3.617708</td>
      <td>2023-11-10 06:00:00</td>
      <td>2023-11-19 00:00:00</td>
      <td>True</td>
      <td>POINT (181.35000 -20.65000)</td>
    </tr>
    <tr>
      <th>2089</th>
      <td>4935</td>
      <td>664</td>
      <td>16.476359</td>
      <td>291.094721</td>
      <td>11.500000</td>
      <td>1.019792</td>
      <td>2023-11-11 06:00:00</td>
      <td>2023-11-18 06:00:00</td>
      <td>True</td>
      <td>POINT (181.35000 -20.65000)</td>
    </tr>
    <tr>
      <th>5029</th>
      <td>4834</td>
      <td>679</td>
      <td>7.410534</td>
      <td>192.591137</td>
      <td>10.283333</td>
      <td>3.285938</td>
      <td>2023-11-10 06:00:00</td>
      <td>2023-11-19 00:00:00</td>
      <td>True</td>
      <td>POINT (181.25000 -20.65000)</td>
    </tr>
    <tr>
      <th>2082</th>
      <td>4834</td>
      <td>664</td>
      <td>16.787327</td>
      <td>283.794724</td>
      <td>10.283333</td>
      <td>0.965625</td>
      <td>2023-11-11 06:00:00</td>
      <td>2023-11-18 06:00:00</td>
      <td>True</td>
      <td>POINT (181.25000 -20.65000)</td>
    </tr>
    <tr>
      <th>10871</th>
      <td>3883</td>
      <td>748</td>
      <td>10.465720</td>
      <td>240.987143</td>
      <td>8.600000</td>
      <td>8.302250</td>
      <td>2023-11-11 06:00:00</td>
      <td>2023-11-16 06:00:00</td>
      <td>True</td>
      <td>POINT (180.35000 -16.45000)</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>3481</td>
      <td>672</td>
      <td>0.000000</td>
      <td>534.262792</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2023-11-10 00:00:00</td>
      <td>2023-11-17 06:00:00</td>
      <td>True</td>
      <td>POINT (179.95000 -16.65000)</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>3480</td>
      <td>672</td>
      <td>0.000000</td>
      <td>542.288648</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2023-11-10 00:00:00</td>
      <td>2023-11-17 06:00:00</td>
      <td>True</td>
      <td>POINT (179.95000 -16.55000)</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>3479</td>
      <td>672</td>
      <td>0.000000</td>
      <td>550.421738</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2023-11-10 00:00:00</td>
      <td>2023-11-17 06:00:00</td>
      <td>True</td>
      <td>POINT (179.95000 -16.45000)</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>3477</td>
      <td>672</td>
      <td>0.000000</td>
      <td>566.991097</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2023-11-10 00:00:00</td>
      <td>2023-11-17 06:00:00</td>
      <td>True</td>
      <td>POINT (179.95000 -16.25000)</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>3482</td>
      <td>657</td>
      <td>0.000000</td>
      <td>489.407795</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2023-11-10 00:00:00</td>
      <td>2023-11-16 12:00:00</td>
      <td>True</td>
      <td>POINT (179.95000 -16.75000)</td>
    </tr>
  </tbody>
</table>
<p>13051 rows × 10 columns</p>
</div>



There are some nans: this is because for the last event (usually), since the forecast is up to 16 days and the last event is on the day 16, we have just 1 datetime to compute the rolling. This is depricable because events 16 days from now are not significative and we are going to keep just the ones in the near future (up tp 120h aprox)


```python
df_rainfall[df_rainfall.isna().any(axis=1)] # Its just the last dataset...
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
      <th>id</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12630</th>
      <td>355</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>12631</th>
      <td>409</td>
      <td>0.18</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>12632</th>
      <td>456</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>12633</th>
      <td>510</td>
      <td>0.18</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>12634</th>
      <td>511</td>
      <td>0.18</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13046</th>
      <td>5122</td>
      <td>1.66</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>13047</th>
      <td>5123</td>
      <td>0.95</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>13048</th>
      <td>5221</td>
      <td>1.66</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>13049</th>
      <td>5223</td>
      <td>1.66</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
    <tr>
      <th>13050</th>
      <td>5331</td>
      <td>0.27</td>
      <td>NaN</td>
      <td>789</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 4 columns</p>
</div>



To compare, let's see the values of rainfall for the historical data.


```python
from utils import get_training_dataset_complete
df_complete = get_training_dataset_complete()
df_complete[['rainfall_max_6h','rainfall_max_24h']].sort_values('rainfall_max_6h', ascending=False)
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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>15.700000</td>
      <td>4.856250</td>
    </tr>
    <tr>
      <th>607</th>
      <td>13.733333</td>
      <td>8.225000</td>
    </tr>
    <tr>
      <th>148</th>
      <td>13.125000</td>
      <td>6.029167</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>12.625000</td>
      <td>6.891667</td>
    </tr>
    <tr>
      <th>1129</th>
      <td>12.075000</td>
      <td>5.812500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3015</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3014</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3013</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>3789 rows × 2 columns</p>
</div>
