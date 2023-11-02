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

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_40123/386341765.py:9: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      grids.geometry = grids.geometry.to_crs(grids.crs).centroid



```python
tc_fcast = TCForecast()
tc_fcast.fetch_ecmwf()
```

    Download: 100%|██████████| 38/38 [00:53<00:00,  1.40s/ files]
    Processing: 100%|██████████| 38/38 [00:03<00:00, 10.60 files/s]



```python
tc_fcast.data[0]
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
Dimensions:                 (time: 26)
Coordinates:
  * time                    (time) datetime64[ns] 2023-11-01 ... 2023-11-08
    lat                     (time) float64 11.7 12.2 12.6 ... 11.0 11.7 12.1
    lon                     (time) float64 -89.5 -89.8 -90.7 ... -123.3 -124.3
Data variables:
    max_sustained_wind      (time) float64 20.1 22.6 24.7 ... 11.8 11.8 11.8
    central_pressure        (time) float64 998.0 998.0 ... 1.009e+03 1.005e+03
    radius_max_wind         (time) float64 31.73 24.0 32.2 ... 164.9 156.1 150.2
    time_step               (time) float64 0.0 6.0 6.0 6.0 ... 6.0 6.0 12.0 6.0
    environmental_pressure  (time) float64 1.01e+03 1.01e+03 ... 1.01e+03
    basin                   (time) &lt;U1 &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27; ... &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27;
Attributes:
    max_sustained_wind_unit:  m/s
    central_pressure_unit:    mb
    name:                     PILAR
    sid:                      19E
    orig_event_flag:          False
    data_provider:            ECMWF
    id_no:                    1.0
    ensemble_number:          1
    is_ensemble:              True
    run_datetime:             2023-11-01T00:00:00.000000
    category:                 Tropical Storm</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-c147519e-6012-4ca7-ba64-68552aaedd0b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-c147519e-6012-4ca7-ba64-68552aaedd0b' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 26</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-04fba129-8207-4ec7-90db-db2dded8ee11' class='xr-section-summary-in' type='checkbox'  checked><label for='section-04fba129-8207-4ec7-90db-db2dded8ee11' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2023-11-01 ... 2023-11-08</div><input id='attrs-e4ea9e7c-3ae9-4f1d-b9fa-ec662a23a637' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e4ea9e7c-3ae9-4f1d-b9fa-ec662a23a637' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-15489e8c-e26d-4970-b556-cc8e9ce22939' class='xr-var-data-in' type='checkbox'><label for='data-15489e8c-e26d-4970-b556-cc8e9ce22939' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2023-11-01T00:00:00.000000000&#x27;, &#x27;2023-11-01T06:00:00.000000000&#x27;,
       &#x27;2023-11-01T12:00:00.000000000&#x27;, &#x27;2023-11-01T18:00:00.000000000&#x27;,
       &#x27;2023-11-02T00:00:00.000000000&#x27;, &#x27;2023-11-02T06:00:00.000000000&#x27;,
       &#x27;2023-11-02T12:00:00.000000000&#x27;, &#x27;2023-11-02T18:00:00.000000000&#x27;,
       &#x27;2023-11-03T00:00:00.000000000&#x27;, &#x27;2023-11-03T06:00:00.000000000&#x27;,
       &#x27;2023-11-03T12:00:00.000000000&#x27;, &#x27;2023-11-03T18:00:00.000000000&#x27;,
       &#x27;2023-11-04T06:00:00.000000000&#x27;, &#x27;2023-11-04T18:00:00.000000000&#x27;,
       &#x27;2023-11-05T00:00:00.000000000&#x27;, &#x27;2023-11-05T06:00:00.000000000&#x27;,
       &#x27;2023-11-05T12:00:00.000000000&#x27;, &#x27;2023-11-05T18:00:00.000000000&#x27;,
       &#x27;2023-11-06T00:00:00.000000000&#x27;, &#x27;2023-11-06T06:00:00.000000000&#x27;,
       &#x27;2023-11-06T12:00:00.000000000&#x27;, &#x27;2023-11-06T18:00:00.000000000&#x27;,
       &#x27;2023-11-07T00:00:00.000000000&#x27;, &#x27;2023-11-07T06:00:00.000000000&#x27;,
       &#x27;2023-11-07T18:00:00.000000000&#x27;, &#x27;2023-11-08T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>11.7 12.2 12.6 ... 11.0 11.7 12.1</div><input id='attrs-f754dd8c-2b05-44fb-b487-af15f2b96493' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f754dd8c-2b05-44fb-b487-af15f2b96493' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c328cf33-f4b7-4d2c-8812-455d2b796f0d' class='xr-var-data-in' type='checkbox'><label for='data-c328cf33-f4b7-4d2c-8812-455d2b796f0d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([11.7, 12.2, 12.6, 12.3, 12.1, 11.7, 11.1, 10.4,  9.9,  9.3,  8.8,
        8.4,  8.4,  8.6,  8.8,  9.1,  9.2,  9.4,  9.7, 10. , 10.4, 10.6,
       10.7, 11. , 11.7, 12.1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-89.5 -89.8 -90.7 ... -123.3 -124.3</div><input id='attrs-0c871154-c9df-4a6a-9147-4f61f1e9234f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0c871154-c9df-4a6a-9147-4f61f1e9234f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-99065fe3-8c2a-4605-89c9-431b854a56c2' class='xr-var-data-in' type='checkbox'><label for='data-99065fe3-8c2a-4605-89c9-431b854a56c2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ -89.5,  -89.8,  -90.7,  -91. ,  -92.2,  -93.5,  -94.9,  -96.3,
        -97.9,  -99.8, -101.8, -103.6, -106.6, -109.8, -111.1, -112.3,
       -113.7, -115. , -116.2, -117.3, -118.3, -119.1, -120.2, -121.2,
       -123.3, -124.3])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-0cb83b31-ef2b-4e3f-93b9-148bbef64fb4' class='xr-section-summary-in' type='checkbox'  checked><label for='section-0cb83b31-ef2b-4e3f-93b9-148bbef64fb4' class='xr-section-summary' >Data variables: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>max_sustained_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>20.1 22.6 24.7 ... 11.8 11.8 11.8</div><input id='attrs-3227f9b1-f5c9-43e0-88bf-a2cfce152858' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3227f9b1-f5c9-43e0-88bf-a2cfce152858' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d08bb159-c06c-4230-ab47-936befd7b582' class='xr-var-data-in' type='checkbox'><label for='data-d08bb159-c06c-4230-ab47-936befd7b582' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([20.1, 22.6, 24.7, 21.1, 20.1, 23.7, 24.2, 21.1, 19. , 19. , 19. ,
       17.5, 14.4, 14.4, 12.9, 13.4, 12.9, 11.8, 11.8, 11.3, 11.3, 12.4,
       12.4, 11.8, 11.8, 11.8])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>central_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>998.0 998.0 ... 1.009e+03 1.005e+03</div><input id='attrs-711d8cbd-905b-49d7-b0e0-8a7adc711dca' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-711d8cbd-905b-49d7-b0e0-8a7adc711dca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b350cdb6-335b-4cc0-a8ea-100accc9679b' class='xr-var-data-in' type='checkbox'><label for='data-b350cdb6-335b-4cc0-a8ea-100accc9679b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 998.,  998., 1006., 1000.,  999., 1000.,  999., 1002., 1003.,
       1007., 1004., 1006., 1007., 1008., 1005., 1007., 1004., 1007.,
       1004., 1007., 1004., 1007., 1004., 1007., 1009., 1005.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>radius_max_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>31.73 24.0 32.2 ... 156.1 150.2</div><input id='attrs-590ef122-b072-4081-bc67-95a41bd9eaf5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-590ef122-b072-4081-bc67-95a41bd9eaf5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d29f4a18-868f-4749-aa43-ba6e1eafbcb2' class='xr-var-data-in' type='checkbox'><label for='data-d29f4a18-868f-4749-aa43-ba6e1eafbcb2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 31.73309915,  24.        ,  32.20477523,  12.        ,
        13.35732616,   8.3975939 ,  13.36658784,  13.37261613,
       315.99517917,  24.71962442,  24.        ,  18.        ,
        18.        ,  24.72235851,  24.72159849,  18.94992076,
        25.29264545,  32.25149216,  90.19411293, 192.77474834,
       198.76595117, 126.90275644, 222.42185005, 164.90489908,
       156.12866349, 150.21054472])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>time_step</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 6.0 6.0 6.0 ... 6.0 12.0 6.0</div><input id='attrs-09be9c6e-1df2-435f-b62f-f3851121d690' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-09be9c6e-1df2-435f-b62f-f3851121d690' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9a406d7b-5764-4009-ba52-d1fe23ba86c0' class='xr-var-data-in' type='checkbox'><label for='data-9a406d7b-5764-4009-ba52-d1fe23ba86c0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6., 12.,
       12.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6., 12.,  6.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>environmental_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.01e+03 1.01e+03 ... 1.01e+03</div><input id='attrs-e6dcb31f-98ef-4b2e-928f-655fa9819fb5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e6dcb31f-98ef-4b2e-928f-655fa9819fb5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c2de076b-7559-4837-ba41-6eeb0ee089e8' class='xr-var-data-in' type='checkbox'><label for='data-c2de076b-7559-4837-ba41-6eeb0ee089e8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010.,
       1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010.,
       1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>basin</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U1</div><div class='xr-var-preview xr-preview'>&#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27; ... &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27; &#x27;E&#x27;</div><input id='attrs-8dc53d99-47c4-44ce-a793-a6b1b06f479a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8dc53d99-47c4-44ce-a793-a6b1b06f479a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d48906d3-c8fc-4f2b-9fa8-a4b250896980' class='xr-var-data-in' type='checkbox'><label for='data-d48906d3-c8fc-4f2b-9fa8-a4b250896980' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;,
       &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;, &#x27;E&#x27;],
      dtype=&#x27;&lt;U1&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4d05368f-e799-460c-9c2e-b03c0988e950' class='xr-section-summary-in' type='checkbox'  ><label for='section-4d05368f-e799-460c-9c2e-b03c0988e950' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-00522db6-0e14-4559-bff6-a809b349cbd6' class='xr-index-data-in' type='checkbox'/><label for='index-00522db6-0e14-4559-bff6-a809b349cbd6' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2023-11-01 00:00:00&#x27;, &#x27;2023-11-01 06:00:00&#x27;,
               &#x27;2023-11-01 12:00:00&#x27;, &#x27;2023-11-01 18:00:00&#x27;,
               &#x27;2023-11-02 00:00:00&#x27;, &#x27;2023-11-02 06:00:00&#x27;,
               &#x27;2023-11-02 12:00:00&#x27;, &#x27;2023-11-02 18:00:00&#x27;,
               &#x27;2023-11-03 00:00:00&#x27;, &#x27;2023-11-03 06:00:00&#x27;,
               &#x27;2023-11-03 12:00:00&#x27;, &#x27;2023-11-03 18:00:00&#x27;,
               &#x27;2023-11-04 06:00:00&#x27;, &#x27;2023-11-04 18:00:00&#x27;,
               &#x27;2023-11-05 00:00:00&#x27;, &#x27;2023-11-05 06:00:00&#x27;,
               &#x27;2023-11-05 12:00:00&#x27;, &#x27;2023-11-05 18:00:00&#x27;,
               &#x27;2023-11-06 00:00:00&#x27;, &#x27;2023-11-06 06:00:00&#x27;,
               &#x27;2023-11-06 12:00:00&#x27;, &#x27;2023-11-06 18:00:00&#x27;,
               &#x27;2023-11-07 00:00:00&#x27;, &#x27;2023-11-07 06:00:00&#x27;,
               &#x27;2023-11-07 18:00:00&#x27;, &#x27;2023-11-08 00:00:00&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;time&#x27;, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f1a66572-6995-413c-a0c5-65def40fe1a3' class='xr-section-summary-in' type='checkbox'  ><label for='section-f1a66572-6995-413c-a0c5-65def40fe1a3' class='xr-section-summary' >Attributes: <span>(11)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>max_sustained_wind_unit :</span></dt><dd>m/s</dd><dt><span>central_pressure_unit :</span></dt><dd>mb</dd><dt><span>name :</span></dt><dd>PILAR</dd><dt><span>sid :</span></dt><dd>19E</dd><dt><span>orig_event_flag :</span></dt><dd>False</dd><dt><span>data_provider :</span></dt><dd>ECMWF</dd><dt><span>id_no :</span></dt><dd>1.0</dd><dt><span>ensemble_number :</span></dt><dd>1</dd><dt><span>is_ensemble :</span></dt><dd>True</dd><dt><span>run_datetime :</span></dt><dd>2023-11-01T00:00:00.000000</dd><dt><span>category :</span></dt><dd>Tropical Storm</dd></dl></div></li></ul></div></div>




```python
tc_fcast.plot()
```




    <GeoAxes: >





![png](wind_to_grid_ECMWF_files/wind_to_grid_ECMWF_4_1.png)




```python
cent = Centroids.from_geodataframe(grids)
tc = TropCyclone.from_tracks(
    tc_fcast, centroids=cent, store_windfields=True, intensity_thres=0
)
```

    2023-11-01 13:45:31,507 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.



```python
np.unique(tc_track.time)[-1]
```




    numpy.datetime64('2023-11-09T06:00:00.000000000')




```python
event_names = list(tc.event_name)

# Define the boundaries for Fiji region
xmin, xmax, ymin, ymax = 176, 182, -21, -12
fiji_polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

df_windfield = pd.DataFrame()
for i, intensity_sparse in enumerate(tc.intensity):
    # Get the windfield
    windfield = intensity_sparse.toarray().flatten()
    npoints = len(windfield)
    event_id = event_names[i]

    # Track distance
    DEG_TO_KM = 111.1
    tc_track = tc_fcast.get_track()[i]
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

    /Users/federico/anaconda3/envs/env6/lib/python3.11/site-packages/shapely/measurement.py:72: RuntimeWarning: invalid value encountered in distance
      return lib.distance(a, b, **kwargs)



```python
fiji_forecast = df_windfield[df_windfield.in_fiji == True]
events_fiji = fiji_forecast.unique_id.unique()

fiji_forecast[fiji_forecast.unique_id == events_fiji[0]]
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
      <th>event_id_ecmwf</th>
      <th>unique_id</th>
      <th>basins</th>
      <th>time_init</th>
      <th>time_end</th>
      <th>in_fiji</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>272332</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>175</td>
      <td>8.364638</td>
      <td>207.997478</td>
      <td>POINT (176.85000 -17.15000)</td>
    </tr>
    <tr>
      <th>272333</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>226</td>
      <td>18.161678</td>
      <td>198.896928</td>
      <td>POINT (176.95000 -17.15000)</td>
    </tr>
    <tr>
      <th>272334</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>278</td>
      <td>8.917662</td>
      <td>197.026383</td>
      <td>POINT (177.05000 -17.25000)</td>
    </tr>
    <tr>
      <th>272335</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>280</td>
      <td>8.216096</td>
      <td>212.110821</td>
      <td>POINT (177.05000 -17.45000)</td>
    </tr>
    <tr>
      <th>272336</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>281</td>
      <td>7.869920</td>
      <td>220.107018</td>
      <td>POINT (177.05000 -17.55000)</td>
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
      <th>272739</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>2592</td>
      <td>0.000000</td>
      <td>487.132643</td>
      <td>POINT (181.55000 -19.15000)</td>
    </tr>
    <tr>
      <th>272740</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>2593</td>
      <td>0.000000</td>
      <td>494.925249</td>
      <td>POINT (181.55000 -19.25000)</td>
    </tr>
    <tr>
      <th>272741</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>2641</td>
      <td>0.000000</td>
      <td>480.242521</td>
      <td>POINT (181.65000 -18.95000)</td>
    </tr>
    <tr>
      <th>272742</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>2643</td>
      <td>0.000000</td>
      <td>495.174582</td>
      <td>POINT (181.65000 -19.15000)</td>
    </tr>
    <tr>
      <th>272743</th>
      <td>71P</td>
      <td>661</td>
      <td>[P]</td>
      <td>2023-11-07 06:00:00</td>
      <td>2023-11-11</td>
      <td>True</td>
      <td>2701</td>
      <td>0.000000</td>
      <td>558.657377</td>
      <td>POINT (181.75000 -19.85000)</td>
    </tr>
  </tbody>
</table>
<p>412 rows × 10 columns</p>
</div>




```python
event1 = fiji_forecast[fiji_forecast.unique_id == events_fiji[0]]
gdf_aux = gpd.GeoDataFrame(event1)

# Plot
fig, ax = plt.subplots(1,1)
gdf_aux.plot(ax=ax, column='wind_speed', cmap='coolwarm', markersize=20, legend=True, label= 'Wind Speed [m/s]')
```




    <Axes: >





![png](wind_to_grid_ECMWF_files/wind_to_grid_ECMWF_9_1.png)




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
      <th>IWI</th>
      <th>total_buildings</th>
      <th>with_coast</th>
      <th>coast_length</th>
      <th>mean_altitude</th>
      <th>mean_slope</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>event_id_ecmwf</th>
      <th>unique_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>175</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.364638</td>
      <td>207.997478</td>
      <td>71P</td>
      <td>661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>175</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>476.112436</td>
      <td>71P</td>
      <td>674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>175</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.519369</td>
      <td>138.692749</td>
      <td>71P</td>
      <td>677</td>
    </tr>
    <tr>
      <th>3</th>
      <td>175</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>385.262378</td>
      <td>71P</td>
      <td>678</td>
    </tr>
    <tr>
      <th>4</th>
      <td>175</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24.397475</td>
      <td>121.866768</td>
      <td>71P</td>
      <td>679</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>4115</th>
      <td>2701</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>924.102258</td>
      <td>71P</td>
      <td>686</td>
    </tr>
    <tr>
      <th>4116</th>
      <td>2701</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1074.544359</td>
      <td>72P</td>
      <td>697</td>
    </tr>
    <tr>
      <th>4117</th>
      <td>2701</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.953982</td>
      <td>19.981376</td>
      <td>72P</td>
      <td>704</td>
    </tr>
    <tr>
      <th>4118</th>
      <td>2701</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>565.683409</td>
      <td>72P</td>
      <td>708</td>
    </tr>
    <tr>
      <th>4119</th>
      <td>2701</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>329.046350</td>
      <td>72P</td>
      <td>709</td>
    </tr>
  </tbody>
</table>
<p>4120 rows × 11 columns</p>
</div>
