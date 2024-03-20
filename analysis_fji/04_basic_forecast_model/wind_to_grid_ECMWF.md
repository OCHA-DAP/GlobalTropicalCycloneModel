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
    / "analysis_fji/02_model_features/02_housing_damage/output/"
)

# Load grid and stationary data
df = get_stationary_data_fiji()
grids = gpd.read_file(output_dir / "fji_0.1_degree_grid_land_overlap_new.gpkg")
grids.geometry = grids.geometry.to_crs(grids.crs).centroid
df_stationary = df.merge(grids, right_on='id', left_on='grid_point_id').drop(['index', 'id'], axis=1)
```

    /var/folders/dy/vms3cfrn4q9952h8s6l586dr0000gp/T/ipykernel_52863/386341765.py:9: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.

      grids.geometry = grids.geometry.to_crs(grids.crs).centroid



```python
tc_fcast = TCForecast()
tc_fcast.fetch_ecmwf()
```

    Download: 100%|██████████| 29/29 [00:37<00:00,  1.28s/ files]
    Processing: 100%|██████████| 29/29 [00:02<00:00,  9.85 files/s]



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
Dimensions:                 (time: 40)
Coordinates:
  * time                    (time) datetime64[ns] 2023-12-07 ... 2023-12-17
    lat                     (time) float64 -11.9 -12.6 -12.8 ... -16.6 -15.7
    lon                     (time) float64 157.0 157.0 156.4 ... 139.0 138.4
Data variables:
    max_sustained_wind      (time) float64 32.4 30.9 31.9 ... 11.8 14.4 15.4
    central_pressure        (time) float64 979.0 977.0 ... 1.001e+03 1.002e+03
    radius_max_wind         (time) float64 48.12 30.57 29.73 ... 39.92 8.329
    time_step               (time) float64 0.0 6.0 6.0 6.0 ... 6.0 12.0 6.0 6.0
    environmental_pressure  (time) float64 1.01e+03 1.01e+03 ... 1.01e+03
    basin                   (time) &lt;U1 &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27; ... &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27;
Attributes:
    max_sustained_wind_unit:  m/s
    central_pressure_unit:    mb
    name:                     JASPER
    sid:                      03P
    orig_event_flag:          False
    data_provider:            ECMWF
    id_no:                    1.0
    ensemble_number:          1
    is_ensemble:              True
    run_datetime:             2023-12-07T00:00:00.000000
    category:                 Hurricane Cat. 1</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-bddb10c1-45bd-4f90-9ac2-1d876ec156d8' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-bddb10c1-45bd-4f90-9ac2-1d876ec156d8' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 40</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-84f6a988-4a0f-4829-9542-f2cacac1bb27' class='xr-section-summary-in' type='checkbox'  checked><label for='section-84f6a988-4a0f-4829-9542-f2cacac1bb27' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2023-12-07 ... 2023-12-17</div><input id='attrs-d2c056e5-1ee3-4f3b-9773-fe7ce735c206' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d2c056e5-1ee3-4f3b-9773-fe7ce735c206' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e10be5ca-f3e4-4d79-b0bc-4c4a0675611d' class='xr-var-data-in' type='checkbox'><label for='data-e10be5ca-f3e4-4d79-b0bc-4c4a0675611d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2023-12-07T00:00:00.000000000&#x27;, &#x27;2023-12-07T06:00:00.000000000&#x27;,
       &#x27;2023-12-07T12:00:00.000000000&#x27;, &#x27;2023-12-07T18:00:00.000000000&#x27;,
       &#x27;2023-12-08T00:00:00.000000000&#x27;, &#x27;2023-12-08T06:00:00.000000000&#x27;,
       &#x27;2023-12-08T12:00:00.000000000&#x27;, &#x27;2023-12-08T18:00:00.000000000&#x27;,
       &#x27;2023-12-09T00:00:00.000000000&#x27;, &#x27;2023-12-09T06:00:00.000000000&#x27;,
       &#x27;2023-12-09T12:00:00.000000000&#x27;, &#x27;2023-12-09T18:00:00.000000000&#x27;,
       &#x27;2023-12-10T00:00:00.000000000&#x27;, &#x27;2023-12-10T06:00:00.000000000&#x27;,
       &#x27;2023-12-10T12:00:00.000000000&#x27;, &#x27;2023-12-10T18:00:00.000000000&#x27;,
       &#x27;2023-12-11T00:00:00.000000000&#x27;, &#x27;2023-12-11T06:00:00.000000000&#x27;,
       &#x27;2023-12-11T12:00:00.000000000&#x27;, &#x27;2023-12-11T18:00:00.000000000&#x27;,
       &#x27;2023-12-12T00:00:00.000000000&#x27;, &#x27;2023-12-12T06:00:00.000000000&#x27;,
       &#x27;2023-12-12T12:00:00.000000000&#x27;, &#x27;2023-12-12T18:00:00.000000000&#x27;,
       &#x27;2023-12-13T00:00:00.000000000&#x27;, &#x27;2023-12-13T06:00:00.000000000&#x27;,
       &#x27;2023-12-13T12:00:00.000000000&#x27;, &#x27;2023-12-13T18:00:00.000000000&#x27;,
       &#x27;2023-12-14T00:00:00.000000000&#x27;, &#x27;2023-12-14T06:00:00.000000000&#x27;,
       &#x27;2023-12-14T12:00:00.000000000&#x27;, &#x27;2023-12-14T18:00:00.000000000&#x27;,
       &#x27;2023-12-15T00:00:00.000000000&#x27;, &#x27;2023-12-15T06:00:00.000000000&#x27;,
       &#x27;2023-12-15T12:00:00.000000000&#x27;, &#x27;2023-12-15T18:00:00.000000000&#x27;,
       &#x27;2023-12-16T00:00:00.000000000&#x27;, &#x27;2023-12-16T12:00:00.000000000&#x27;,
       &#x27;2023-12-16T18:00:00.000000000&#x27;, &#x27;2023-12-17T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-11.9 -12.6 -12.8 ... -16.6 -15.7</div><input id='attrs-e12a7646-36de-4963-978c-ec8c814fdef7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e12a7646-36de-4963-978c-ec8c814fdef7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d8a5862b-d9e6-4c9e-9ac2-b94938f53fce' class='xr-var-data-in' type='checkbox'><label for='data-d8a5862b-d9e6-4c9e-9ac2-b94938f53fce' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-11.9, -12.6, -12.8, -13.5, -13.8, -13.8, -13.7, -13.7, -13.8,
       -13.9, -13.9, -14. , -14.8, -15.1, -15.4, -15.8, -15.9, -16.4,
       -16.9, -16.9, -16.5, -16.4, -16.2, -16.4, -16.6, -16.9, -16.8,
       -16.8, -16.8, -17.1, -17.1, -16.8, -17. , -17.1, -17.1, -17.3,
       -17.4, -16.7, -16.6, -15.7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>157.0 157.0 156.4 ... 139.0 138.4</div><input id='attrs-5db27c72-ed71-41d9-95f3-d3ac70e3f69b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5db27c72-ed71-41d9-95f3-d3ac70e3f69b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bf78e2be-732e-4543-9e5c-9f3a0cd35640' class='xr-var-data-in' type='checkbox'><label for='data-bf78e2be-732e-4543-9e5c-9f3a0cd35640' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([157. , 157. , 156.4, 156.3, 155.7, 155.6, 155.3, 154.8, 154.3,
       154.1, 153.9, 153.5, 153.2, 152.9, 152.7, 152.4, 152.1, 151.8,
       151.3, 150.3, 150.1, 149.6, 149.4, 149.4, 149. , 148.7, 148.2,
       147.6, 147.5, 146.9, 146.3, 145.8, 145.3, 144.2, 143.3, 142.4,
       141.3, 140.4, 139. , 138.4])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-535fab84-72c6-4a6d-b53c-d660b4a79bfb' class='xr-section-summary-in' type='checkbox'  checked><label for='section-535fab84-72c6-4a6d-b53c-d660b4a79bfb' class='xr-section-summary' >Data variables: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>max_sustained_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>32.4 30.9 31.9 ... 11.8 14.4 15.4</div><input id='attrs-18a6a682-e57c-4350-b914-e28dc7003b70' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-18a6a682-e57c-4350-b914-e28dc7003b70' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-05128185-9d76-443c-9148-50e09b345e17' class='xr-var-data-in' type='checkbox'><label for='data-05128185-9d76-443c-9148-50e09b345e17' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([32.4, 30.9, 31.9, 35.5, 29.3, 39.6, 32.4, 24.2, 25.2, 22.6, 24.7,
       23.7, 27.8, 29.3, 34. , 27.3, 28.3, 24.7, 27.8, 29.8, 26.2, 23.2,
       23.7, 24.7, 25.7, 28.8, 26.8, 26.2, 24.7, 26.2, 30.9, 24.2, 22.1,
       17. , 14.9, 13.9, 10.3, 11.8, 14.4, 15.4])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>central_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>979.0 977.0 ... 1.001e+03 1.002e+03</div><input id='attrs-20d02537-1da5-49e3-acb9-e0e654cd7f94' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-20d02537-1da5-49e3-acb9-e0e654cd7f94' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9666a5d1-7dfe-4188-a2e1-7a696b85bcec' class='xr-var-data-in' type='checkbox'><label for='data-9666a5d1-7dfe-4188-a2e1-7a696b85bcec' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 979.,  977.,  979.,  975.,  978.,  971.,  978.,  980.,  983.,
        983.,  986.,  981.,  980.,  972.,  971.,  973.,  972.,  974.,
        977.,  972.,  974.,  975.,  981.,  977.,  980.,  976.,  982.,
        981.,  985.,  984.,  984.,  987.,  994.,  995., 1000., 1000.,
       1004., 1004., 1001., 1002.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>radius_max_wind</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>48.12 30.57 29.73 ... 39.92 8.329</div><input id='attrs-12bf6594-f30e-4eb7-91bf-424acef8554a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-12bf6594-f30e-4eb7-91bf-424acef8554a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b082ce5f-46f1-4bc5-bd0e-16aebb289149' class='xr-var-data-in' type='checkbox'><label for='data-b082ce5f-46f1-4bc5-bd0e-16aebb289149' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 48.11972968,  30.56610683,  29.73376648,  24.69894972,
        18.        ,  26.67970498,  34.72500673,  32.18575318,
        50.18226698,  56.14322255,  48.02873433,  47.29439477,
        18.91166066,  11.58567157,  13.0324638 ,  18.32975077,
        16.64909228,  16.62891444,  16.60815709,  18.89332476,
        21.36315241,  18.28036624,  13.31157026,  21.02788125,
        18.26351058,  13.30254564,  18.89424733,  13.30385592,
        29.33964141,  34.58304575,  18.22051188,   6.        ,
        83.80177075, 155.41768387, 168.03078805, 217.68546229,
        53.18355923,  51.00240537,  39.91936646,   8.32850013])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>time_step</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 6.0 6.0 6.0 ... 12.0 6.0 6.0</div><input id='attrs-4a3b7094-cb0c-4249-b108-693206ecc97b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4a3b7094-cb0c-4249-b108-693206ecc97b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d4313abe-da7b-4200-8152-5379013ab93a' class='xr-var-data-in' type='checkbox'><label for='data-d4313abe-da7b-4200-8152-5379013ab93a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,
        6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,
        6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6., 12.,  6.,
        6.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>environmental_pressure</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.01e+03 1.01e+03 ... 1.01e+03</div><input id='attrs-4058fcb7-a807-4243-a4e2-3adfc66bbc2b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4058fcb7-a807-4243-a4e2-3adfc66bbc2b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3b36ec0c-4172-45da-a4e7-5fd5dbe697ba' class='xr-var-data-in' type='checkbox'><label for='data-3b36ec0c-4172-45da-a4e7-5fd5dbe697ba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010.,
       1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010.,
       1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010.,
       1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010., 1010.,
       1010., 1010., 1010., 1010.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>basin</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U1</div><div class='xr-var-preview xr-preview'>&#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27; ... &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27; &#x27;P&#x27;</div><input id='attrs-e516d352-743c-4c21-a6be-cad19fb16769' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e516d352-743c-4c21-a6be-cad19fb16769' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1b95a717-aca5-4655-8fdf-76f20a35f21c' class='xr-var-data-in' type='checkbox'><label for='data-1b95a717-aca5-4655-8fdf-76f20a35f21c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;,
       &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;,
       &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;, &#x27;P&#x27;,
       &#x27;P&#x27;], dtype=&#x27;&lt;U1&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-705b772c-831a-4f76-85f0-d813b2a41f1d' class='xr-section-summary-in' type='checkbox'  ><label for='section-705b772c-831a-4f76-85f0-d813b2a41f1d' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-4e120ade-a1ed-41bc-94b2-d5e270c532a0' class='xr-index-data-in' type='checkbox'/><label for='index-4e120ade-a1ed-41bc-94b2-d5e270c532a0' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;2023-12-07 00:00:00&#x27;, &#x27;2023-12-07 06:00:00&#x27;,
               &#x27;2023-12-07 12:00:00&#x27;, &#x27;2023-12-07 18:00:00&#x27;,
               &#x27;2023-12-08 00:00:00&#x27;, &#x27;2023-12-08 06:00:00&#x27;,
               &#x27;2023-12-08 12:00:00&#x27;, &#x27;2023-12-08 18:00:00&#x27;,
               &#x27;2023-12-09 00:00:00&#x27;, &#x27;2023-12-09 06:00:00&#x27;,
               &#x27;2023-12-09 12:00:00&#x27;, &#x27;2023-12-09 18:00:00&#x27;,
               &#x27;2023-12-10 00:00:00&#x27;, &#x27;2023-12-10 06:00:00&#x27;,
               &#x27;2023-12-10 12:00:00&#x27;, &#x27;2023-12-10 18:00:00&#x27;,
               &#x27;2023-12-11 00:00:00&#x27;, &#x27;2023-12-11 06:00:00&#x27;,
               &#x27;2023-12-11 12:00:00&#x27;, &#x27;2023-12-11 18:00:00&#x27;,
               &#x27;2023-12-12 00:00:00&#x27;, &#x27;2023-12-12 06:00:00&#x27;,
               &#x27;2023-12-12 12:00:00&#x27;, &#x27;2023-12-12 18:00:00&#x27;,
               &#x27;2023-12-13 00:00:00&#x27;, &#x27;2023-12-13 06:00:00&#x27;,
               &#x27;2023-12-13 12:00:00&#x27;, &#x27;2023-12-13 18:00:00&#x27;,
               &#x27;2023-12-14 00:00:00&#x27;, &#x27;2023-12-14 06:00:00&#x27;,
               &#x27;2023-12-14 12:00:00&#x27;, &#x27;2023-12-14 18:00:00&#x27;,
               &#x27;2023-12-15 00:00:00&#x27;, &#x27;2023-12-15 06:00:00&#x27;,
               &#x27;2023-12-15 12:00:00&#x27;, &#x27;2023-12-15 18:00:00&#x27;,
               &#x27;2023-12-16 00:00:00&#x27;, &#x27;2023-12-16 12:00:00&#x27;,
               &#x27;2023-12-16 18:00:00&#x27;, &#x27;2023-12-17 00:00:00&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;time&#x27;, freq=None))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2e54675d-ca39-456d-9619-83a3510895d6' class='xr-section-summary-in' type='checkbox'  ><label for='section-2e54675d-ca39-456d-9619-83a3510895d6' class='xr-section-summary' >Attributes: <span>(11)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>max_sustained_wind_unit :</span></dt><dd>m/s</dd><dt><span>central_pressure_unit :</span></dt><dd>mb</dd><dt><span>name :</span></dt><dd>JASPER</dd><dt><span>sid :</span></dt><dd>03P</dd><dt><span>orig_event_flag :</span></dt><dd>False</dd><dt><span>data_provider :</span></dt><dd>ECMWF</dd><dt><span>id_no :</span></dt><dd>1.0</dd><dt><span>ensemble_number :</span></dt><dd>1</dd><dt><span>is_ensemble :</span></dt><dd>True</dd><dt><span>run_datetime :</span></dt><dd>2023-12-07T00:00:00.000000</dd><dt><span>category :</span></dt><dd>Hurricane Cat. 1</dd></dl></div></li></ul></div></div>




```python
tc_fcast.plot()
plt.show()
```



![png](wind_to_grid_ECMWF_files/wind_to_grid_ECMWF_4_0.png)




```python
cent = Centroids.from_geodataframe(grids)
tc = TropCyclone.from_tracks(
    tc_fcast, centroids=cent, store_windfields=True, intensity_thres=0
)
```

    2023-12-07 16:25:05,073 - climada.hazard.centroids.centr - WARNING - Centroids.from_geodataframe has been deprecated and will be removed in a future version. Use ther default constructor instead.


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
      <th>0</th>
      <td>03P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-12-07 00:00:00</td>
      <td>2023-12-10 12:00:00</td>
      <td>False</td>
      <td>355</td>
      <td>0.0</td>
      <td>2262.529064</td>
      <td>POINT (176.85000 -17.15000)</td>
    </tr>
    <tr>
      <th>75647</th>
      <td>70A</td>
      <td>179</td>
      <td>[A]</td>
      <td>2023-12-07 00:00:00</td>
      <td>2023-12-10 00:00:00</td>
      <td>False</td>
      <td>3079</td>
      <td>0.0</td>
      <td>12296.087551</td>
      <td>POINT (179.55000 -16.85000)</td>
    </tr>
    <tr>
      <th>75646</th>
      <td>70A</td>
      <td>179</td>
      <td>[A]</td>
      <td>2023-12-07 00:00:00</td>
      <td>2023-12-10 00:00:00</td>
      <td>False</td>
      <td>3078</td>
      <td>0.0</td>
      <td>12293.597795</td>
      <td>POINT (179.55000 -16.75000)</td>
    </tr>
    <tr>
      <th>75645</th>
      <td>70A</td>
      <td>179</td>
      <td>[A]</td>
      <td>2023-12-07 00:00:00</td>
      <td>2023-12-10 00:00:00</td>
      <td>False</td>
      <td>3077</td>
      <td>0.0</td>
      <td>12291.117576</td>
      <td>POINT (179.55000 -16.65000)</td>
    </tr>
    <tr>
      <th>75644</th>
      <td>70A</td>
      <td>179</td>
      <td>[A]</td>
      <td>2023-12-07 00:00:00</td>
      <td>2023-12-10 00:00:00</td>
      <td>False</td>
      <td>3076</td>
      <td>0.0</td>
      <td>12288.646902</td>
      <td>POINT (179.55000 -16.55000)</td>
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
      <th>118856</th>
      <td>70P</td>
      <td>282</td>
      <td>[P]</td>
      <td>2023-12-09 12:00:00</td>
      <td>2023-12-10 12:00:00</td>
      <td>False</td>
      <td>1772</td>
      <td>0.0</td>
      <td>1829.493520</td>
      <td>POINT (178.25000 -17.45000)</td>
    </tr>
    <tr>
      <th>118855</th>
      <td>70P</td>
      <td>282</td>
      <td>[P]</td>
      <td>2023-12-09 12:00:00</td>
      <td>2023-12-10 12:00:00</td>
      <td>False</td>
      <td>1771</td>
      <td>0.0</td>
      <td>1818.395120</td>
      <td>POINT (178.25000 -17.35000)</td>
    </tr>
    <tr>
      <th>118854</th>
      <td>70P</td>
      <td>282</td>
      <td>[P]</td>
      <td>2023-12-09 12:00:00</td>
      <td>2023-12-10 12:00:00</td>
      <td>False</td>
      <td>1770</td>
      <td>0.0</td>
      <td>1807.296862</td>
      <td>POINT (178.25000 -17.25000)</td>
    </tr>
    <tr>
      <th>118878</th>
      <td>70P</td>
      <td>282</td>
      <td>[P]</td>
      <td>2023-12-09 12:00:00</td>
      <td>2023-12-10 12:00:00</td>
      <td>False</td>
      <td>1889</td>
      <td>0.0</td>
      <td>2006.654849</td>
      <td>POINT (178.35000 -19.05000)</td>
    </tr>
    <tr>
      <th>119141</th>
      <td>70P</td>
      <td>282</td>
      <td>[P]</td>
      <td>2023-12-09 12:00:00</td>
      <td>2023-12-10 12:00:00</td>
      <td>False</td>
      <td>5223</td>
      <td>0.0</td>
      <td>2037.844766</td>
      <td>POINT (181.65000 -19.15000)</td>
    </tr>
  </tbody>
</table>
<p>121669 rows × 10 columns</p>
</div>




```python
df_windfield[df_windfield.in_fiji == True]
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
  </tbody>
</table>
</div>




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
      <th>0</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>355</td>
      <td>29.265668</td>
      <td>109.989000</td>
      <td>POINT (176.85000 -17.15000)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>409</td>
      <td>0.000000</td>
      <td>413.233268</td>
      <td>POINT (176.95000 -12.45000)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>456</td>
      <td>27.815857</td>
      <td>118.877000</td>
      <td>POINT (176.95000 -17.15000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>510</td>
      <td>0.000000</td>
      <td>421.768207</td>
      <td>POINT (177.05000 -12.45000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>511</td>
      <td>0.000000</td>
      <td>414.655758</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>416</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>5122</td>
      <td>0.000000</td>
      <td>475.618487</td>
      <td>POINT (181.55000 -19.15000)</td>
    </tr>
    <tr>
      <th>417</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>5123</td>
      <td>0.000000</td>
      <td>468.237405</td>
      <td>POINT (181.55000 -19.25000)</td>
    </tr>
    <tr>
      <th>418</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>5221</td>
      <td>0.000000</td>
      <td>498.684369</td>
      <td>POINT (181.65000 -18.95000)</td>
    </tr>
    <tr>
      <th>419</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>5223</td>
      <td>0.000000</td>
      <td>483.922205</td>
      <td>POINT (181.65000 -19.15000)</td>
    </tr>
    <tr>
      <th>420</th>
      <td>02P</td>
      <td>0</td>
      <td>[P]</td>
      <td>2023-11-13</td>
      <td>2023-11-16 18:00:00</td>
      <td>True</td>
      <td>5331</td>
      <td>0.000000</td>
      <td>440.558347</td>
      <td>POINT (181.75000 -19.85000)</td>
    </tr>
  </tbody>
</table>
<p>421 rows × 10 columns</p>
</div>




```python
event1 = fiji_forecast[fiji_forecast.unique_id == events_fiji[0]]
gdf_aux = gpd.GeoDataFrame(event1)

# Plot
fig, ax = plt.subplots(1,1)
gdf_aux.plot(ax=ax, column='wind_speed', cmap='coolwarm', markersize=20, legend=True, label= 'Wind Speed [m/s]')
```




    <Axes: >





![png](wind_to_grid_ECMWF_files/wind_to_grid_ECMWF_13_1.png)




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
      <td>355</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29.265668</td>
      <td>109.989000</td>
      <td>02P</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>355</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26.595394</td>
      <td>179.177815</td>
      <td>02P</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>355</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.802792</td>
      <td>173.938078</td>
      <td>02P</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>355</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28.472892</td>
      <td>132.802380</td>
      <td>02P</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>355</td>
      <td>86.0</td>
      <td>0</td>
      <td>1</td>
      <td>224.976542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.525152</td>
      <td>194.730675</td>
      <td>02P</td>
      <td>4</td>
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
      <th>20624</th>
      <td>5331</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>489.401490</td>
      <td>02P</td>
      <td>47</td>
    </tr>
    <tr>
      <th>20625</th>
      <td>5331</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>578.200521</td>
      <td>02P</td>
      <td>48</td>
    </tr>
    <tr>
      <th>20626</th>
      <td>5331</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>572.191965</td>
      <td>02P</td>
      <td>49</td>
    </tr>
    <tr>
      <th>20627</th>
      <td>5331</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>460.294688</td>
      <td>02P</td>
      <td>50</td>
    </tr>
    <tr>
      <th>20628</th>
      <td>5331</td>
      <td>71.4</td>
      <td>0</td>
      <td>1</td>
      <td>17774.293574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>476.968657</td>
      <td>02P</td>
      <td>51</td>
    </tr>
  </tbody>
</table>
<p>20629 rows × 11 columns</p>
</div>
