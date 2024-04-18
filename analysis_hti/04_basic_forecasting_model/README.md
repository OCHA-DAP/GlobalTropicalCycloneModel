# Implementing real-time forecast and making predictions
---

## Report: .py codes

### Code: utils.py

Here we create a repository for calling functions to load datasets. For example, we use it to load the Philippines Dataset, the Fiji dataset, etc, and the combined dataset.

### Code: input_dataset.py

Here we create 3 important functions:
-  create_windfield_dataset(thres)
-  create_rainfall_dataset()
-  create_input_dataset()


```create_windfield_dataset(thres=120, deg=3)``` loads ECMWF real-time forecasts, and merge it with the grid cells dataset that we have defined in *02_new_model_input*. Also if the forecast event took place in the Region of interest, a new feature *in_region* is set True. To define the region of interest, we create a polygon (a square) with a custom size and if any point of the forecast track falls into this square, we automatically set that forecast as region-related. The user can obviously play with the dimension of this square.

Parameters:

*thres*: The user can set a threshold in hours: this threshold let the user to just consider datapoints up to a certain time in hours starting from the collection datetime of the data. The default value is 120h. So every wind forecast that the ECMWF provides is cut to just consider wind paths that just contemplates datapoints values up to 120h. Also, a wind path is considered just if it has at least 4 datapoints.

*deg*: Degrees to consider to extend the Polygone (in evey direction). By default, a polygone with coordinates (long_min, long_max, lat_min, lat_max) = (-75, -71, 17, 21) is set. What *deg* does is basically a transformation (-75-deg, -71+deg, 17-deg, 21+deg).


```create_windfield_dataset()``` uses NOMADS real-time rainfall data and compute the max rainfall accumulated in 6h and 24h periods for every grid cell. The output is a .csv of rainfall data for each wind event event that takes place in Fiji.


```create_input_dataset()``` takes the output of ```create_windfield_dataset()``` and ```create_windfield_dataset()`` and just consider the forecasts that took place in Fiji. Also, it merges this windspeed data with all the other stationary features defined in *03_new_model_training/01_collage_data.ipynb*.

### Code: predict_damage.py

Here we define the function ```apply_model()```. This function considers as input a list of datasets (in our case: windfield data + stationary features for various forecasts) and applies a weighted XGBoost model with training data from Fiji + Philippines + Vietnam + Haiti datasets. This function predicts damage for every input dataset at grid level and it then transforms the damage to municipality level.

**Output**: a list of datasets of predicted damage by province.

**Note**: The damage in some grid cells are <0. This is notorious specially if we're not using real typhoons windspeed data. If we use real typhoon windspeed data, in the regions affected, the damage is always >0. I don't want to put any constrains on this, but if you want, you can set all values of predicted damage < 0 to 0 just by uncommentening line 96 in the *predict_damage.py* script.

### Code: run_forecast.py

Automatization code for running everything. Here: *thres*=120h and *deg*=3 degrees. The output is either a list of .csv for each forecast (inside a .zip file) or a .txt file with a 'Trigger not activated' message.

---

## Report: .ipynb codes

- In *apply_model.ipynb* we use all the functions that we have already defined to predict damage at municipality-level based on ECMWF forecasts in Haiti.
