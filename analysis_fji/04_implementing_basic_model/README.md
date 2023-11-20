# Implementing real-time forecast and making predictions
---

## Report: .py codes

### Code: *creating_grid.py*

Here we automatise the creation of grid cells on Fiji. We don't use it for predictions, but it is there to, in the future, create grid cells for every location of the world.

### Code: *utils.py*

Here we create a repository for calling functions to load datasets. For example, we use it to load the Philippines Dataset, the Fiji dataset, and the combined dataset.

### Code: *input_dataset.py*

Here we create 3 important functions:
-  create_windfield_dataset(thres)
-  create_rainfall_dataset()
-  create_input_dataset()


```create_windfield_dataset()``` loads ECMWF real-time forecasts, and merge it with the grid cells dataset that we have defined in *02_new_model_input*. Also if the forecast event took place in Fiji, a new feature *in_fiji* is set True. To define the Fiji region, we create a polygon (a square) with a custom size and if any point of the forecast track falls into this square, we automatically set that forecast as Fiji-related. The user can obviously play with the dimension of this square.

Also, the user can set a threshold in hours (thres): this threshold let the user to just consider datapoints up to a certain time in hours starting from the collection datetime of the data. The default value is 120h. So every wind forecast that the ECMWF provides is cut to just consider wind paths that just contemplates datapoints values up to 120h. Also, a wind path is considered just if it has at least 4 datapoints.


```create_windfield_dataset()``` uses NOMADS real-time rainfall data and compute the max rainfall accumulated in 6h and 24h periods for every grid cell. The output is a .csv of rainfall data for each wind event event that takes place in Fiji.


```create_input_dataset()``` takes the output of ```create_windfield_dataset()``` and ```create_windfield_dataset()`` and just consider the forecasts that took place in Fiji. Also, it merges this windspeed data with all the other stationary features defined in *03_new_model_training/01_collage_data.ipynb*.

### Code: *predict_damage.py*

Here we define the function ```apply_model()```. This function considers as input a list of datasets (in our case: windfield data + stationary features for various forecasts) and applies a weighted XGBoost model with training data from Fiji + Philippines datasets. This function predicts damage for every input dataset at grid level and it then transforms the damage to municipality level.

**Output**: a list of datasets of predicted damage by province.

**Note**: The damage in some grid cells are <0. This is notorious specially if we're not using real typhoons windspeed data. If we use real typhoon windspeed data, in the regions affected, the damage is always >0. I don't want to put any constrains on this, but if you want, you can set all values of predicted damage < 0 to 0 just by uncommentening line 96 in the *predict_damage.py* script.

---

## Report: .ipynb codes

- In *wind_to_grid_experiment.ipynb* we play with getting windspeed data in every position based on forecasts dataset not provided by the ECMWF. Since it's a failure, I'm not consider any of what it's written in this notebook. We keep it for completeness.

- In *wind_to_grid_ECMWF.ipynb* we consider ECMWF data and we define the approach that we are later going to define in the ```create_windfield_dataset()``` function in the *input_dataset.py* file.

- In *rainfall_data.ipynb* is everything regarding the rainfall forecast data: the downloading and the processing of the data.

- In *apply_model.ipynb* we have 2 sections: Manual process and Automated Process. The Manual section is left for completeness but I encourage anyone to use the Automated Process section. In this section we use all the functions that we have already defined to predict damage at municipality-level based on ECMWF forecasts in Fiji.
