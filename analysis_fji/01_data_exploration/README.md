# Data exploration

Here we explore the Fiji damage dataset that we have access to.

## Report: codes

### Code 00: the basics

In 00.0_damage_data_complete.ipynb we first complete the damage dataset the we have access to. There were some missing information (year when the typhoon happened and which municipalities it affected if we just have admin 1 data). We also compute the total houses destroyed for each municipality as the sum of *houses destroyed* and *houses with major damage*.

In 00.1_fix_shapefile.ipynb we modify the coordinates of the Fiji shape file that we are going to work with. Since the 180 degree meridian falls into Fiji, it's a little bit tricky to work with the data because all the buildings on the east side have anti-intuitive coordinates (what is suppose to be 182 degrees is -178 degrees, and so on). So we fix changed the geometry of the shapefile.

### Code 01: damage data exploration

In 01_damage_data_exploration.ipynb we plot the maps of the municipalities and divisions affected by each typhoon. In each province we show the total number of houses affected by each typhoon.
