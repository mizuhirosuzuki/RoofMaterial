# "Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery" (20th / 1425 participants)

This repository stores files used for a competition [Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery](https://www.drivendata.org/competitions/58/disaster-response-roof-type/), which resulted in 20th / 1425 competitors.

## Overview of the workflow

1. From TIFF files, create images of each building in each city and create variables related to each building.
2. Train models, in which both image data and numerical data are used.
3. Create predicted probabilities both for training and test data.
4. Train XGBoost model on the predicted probabilities and numerical variables, and obtain final predicted probabilities.

## Detailed steps

### 1. Save building images and create numerical variables

From the TIFF files for each city, I create building images.
I use both RGB channels and HSV channels.
The reason that I use [HSV](http://www.roborealm.com/help/HSV_Channel.php) channels is to account for shadows in images:
looking at raw images, I found many buildings with shadows, which I thought might affects training negatively.
Also, for both RGB and HSV channels, I use rectangular images just including the whole building images and images in which non-building parts are masked.
The reason that I use masked images is to avoid confounding effects of non-building objects.
These rectangular shaped images are converted to squared images (224 by 244 pixels) in model trainings below.

For each building image, I also create numerical variables.
This is to capture information that cannot be captured by images only.
These variables include:
means and standard deviations of each R, G, and B channel; means and standard deviations of each H, S, and V channel; area of building; the number of vertices of building; width of building image; height of building image; number of buildings within 10, 20, and 50 meters; number of each type of training buildings (concrete, healthy metal, incomplete, irregular metal, and other) within 10, 20, and 50 meters.
The jupyter notebook used for this data cleaning is [this](ipynb_folder/roof_image_data.ipynb).
### 2. Train models

I train convolutional neural network (CNN) models with numerical variables. 
In particular, models have CNN parts with image data as inputs and standard neural network parts with numerical variables as inputs.
Then, those networks are merged to make a full network.

I separately train models using four different types of images created above.
The jupyter notebooks for them are: 
[whole image with RGB channels](ipynb_folder/mixed_inputs_cnn_resize.ipynb), 
[masked image with RGB channels](ipynb_folder/mixed_inputs_cnn_resize_hsv.ipynb), 
[whole image with HSV channels](ipynb_folder/mixed_inputs_cnn_mask.ipynb), and
[masked image with HSV channels](ipynb_folder/mixed_inputs_cnn_mask_hsv.ipynb).

### 3. Create predicted probabilities 

### 4. Train XGBoost model on the predicted probabilities and numerical variables



