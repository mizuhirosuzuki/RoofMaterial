# "Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery" (20th / 1425 participants)

This repository stores files used for a competition [Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery](https://www.drivendata.org/competitions/58/disaster-response-roof-type/), which resulted in 20th / 1425 competitors.
In this competition, we are given images of each city in Colombia, Guatemala, and St. Lucia, coordinates of buildings, and for training buildings, their roof materials (cement, healthy metal, incomplete, irregular metal, or other).
Using the information, we predict the roof materials of test buildings.

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

To increase generalizability, I take the following strategies:
- training data augmentation;
- early stopping to avoid overfitting, with 25\% of training observations used for validation.

Also, instead of tuning learning rates, which are presumably one of the most important parameters, I use the [cyclical learning rates](https://www.pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/).
For this, I used automaticaly found [optimal learning rates](https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/).

I separately train models using four different types of images created above.
The jupyter notebooks for them are: 
[whole image with RGB channels](ipynb_folder/mixed_inputs_cnn_resize.ipynb), 
[masked image with RGB channels](ipynb_folder/mixed_inputs_cnn_resize_hsv.ipynb), 
[whole image with HSV channels](ipynb_folder/mixed_inputs_cnn_mask.ipynb), and
[masked image with HSV channels](ipynb_folder/mixed_inputs_cnn_mask_hsv.ipynb).

### 3. Create predicted probabilities 

Based on the trained models, I create predicted probabilities of roof materials for each building.
For better predictions, I use random [test-time data augmentation](https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/), predict probabilities 20 times, and take the averages.
As I train four models and there are five classes (cement, healthy metal, incomplete, irregular metal, and other), I obtain 20 variables in the end, both for training and test data.
This steps is included in the jupyter notebooks in the Step 2.

### 4. Train XGBoost model on the predicted probabilities and numerical variables

To ensemble four models, I use [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html).
For this, I use the 20 variables in the previous step and all the numerical variables used in the Step 2.
The end results are the final predicted probabilities.

