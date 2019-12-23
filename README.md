# "Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery" (20th / 1425 participants)

This repository stores files used for a competition [Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery](https://www.drivendata.org/competitions/58/disaster-response-roof-type/), which resulted in 14th / 1421 competitors.

## Overview of the flow

1. From TIFF files, create images of each building in each city and create variables related to each building.
2. Train models, in which both image data and numerical data are used.
3. Create predicted probabilities both for training and test data.
4. Train XGBoost model on the predicted probabilities and numerical variables, and obtain final predicted probabilities.


