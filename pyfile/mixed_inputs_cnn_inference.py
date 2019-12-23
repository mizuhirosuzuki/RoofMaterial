
# import packages
import tensorflow as tf
from keras import datasets, layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as krs_image
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.utils import class_weight
import random
from glob import glob
import tempfile

# import self-defined packages
from CLR import *
from custom_generator_mixed import *

def mixed_inputs_cnn_inference(mode, 
        channel = 'RGB',
        batch_size, 
        IMG_HEIGHT, 
        IMG_WIDTH, 
        continuous_var_list,
        categorical_var_list):

    # set parameters
    path = '../rawdata/'
    image_folder_name = 'image_' + mode
    train_dir = os.path.join(path, image_folder_name, 'train/')
    test_dir = os.path.join(path, image_folder_name, 'test/')

    # load CSV files
    train_data_csv = pd.read_csv(os.path.join(train_dir, 'train_data_with_dist.csv'))
    test_data_csv = pd.read_csv(os.path.join(test_dir, 'test_data_with_dist.csv'))

    # add image file names to the datasets
    train_data_csv['filename'] = train_data_csv.id + '.jpg'
    test_data_csv['filename'] = test_data_csv.id + '.jpg'

    # split training and validation data
    train_data, valid_data, train_label, valid_label = train_test_split(train_data_csv, 
                                                                        train_data_csv.label, 
                                                                        test_size = 0.25, 
                                                                        random_state = 123,
                                                                        stratify = train_data_csv.label)

    total_train = len(train_data)
    total_valid = len(valid_data)
    total_test = len(test_data_csv)

    if channel == 'RGB':
        train_data_mean = [train_data.R_mean.mean() / 255, train_data.G_mean.mean() / 255, train_data.B_mean.mean() / 255]
    elif channel == 'HSV':
        train_data_mean = [train_data.H_mean.mean() / 255, train_data.S_mean.mean() / 255, train_data.V_mean.mean() / 255]

    # Define a rescaler so that continuous variables are within the range of [0, 1]
    minmaxscaler = MinMaxScaler()
    minmaxscaler.fit(train_data_csv[continuous_var_list])

    # Define one-hot encoders for categorical variables
    lb0 = LabelBinarizer()
    lb0.fit(train_data_csv[categorical_var_list[0]])
    lb_list = [lb0]

    # Define a one-hot encoder for label
    lb_label = LabelBinarizer()
    lb_label.fit(train_data_csv.label)

    # Create an empty data generator, which will be used in training data generator
    datagen = ImageDataGenerator(rescale = 1./255,
                                horizontal_flip = True,
                                vertical_flip = True,
                                rotation_range = 45,
                                featurewise_center = True)
    datagen.mean = train_data_mean

    # (sub)generators in the main generators below
    datagen_test = ImageDataGenerator(rescale = 1./255,
                                      shear_range=0.1,
                                      zoom_range=0.1,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      rotation_range=10.,
                                      width_shift_range = 0.1,
                                      height_shift_range = 0.1,
                                      featurewise_center = True)
    datagen_test.mean = train_data_mean

    # Read the image list and csv
    train_image_file_list = glob(os.path.join(train_dir, '*.jpg'))
    test_image_file_list = glob(os.path.join(test_dir, '*.jpg'))

    valid_image_file_index = [os.path.basename(train_image_file_list[i]).replace('.jpg','') in list(valid_data.id) for i in range(len(train_image_file_list))]
    train_image_file_index = [os.path.basename(train_image_file_list[i]).replace('.jpg','') in list(train_data.id) for i in range(len(train_image_file_list))]
    valid_image_file_list = list(np.array(train_image_file_list)[valid_image_file_index])
    train_image_file_list = list(np.array(train_image_file_list)[train_image_file_index])

    # generators
    train_data_generator = custom_generator(train_image_file_list, train_data, 
                                             lb_label, continuous_var_list, categorical_var_list, 
                                             minmaxscaler, lb_list, 
                                             batch_size, mode = 'train', 
                                             IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH,
                                             train_data_mean = train_data_mean,
                                             augment = datagen)

    validation_data_generator = custom_generator(valid_image_file_list, valid_data, 
                                                 lb_label, continuous_var_list, categorical_var_list, 
                                                 minmaxscaler, lb_list, 
                                                 batch_size, mode = 'train', 
                                                 IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH,
                                                 train_data_mean = train_data_mean,
                                                 augment = None)

    test_data_generator = custom_generator(test_image_file_list, test_data_csv, 
                                             lb_label, continuous_var_list, categorical_var_list, 
                                             minmaxscaler, lb_list, 
                                             batch_size, mode = 'eval', 
                                             IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH,
                                             train_data_mean = train_data_mean,
                                             augment = datagen_test)

    # number of categorical variables used in the training
    num_categorical_var = 0
    for i in range(len(categorical_var_list)):
      num_categorical_var += len(train_data[categorical_var_list[i]].unique())
    total_num_var = len(continuous_var_list) + num_categorical_var

    # Prediction for test data =======================

    # prediction with test-time augmentation
    tta_steps = 10
    predictions = []

    for i in range(tta_steps):
        print('Iteration', i)
        test_data_generator = custom_generator(test_image_file_list, test_data_csv, 
                                             lb_label, continuous_var_list, categorical_var_list, 
                                             minmaxscaler, lb_list, 
                                             batch_size, mode = 'eval', 
                                             IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH,
                                             train_data_mean = train_data_mean,
                                             augment = datagen_test)

        pred = model.predict_generator(test_data_generator,
                                       steps = (total_test // batch_size) + 1,
                                      verbose = 1)
        predictions.append(preds)

    pred = np.mean(predictions, axis=0)

    # create an output table
    pred_pd = pd.DataFrame({'id': test_data_csv.id})

    test_image_file_list = glob(os.path.join(test_dir, '*.jpg'))
    pred_id_temp = [os.path.basename(test_image_file_list[i]).replace('.jpg', '') for i in range(total_test)]
    pred_pd_temp = pd.DataFrame({'id': pred_id_temp})
    pred_pd_temp['concrete_cement'] = pred[:, 0]
    pred_pd_temp['healthy_metal'] = pred[:, 1]
    pred_pd_temp['incomplete'] = pred[:, 2]
    pred_pd_temp['irregular_metal'] = pred[:, 3]
    pred_pd_temp['other'] = pred[:, 4]

    pred_pd = pd.merge(pred_pd, pred_pd_temp, on = 'id')

    submission_file_name = 'mixed_inputs_' + mode + '.csv'
    pred_pd.to_csv(os.path.join(path, '../submission/', submission_file_name), index = False)


    # Prediction for training data =======================

    # prediction with test-time augmentation
    tta_steps = 10
    predictions = []
    train_image_file_list = glob(os.path.join(train_dir, '*.jpg'))
    total_train = len(train_data_csv)

    for i in range(tta_steps):
        print('Iteration', i)
        train_data_generator = custom_generator(train_image_file_list, train_data_csv, 
                                             lb_label, continuous_var_list, categorical_var_list, 
                                             minmaxscaler, lb_list, 
                                             batch_size, mode = 'eval', 
                                             IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH,
                                             train_data_mean = train_data_mean,
                                             augment = datagen_train)

        pred = model.predict_generator(train_data_generator,
                                       steps = (total_train // batch_size) + 1,
                                      verbose = 1)
        predictions.append(preds)

    pred = np.mean(predictions, axis=0)

    # create an output table
    pred_pd = pd.DataFrame({'id': train_data_csv.id})

    pred_id_temp = [os.path.basename(train_image_file_list[i]).replace('.jpg', '') for i in range(total_train)]
    pred_pd_temp = pd.DataFrame({'id': pred_id_temp})
    pred_pd_temp['concrete_cement'] = pred[:, 0]
    pred_pd_temp['healthy_metal'] = pred[:, 1]
    pred_pd_temp['incomplete'] = pred[:, 2]
    pred_pd_temp['irregular_metal'] = pred[:, 3]
    pred_pd_temp['other'] = pred[:, 4]

    pred_pd = pd.merge(pred_pd, pred_pd_temp, on = 'id')

    feature_file_name = 'mixed_inputs_' + mode + '.csv'
    pred_pd.to_csv(os.path.join(path, '../feature/', feature_file_name), index = False)





