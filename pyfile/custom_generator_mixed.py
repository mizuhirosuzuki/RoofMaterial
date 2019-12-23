
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

# Generator to load datasets
def custom_generator(images_list,
        dataframe,
        lb_label,
        continuous_var_list,
        categorical_var_list,
        minmaxscaler,
        lb_list,
        batch_size,
        mode,
        IMG_HEIGHT,
        IMG_WIDTH,
        train_data_mean,
        augment = None):
    i = 0
    # if not evaluation generator, shuffle the image 
    if mode == 'train':
        random.shuffle(images_list)
    while True:        
        images = []
        csv_continuous_features = []
        csv_categorical_features = []
        labels = []
        
        while len(images) < batch_size:
            if i == len(images_list):
                i = 0
                # if evaluation generator, break the loop when the last image is retrieved
                if mode == 'eval':
                    break
                random.shuffle(images_list)                
                  
            # Read image from list and convert to array
            image_path = images_list[i]
            image_name = os.path.basename(image_path).replace('.jpg', '')
            image = krs_image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            image = np.asarray(image)
            images.append(image)

            # Read data from csv using the name of current image
            csv_row = dataframe[dataframe.id == image_name]
            
            # extract continuous features
            csv_continuous_features.append(np.array(csv_row[continuous_var_list])[0])
            
            # extract categorical features
            csv_categorical_features.append(np.array(csv_row[categorical_var_list])[0])

            # # just to check if data are correctly retrieved...
            # print(image_name)
            
            if mode == 'train':
                label = np.array(csv_row['label'])[0]
                labels.append(label)

            i += 1

        images = np.array(images)
        if augment != None:
            if mode == 'train':
                (images, labels) = next(augment.flow(images, labels, batch_size = batch_size, shuffle = False))
            elif mode == 'eval':
                images = next(augment.flow(images, batch_size = batch_size, shuffle = False))
        elif augment == None:
            datagen_rescale = ImageDataGenerator(rescale = 1./255, featurewise_center = True)
            datagen_rescale.mean = train_data_mean
            if mode == 'train':
                (images, labels) = next(datagen_rescale.flow(images, labels, batch_size = batch_size, shuffle = False))
            elif mode == 'eval':
                images = next(datagen_rescale.flow(images, batch_size = batch_size, shuffle = False))
            
        # rescale continuous features
        csv_continuous_features = minmaxscaler.transform(np.array(csv_continuous_features))
        
        # convert categorical features into one-hot encoding
        csv_categorical_features_temp = lb_list[0].transform(np.array(csv_categorical_features)[:, 0])
        
        if len(categorical_var_list) > 0:
            for j in range(1, len(categorical_var_list)):
                np.hstack(csv_categorical_features_temp, lb_list[j].transform(csv_categorical_features_temp[:, j]))
        csv_categorical_features = csv_categorical_features_temp
        
        csv = np.hstack([csv_continuous_features, csv_categorical_features])
        
        if mode == 'train':
            labels = lb_label.transform(labels)
            yield [np.array(images), csv], labels
        elif mode == 'eval':
            yield [np.array(images), csv]

