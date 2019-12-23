
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

def mixed_inputs_cnn(mode, 
        channel = 'RGB',
        batch_size, 
        IMG_HEIGHT, 
        IMG_WIDTH, 
        continuous_var_list,
        categorical_var_list):

    # Define models
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

    # Training model with CLR

    # Define a MLP for numerical variables
    model_mlp = Sequential()
    model_mlp.add(Dense(512, input_dim = total_num_var, activation = 'relu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dropout(0.5))
    model_mlp.add(Dense(256, activation = 'relu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dropout(0.5))
    model_mlp.add(Dense(128, activation = 'relu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dropout(0.5))
    model_mlp.add(Dense(64, activation = 'relu'))

    # Define a CNN for image data
    inputs = Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    x = inputs

    filters = (16, 32, 64)
    for (i, f) in enumerate(filters):
      x = Conv2D(f, (3, 3), padding = 'same')(x)
      x = Activation('relu')(x)
      x = BatchNormalization(axis = -1)(x)
      x = MaxPooling2D(pool_size = (2, 2))(x)
      x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis = -1)(x)
    x = Dropout(0.5)(x)

    x = Dense(64)(x)
    x = Activation('relu')(x)

    model_cnn = Model(inputs, x)

    combined_input = concatenate([model_cnn.output, model_mlp.output])

    x = Dense(256, activation = 'relu')(combined_input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation = 'softmax')(x)

    model = Model(inputs = [model_cnn.input, model_mlp.input], outputs = x)

    opt = Adam(lr = 1e-5)

    model.compile(optimizer = opt,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    clr = CyclicLR(
            mode = 'triangular2',
            base_lr = 1e-5,
            max_lr = 1e-2,
            step_size = 4 * (total_train // batch_size))

    # path to the folder where checkpoint model is saved
    model_file_name = 'mixed_inputs_' + mode + '_cp.h5'
    checkpoint_path = os.path.join(path, '../model_cp/', model_file_name)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(
        filepath = checkpoint_path, 
        verbose = 1, 
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True)

    # define an early stopping rule
    early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                            mode = 'min',
                                            verbose = 1,
                                            patience = 10)

    # define class weights to account for imbalance in training data
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(train_label),
                                                     train_label)

    # Training
    epochs = 100

    history = model.fit_generator(
        train_data_generator,
        steps_per_epoch = total_train // batch_size,
        epochs = epochs,
        validation_data = validation_data_generator,
        validation_steps = total_valid // batch_size,
        callbacks = [clr, early_stopping_callback, cp_callback],
        class_weight = class_weights
    )


