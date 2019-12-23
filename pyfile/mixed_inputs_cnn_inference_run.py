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
from mixed_inputs_cnn import *

# set parameters

batch_size = 128
IMG_HEIGHT = 244
IMG_WIDTH = 244

# Specify which variables are used in NN

continuous_var_list = ['log_building_area', 'building_vertices', 'building_width', 'building_height',
                        'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std',
                        'H_mean', 'S_mean', 'V_mean', 'H_std', 'S_std', 'V_std',
                        'distance_10_all', 'distance_20_all', 'distance_50_all',
                        'concrete_10_train', 'healthy_10_train', 'incomplete_10_train', 'irregular_10_train', 'other_10_train',
                        'concrete_20_train', 'healthy_20_train', 'incomplete_20_train', 'irregular_20_train', 'other_20_train',
                        'concrete_50_train', 'healthy_50_train', 'incomplete_50_train', 'irregular_50_train', 'other_50_train'
                      ]
categorical_var_list = ['place']

mixed_inputs_cnn_inference('resize', 
        batch_size, 
        IMG_HEIGHT, 
        IMG_WIDTH, 
        continuous_var_list,
        categorical_var_list
        )

mixed_inputs_cnn_inference('mask', 
        batch_size, 
        IMG_HEIGHT, 
        IMG_WIDTH, 
        continuous_var_list,
        categorical_var_list
        )

mixed_inputs_cnn_inference('resize_hsv', 
        'HSV',
        batch_size, 
        IMG_HEIGHT, 
        IMG_WIDTH, 
        continuous_var_list,
        categorical_var_list
        )

mixed_inputs_cnn_inference('mask_hsv', 
        'HSV',
        batch_size, 
        IMG_HEIGHT, 
        IMG_WIDTH, 
        continuous_var_list,
        categorical_var_list
        )




