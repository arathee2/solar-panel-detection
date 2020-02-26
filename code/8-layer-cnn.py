# Input data files are available in the "../input/" directory.

# import libraries
import os
import sys
import re
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings('ignore')
    
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import print_summary
from keras.metrics import categorical_accuracy
from keras import losses
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import Dense
from keras.layers import GlobalMaxPooling2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Add
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

import tensorflow as tf

import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



# define helper functions
def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory

    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory

    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids = labels_pd.id.values
    data = []
    for identifier in ids:
        fname = dir_data + identifier.astype(str) + '.tif'
        image = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data)  # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids


def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on')
    plt.axis('square')
    plt.legend()
    plt.tight_layout()
    pass

# define constants
IMAGE_HEIGHT = 101
IMAGE_WIDTH = 101
IMAGE_CHANNELS = 3
NUM_CLASSES = 2

DIR_TRAIN_IMAGES = "../input/ids705sp2020/training/"
DIR_TEST_IMAGES = "../input/ids705sp2020/testing/"
DIR_TRAIN_LABELS = "../input/ids705sp2020/labels_training.csv"
DIR_TEST_IDS = "../input/ids705sp2020/sample_submission.csv"

# load training data
X, y = load_data(DIR_TRAIN_IMAGES, DIR_TRAIN_LABELS)
print('X shape:\n', X.shape)

# y class balance
print('Distribution of y', np.bincount(y))

# load test data
X_test, test_ids = load_data(DIR_TEST_IMAGES, DIR_TEST_IDS, training=False)
print('X_test shape:\n', X_test.shape)

# normalize pixel values
X = X/255.0
X_test = X_test/255.0

print(np.sum(X[0, :, :, :]))
print(np.sum(X_test[0, :, :, :]))


# create train and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=6)


# data shape
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_validation shape:', X_validation.shape)
print('y_validation shape:', y_validation.shape)


# MODELING
# validation auc = 0.9938; epoch = 50
def build_model(lr=0.05, dropout=0.05):

    # model 
    model = Sequential()
    model.add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (101, 101, 3)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(2))
    
    
    
    model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(2))
    
    
    
    model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    
    model.add(GlobalMaxPooling2D())
    
    
    
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # compile 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



# ------- Validate Model ------------

# create model
model = build_model()

# print model architecture
print_summary(model, line_length=None, positions=None, print_fn=None)

# fit
training = model.fit(x=X_train,
                     y=y_train,
                     class_weight={0: 505/1500, 1: 995/1500},
                     batch_size=32,
                     epochs=40,
                     validation_data=(X_validation, y_validation),
                     shuffle=True,
                     verbose=1)

training.history['accuracy']
training.history['val_accuracy']

# validation set performance
y_validation_pred = model.predict_proba(X_validation)

# AUC
metrics.roc_auc_score(y_validation, y_validation_pred)




# ------ Train on full data -------

model = build_model()

# fit
training = model.fit(x=X,
                     y=y,
                     class_weight={0: 505/1500, 1: 995/1500},
                     batch_size=32,
                     epochs=25,
                     shuffle=True,
                     verbose=1)

training.history['accuracy']

y_test_hat = model.predict(X_test)

print(metrics.roc_auc_score(y_train, model.predict(X_train)))
print(metrics.roc_auc_score(y_validation, model.predict(X_validation)))

submission = pd.DataFrame({"id": test_ids, "score": y_test_hat.ravel()})
submission.to_csv("submission.csv", index = False)
submission.head(20)
