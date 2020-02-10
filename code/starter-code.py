import os
import sys
import re

import cv2

import numpy as np
import pandas as pd


def load_images(directory):
    '''
    input:
        directory: string, absolute path of the image folder. e.g. "/Users/amandeep/images/"
        
    output:
        images: 4-dimnesional numpy array of shape (NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    '''
    
    image_numbers = []
    image_name_pattern = r'(^\d+)\.tif$'
    for file in os.listdir(directory):
        if re.search(image_name_pattern, file):
            image_name = re.search(image_name_pattern, file)
            image_number = int(image_name.group(1))
            image_numbers.append(int(image_number))
            pass

    image_numbers.sort()
    image_filenames = map(lambda image_number: str(image_number) + '.tif', image_numbers)
    image_filenames = [image_filename for image_filename in image_filenames]
    
    images = []
    for image_filename in image_filenames:
        image = cv2.imread(directory + image_filename, cv2.IMREAD_UNCHANGED)
        images.append(image)
        pass
    
    # convert list to array
    images = np.array(images)
    
    return images


IMAGE_HEIGHT = 101
IMAGE_WIDTH = 101
IMAGE_CHANNELS = 3

TRAINING_FOLDER = "/Users/amandeeprathee/kaggle-solar-pv/data/training/"
TESTING_FOLDER = "/Users/amandeeprathee/kaggle-solar-pv/data/testing/"

X_train = load_images(TRAINING_FOLDER)
X_train.shape

X_test = load_images(TESTING_FOLDER)
X_test.shape