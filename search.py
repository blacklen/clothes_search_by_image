import os
import pickle
import random
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# %matplotlib inline
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import (decode_predictions,
                                                          preprocess_input)
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, LeakyReLU,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from joblib import dump, load


def get_model():
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  # Create the model
    model = keras.models.Sequential()

  # Add the vgg convolutional base model
    model.add(vgg_conv)

  # Add new layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name="feature"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='softmax'))
    
    # Chi nho sua path checkpoint o day nhe
    model.load_weights("check_points/vgg_0.773306.h5")
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("feature").output)
    
    return model, feat_extractor

def get_image_list(images_path, max_num_images):
    """ Take the path where images are, return a list of filenames. 
    Where the images exceeded max_num_images, randomly sample from the folder.
    It returns a sorted list of filenames of length up to max_num_images."""

    image_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    len(image_list)

    # if dataset contain more than max_num_images, randomly sample from the folder
    if max_num_images < len(image_list):
        image_list = [image_list[i] for i in sorted(random.sample(range(len(image_list)), max_num_images))]

    print("%d images to be analyzed" % len(image_list))
    
    return sorted(image_list)

def extract_features(images, model, feat_extractor):
    """ Take in a list of image file names, extract features, return a list of features, of type list of lists. """
    features = []
    for image_path in tqdm(images):
        img, x = get_image(image_path, model);
        feat = feat_extractor.predict(x)[0]
        features.append(feat) # this is of type list of lists
    return features

def get_image(path, model):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255
    return img, x

def get_closest_images(query_feat, database_features, top_n=5):
    distances = [distance.euclidean(query_feat, feat) for feat in database_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:top_n]
    return idx_closest

def get_database_features(path, num_image):
    model, feat_extractor = get_model()
    images = get_image_list(path, num_image)

    # extract features for images
    features = extract_features(images, model, feat_extractor)
    # convert list to np.array
    features = np.array(features)

    return images, features

def query(image_path):
    model, feat_extractor = get_model()
    images = []
    database_features = []

    mapping_labels = {0:'Blouses_Shirt',
                      1: 'Cardigan',
                      2: 'Dress',
                      3: 'Pant',
                      4: 'Short',
                      5: 'Skirt',
                      6: 'Sweater',
                      7: 'T_Shirt'}
    
    img, inp = get_image(image_path, model)
    clf = load("check_points/svm.joblib")
    topic = mapping_labels[model.predict_classes(inp)[0]]
    query_feat = feat_extractor.predict(inp)[0]
    indexPath = 'static/indexing/' + topic + '.csv'
    try:
        with open(indexPath) as f:
            reader = csv.reader(f)

            for row in reader:
                database_features.append([float(x) for x in row[1:]])
                images.append(row[0])
            f.close()
    except:
        output = open(indexPath, "w")
        images, database_features = get_database_features(f"static/dataset/{topic}", 1000)
        for i in range(len(images)):
            features = database_features[i]
            features = [str(f) for f in features]
            output.write("%s,%s\n" % (images[i], ",".join(features)))
    
    result_id = get_closest_images(query_feat, database_features, top_n=6)
    results = []
    
    for i in range(len(result_id)):
        results.append(images[result_id[i]])

    return results
