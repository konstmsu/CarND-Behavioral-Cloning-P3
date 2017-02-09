#! /usr/bin/env python
"""CarND project 3"""

import os
import csv
import glob
import itertools
import sklearn.model_selection
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
import cv2
from matplotlib import pyplot as plt
from random import random as rnd
from PIL import Image


image_size = (40, 80)


def load_csv(folders):
    paths = []
    angles = []

    for folder in sorted(f for f in folders):
        with open(os.path.join(folder, "driving_log.csv")) as csv_file:
            row_counter = 0
            accepted_samples = 0
            for row in csv.reader(csv_file):
                if row[0] == 'center':
                    continue

                row_counter += 1

                angle = float(row[3])

                if abs(angle) < 0.01 and rnd() < 0.99:
                    continue
                
                if abs(angle) > 0.99 and rnd() < 0.7:
                    continue
                
                accepted_samples += 1
                img_file_name = row[0].replace('\\', '/').split('/')[-1]
                paths.append(os.path.join(folder, "IMG", img_file_name))
                angles.append(angle)

        print("%s images in '%s'" % (accepted_samples, folder))

    print("%s images in total" % len(paths))

    return (paths, angles) 


def get_model():
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
    from keras.layers.convolutional import Convolution2D, MaxPooling2D

    model = Sequential()

    model.add(Convolution2D(7, 5, 7, subsample=(1, 2), input_shape=(*image_size, 3), activation='relu'))
    model.add(Convolution2D(11, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(13, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(15, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(17, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001), loss="mse")

    return model


def show_image(img):
    plt.figure()
    plt.imshow(img)


def normalize_image(img):
    img = img.resize((image_size[1], image_size[0]), Image.ANTIALIAS)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img / 255.0 - 0.5


image_cache = {}


def get_image(path):
    global image_cache

    if path not in image_cache:
        image_cache[path] = normalize_image(load_img(path))
        
    return image_cache[path]


def batch(images, angles):
    image_batch = []
    angle_batch = []
    for (image, angle) in zip(images, angles):
        image_batch.append(image)
        angle_batch.append(angle)
        if len(image_batch) == 50:
            yield np.asarray(image_batch), np.asarray(angle_batch)
            image_batch = []
            angle_batch = []


def batch1(images_and_angles):
    image_batch = []
    angle_batch = []
    for (image, angle) in images_and_angles:
        image_batch.append(image)
        angle_batch.append(angle)
        if len(image_batch) == 50:
            yield np.asarray(image_batch), np.asarray(angle_batch)
            image_batch = []
            angle_batch = []


def transform(image, angle):
    yield image, angle
    yield np.fliplr(image), -angle


def train(model, paths, angles):
    (train_paths, val_paths, train_angles, val_angles) = sklearn.model_selection.train_test_split(*sklearn.utils.shuffle(paths, angles))

    #try:
    #    from matplotlib import pyplot as plt
    #    plt.figure()
    #    plt.imshow(load_image(train_paths[0]) + 0.5)
    #except ImportError as e:
    #    print("Not going to save mode png: %s" % e)

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    train_images = (get_image(p) for p in train_paths)
    transformed = itertools.chain.from_iterable(transform(i, a) for (i, a) in itertools.cycle(zip(train_images, train_angles)))
    model.fit_generator(batch1(transformed),
                        samples_per_epoch=2000,
                        nb_epoch=10,
                        validation_data=batch(itertools.cycle(get_image(p) for p in val_paths), itertools.cycle(val_angles)),
                        nb_val_samples=len(val_angles))


def main():
    img = get_image(r'..\recording\track1_round1\IMG\center_2017_01_28_04_00_11_255.jpg')
    
    
    model = get_model()

    try:
        from keras.utils.visualize_util import plot
        plot(model, show_shapes=True)
    except ImportError as ex:
        print("Could not save model schema: %s" % ex)

    print("Saving model...")
    with open('carnd-p3.json', 'w') as model_file:
        model_file.write(model.to_json())

    recording_folders = [f for f in glob.glob('../recording/*') if os.path.isdir(f)]
    (all_paths, all_angles) = load_csv(recording_folders)

    plt.hist(all_angles, 100)

    try:
        model.load_weights('carnd-p3.h5')
        print("Loaded weights")
    except:
        print("Failed to load weights, training from scratch")

    train(model, all_paths, all_angles)
    model.save_weights('carnd-p3.h5')


if __name__ == '__main__':
    main()
