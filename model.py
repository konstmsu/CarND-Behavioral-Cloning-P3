#! /usr/bin/env python
"""CarND project 3"""

import os
import itertools
import numpy as np
import cv2
from random import random as rnd
from matplotlib import pyplot as plt
from PIL import Image

# We'll shrink images from 320x160 to 160x80 to save memory
IMAGE_SIZE = (80, 160)

"""Data is loaded from all `folders` and concatenated"""
def load_csv(folders):
    import csv
    train_paths = []
    train_angles = []

    for folder in sorted(f for f in folders):
        with open(os.path.join(folder, "driving_log.csv")) as csv_file:
            paths = []
            angles = []
            row_count = 0
            for row in csv.reader(csv_file):
                if row[0] == 'center':
                    continue
                
                row_count += 1
                angle = float(row[3])
                img_file_name = row[0].replace('\\', '/').split('/')[-1]
                paths.append(os.path.join(folder, "IMG", img_file_name))
                angles.append(angle)

        interesting_paths = []
        interesting_angles = []

        look_back = 2
        look_forward = 5 
        for i in range(look_back, len(paths) - look_back - look_forward):
            # Poor man's rolling window sum
            if sum(map(abs, angles[i - look_back : i + look_forward])) > 0.1:
                interesting_paths.append(paths[i])
                interesting_angles.append(angles[i])

        print("Using %s images out of %s from '%s'" % (len(interesting_paths), len(paths), folder))

        train_paths.extend(interesting_paths)
        train_angles.extend(interesting_angles)


    print("%s images in total" % len(train_paths))

    return (train_paths, train_angles)


"""Build and compile model"""
def get_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D

    model = Sequential()

    model.add(Convolution2D(5, 5, 5, subsample=(2, 2), activation='relu', input_shape=(*IMAGE_SIZE, 3)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(7, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(8, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(12, 3, 5, subsample=(1, 1), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(12, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(12, 3, 3, subsample=(1, 1), activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))

    print(model.summary())

    model.compile(optimizer="adam", loss="mse")

    return model


"""Convert back to RBG and show image"""
def show_image(img):
    plt.figure()
    plt.imshow(cv2.cvtColor(np.uint8((img + 0.5) * 255.0), cv2.COLOR_HSV2RGB), interpolation='none')

"""Resize image, convert to HSV, normalize to -0.5..0.5"""
def normalize_image(img):
    img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BICUBIC)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img / 255.0 - 0.5


# Cache to avoid loading same image multiple times
IMAGE_CACHE = {}


"""Loads, normalizes, caches and returns image"""
def get_image(path):
    global IMAGE_CACHE

    if path not in IMAGE_CACHE:
        IMAGE_CACHE[path] = normalize_image(Image.open(path))

    return IMAGE_CACHE[path]


"""Indefinitely cycles input and batch it into numpy arrays"""
def batch(images_and_angles, batch_size):
    images = []
    angles = []
    for image, angle in itertools.cycle(images_and_angles):
        images.append(image)
        angles.append(angle)
        if len(images) >= batch_size:
            yield np.asarray(images), np.asarray(angles)
            images = []
            angles = []


"""Returns image itself and its transformations"""
def transform(image, angle): 
    yield image, angle
    yield np.fliplr(image), -angle

    if abs(angle) < 0.3 and rnd() > 0.7:
        offset = image.shape[1] // 5
        correction = 0.2
        yield shift_right(image, offset), angle - correction
        yield shift_right(image, -offset), angle + correction


"""Shifts image right by offset (left if `offset` is negative)"""
def shift_right(img, offset):
    if offset > 0:
        return np.lib.pad(img, ((0, 0), (offset, 0), (0, 0)), 'constant')[:, :-offset]
    else:
        return np.lib.pad(img, ((0, 0), (0, -offset), (0, 0)), 'constant')[:, -offset:]


"""Train the model"""
def train(model, paths, angles):
    import sklearn.model_selection
    # Shuffle data and split into training and validation set (90/10% by default)
    (train_paths, val_paths, train_angles, val_angles) = sklearn.model_selection.train_test_split(*sklearn.utils.shuffle(paths, angles))

    # Prepare generators
    train_images = (get_image(p) for p in train_paths)
    transformed = (t for ia in zip(train_images, train_angles) for t in transform(*ia))
    validation = zip((get_image(p) for p in val_paths), val_angles)

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint("carnd-p3.h5", monitor='val_loss', verbose=2,
        save_best_only=True, save_weights_only=True) 

    # Train the model
    model.fit_generator(batch(transformed, 40),
                        samples_per_epoch=12000,
                        nb_epoch=20,
                        validation_data=batch(validation, batch_size=320),
                        nb_val_samples=len(val_angles),
                        callbacks=[checkpoint])


def main():
    #img = get_image(r"C:\projects\CarND\Project3\recording\track1_round1\IMG\center_2017_01_28_03_59_58_715.jpg")
    #show_image(shift_right(img, 30))
    #show_image(shift_right(img, -30))
    #return
    import keras.utils

    model = get_model()

    try:
        from keras.utils.visualize_util import plot
        plot(model, show_shapes=True)
    except ImportError as ex:
        print("Could not save model schema: %s" % ex)

    print("Saving model...")
    with open('carnd-p3.json', 'w') as model_file:
        model_file.write(model.to_json())

    import glob
    recording_folders = [f for f in glob.glob('../recording/*') if os.path.isdir(f)]
    (all_paths, all_angles) = load_csv(recording_folders)

    try:
        plt.hist(all_angles, 100)
    except Exception as ex:
        print("Could not show histogram: %s" % ex)

    try:
        model.load_weights('carnd-p3.h5')
        print("Loaded weights")
    except:
        print("Failed to load weights, training from scratch")

    train(model, all_paths, all_angles)

    #print("Saving weights...")
    #model.save_weights('carnd-p3.h5')


if __name__ == '__main__':
    main()
