#! /usr/bin/env python
"""CarND project 3"""

import os
import itertools
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from random import random as rnd


image_size = (80, 160)


def load_csv(folders):
    import csv
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

                if abs(angle) < 0.01 and rnd() < 0.97:
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

    model.add(Convolution2D(7, 5, 5, subsample=(1, 2), activation='relu', input_shape=(*image_size, 3)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(9, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(11, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(13, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(15, 3, 3, subsample=(1, 1), activation='relu'))

    model.add(Flatten()) 
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1))

    print(model.summary())

    model.compile(optimizer=Adam(lr=0.0001), loss="mse")

    return model


def show_image(img):
    plt.figure()
    plt.imshow(cv2.cvtColor(np.uint8((img + 0.5) * 255.0), cv2.COLOR_HSV2RGB), interpolation='none')


def normalize_image(img):
    img = img.resize((image_size[1], image_size[0]), Image.BICUBIC)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img / 255.0 - 0.5


IMAGE_CACHE = {}


def get_image(path):
    global IMAGE_CACHE

    if path not in IMAGE_CACHE:
        IMAGE_CACHE[path] = normalize_image(Image.open(path))

    return IMAGE_CACHE[path]


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


def transform(image, angle): 
    yield image, angle
    yield np.fliplr(image), -angle
    if abs(angle) < 0.4 and rnd() < 0.3:
        offset = image.shape[1] // 5
        correction = 0.15
        yield shift_right(image, offset), angle - correction
        yield shift_right(image, -offset), angle + correction


def shift_right(img, offset):
    if offset > 0:
        return np.lib.pad(img, ((0, 0), (offset, 0), (0, 0)), 'constant')[:, :-offset]
    else:
        return np.lib.pad(img, ((0, 0), (0, -offset), (0, 0)), 'constant')[:, -offset:]


def train(model, paths, angles):
    import sklearn.model_selection
    (train_paths, val_paths, train_angles, val_angles) = sklearn.model_selection.train_test_split(*sklearn.utils.shuffle(paths, angles))

    train_images = (get_image(p) for p in train_paths)
    transformed = (t for ia in zip(train_images, train_angles) for t in transform(*ia))
    validation = zip((get_image(p) for p in val_paths), val_angles)
    model.fit_generator(batch(transformed, 25),
                        samples_per_epoch=10000,
                        nb_epoch=20,
                        validation_data=batch(validation, batch_size=300),
                        nb_val_samples=len(val_angles))


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
    model.save_weights('carnd-p3.h5')


if __name__ == '__main__':
    main()
