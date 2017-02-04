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


image_size = (160, 320)


def load_csv(folders):
    paths = []
    angles = []

    total_images = 0
    for folder in sorted(f for f in folders if 'track1' in f):
        with open(os.path.join(folder, "driving_log.csv")) as csv_file:
            row_counter = -1
            for row in csv.reader(csv_file):
                if row[0] == 'center':
                    continue

                row_counter += 1

                angle = float(row[3])

                if abs(angle) < 0.01 and row_counter % 10 < 8:
                    continue

                img_file_name = row[0].replace('\\', '/').split('/')[-1]
                paths.append(os.path.join(folder, "IMG", img_file_name))
                angles.append(angle)

        total_images += row_counter + 1
        print("%s images in '%s'" % (row_counter + 1, folder))

    print("%s images in total" % total_images)

    return (paths, angles)


def get_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
    from keras.layers.convolutional import Convolution2D, MaxPooling2D

    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(*image_size, 3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def load_image(path):
    return np.array(load_img(path, target_size=image_size)) / 255.0 - 0.5

image_cache = {}

def get_image(path):
    global image_cache

    if path not in image_cache:
        image_cache[path] = load_image(path)
        
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


def train(model, paths, angles):
    (train_paths, val_paths, train_angles, val_angles) = sklearn.model_selection.train_test_split(*sklearn.utils.shuffle(paths, angles))

    #try:
    #    from matplotlib import pyplot as plt
    #    plt.figure()
    #    plt.imshow(load_image(train_paths[0]) + 0.5)
    #except ImportError as e:
    #    print("Not going to save mode png: %s" % e)

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001)
    checkpoint = ModelCheckpoint("carnd-p3.{epoch:02d}-{val_loss:.2f}.h5", save_best_only=True)
    train_images = (get_image(p) for p in train_paths)
    transformed = itertools.chain.from_iterable(transform(i, a) for (i, a) in itertools.cycle(zip(train_images, train_angles)))
    model.fit_generator(batch1(transformed),
                        samples_per_epoch=300,
                        nb_epoch=5,
                        validation_data=batch(itertools.cycle(get_image(p) for p in val_paths), itertools.cycle(val_angles)),
                        nb_val_samples=len(val_angles),
                        callbacks=[checkpoint]
                       )


def take(values, indicies):
    return [values[i] for i in indicies]


def main():
    model = get_model()

    try:        
        from keras.utils.visualize_util import plot
        plot(model, show_shapes=True)
    except ImportError as e:
        print("Could not save model schema: %s" % e)

    print("Saving...")

    with open('carnd-p3.json', 'w') as model_file:
        model_file.write(model.to_json())

    recording_folders = [f for f in glob.glob('../recording/*') if os.path.isdir(f)]
    (all_paths, all_angles) = load_csv(recording_folders)

    #train_indicies = range(len(all_paths))
    train_indicies = list(itertools.chain(range(20, 100), range(700, 720), range(820, 850)))
    train_paths, train_angles = take(all_paths, train_indicies), take(all_angles, train_indicies)

    model.load_weights('carnd-p3.h5')
    train(model, train_paths, train_angles)
    model.save_weights('carnd-p3.h5')

    test_indices = sorted(range(len(all_angles)), key=lambda i: abs(all_angles[i]))[-198:-1]
    (test_paths, test_angles) = (take(all_paths, test_indices), take(all_angles, test_indices))

    test_prediction = model.predict(np.asarray([get_image(p) for p in test_paths]))
    print("Test error: %s" % test_prediction)

    for test, predicted in zip(test_angles, test_prediction):
        if test == 0 and predicted == 0:
            message = "Expected 0 and got 0"
        else:
            message = "Expected %.2f and predicted %.2f (difference %.2f%%)" % (
                test, predicted, 100 * (test - predicted) / (abs(test) + 0.000001))

        print(message)


if __name__ == '__main__':
    main()
