#! /usr/bin/env python
"""CarND project 3"""

import os
import csv
import glob
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle


def load_csv(folders):
    paths = []
    angles = []

    for folder in folders:
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

        print("%s images in '%s'" % (row_counter + 1, folder))

    return (paths, angles)

def get_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
    from keras.layers.convolutional import Convolution2D, MaxPooling2D

    model = Sequential()

    #model.add(Lambda(normalize, input_shape=(160, 320, 3), output_shape=(40, 80, 3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(160, 320, 3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1000))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(400))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(150))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(40))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def load_image(path):
    return np.asarray(load_img(path)) / 255.0 - 0.5


def generate(images, angles):
    pass #images


from keras.preprocessing.image import ImageDataGenerator

def train(model, images, angles):
    #test_indices = sorted(range(len(all_angles)), key=lambda i: abs(all_angles[i]))[-20:-1]

    (train_images, train_angles) = shuffle(images, angles)

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
    checkpoint = ModelCheckpoint("carnd-p3.{epoch:02d}-{val_loss:.2f}.h5", save_best_only=True)
    model.fit(train_images, train_angles, nb_epoch=40,
              callbacks=[reduce_lr, early_stop, checkpoint], validation_split=0.2
              #, sample_weight=np.asarray([abs(a) * 3 + 0.3 for a in train_angles])
              )


def save(model):
    print("Saving...")

    with open('carnd-p3.json', 'w') as model_file:
        model_file.write(model.to_json())

    model.save_weights('carnd-p3.h5')


def take(values, indicies):
    return [values[i] for i in indicies]

def main():
    model = get_model()

    print("Saving...")

    with open('carnd-p3.json', 'w') as model_file:
        model_file.write(model.to_json())

    recording_folders = [f for f in glob.glob('../recording/*') if os.path.isdir(f)]
    (all_paths, all_angles) = load_csv(recording_folders)

    train_indicies = [420, 421, 422, 423]
    #train_indicies = range(len(all_paths))
    (train_paths, train_angles) = (take(all_paths, train_indicies), take(all_angles, train_indicies))

    train_images = np.asarray([load_image(p) for p in train_paths])
    train(model, train_images, train_angles)
    save(model)

    test_indices = sorted(range(len(all_angles)), key=lambda i: abs(all_angles[i]))[-20:-1]
    (test_paths, test_angles) = (take(all_paths, test_indices), take(all_angles, test_indices))
    test_images = np.asarray([load_image(p) for p in test_paths])

    test_prediction = model.predict_on_batch(test_images)
    print("Test error: %s" % test_prediction)

    for i in range(len(test_images)):
        test = test_angles[i]
        predicted = test_prediction[i]
        if test == 0 and predicted == 0:
            message = "Expected 0 and got 0"
        else:
            message = "Expected %.2f and predicted %.2f (difference %.2f%%)" % (
                test, predicted, 100 * (test - predicted) / (abs(test) + 0.000001))

        print(message)

        #plt.figure()
        #plt.title(message)
        #plt.imshow(test_images[i] + 0.5)

if __name__ == '__main__':
    main()
