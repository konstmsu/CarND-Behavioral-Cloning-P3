import numpy as np
import csv
import os
from keras.preprocessing.image import load_img, img_to_array

def load_csv(folder):
    recording_path = lambda path: os.path.join(folder, path)

    image_paths = []
    steering_angles = []
    with open(recording_path("driving_log.csv")) as csv_file:
        for row in csv.reader(csv_file):
            if row[0] == 'center':
                continue
            img_file_name = row[0].replace('\\', '/').split('/')[-1]
            image_paths.append(recording_path("IMG//" + img_file_name))
            steering_angles.append(float(row[3]))
			
    print("%s images in '%s'" % (len(image_paths), folder))

    return (image_paths, steering_angles)

(all_image_paths, all_steering_angles) = load_csv('../recording/track1_round1')
	
def take(array, indexes):
    return [array[i] for i in indexes]

#train_indicies = np.array(np.arange(811, 933))
train_indicies = range(len(all_image_paths))
train_image_paths = take(all_image_paths, train_indicies)
train_steering_angles = np.array(take(all_steering_angles, train_indicies))

def load_image(path):
    return img_to_array(load_img(path)) / 255.0

train_images = np.asarray([load_image(p) for p in train_image_paths])

(test_all_image_paths, test_all_steering_angles) = load_csv('../data')
test_indicies = range(1000)
test_images = np.asarray([load_image(test_all_image_paths[p]) for p in test_indicies])
test_steering_angles = np.array(take(test_all_steering_angles, test_indicies))

#import matplotlib.pyplot as plt
#plt.imshow(images[0])
#print(images[0].shape)

def get_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    
    model = Sequential()
    
    model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    
    model.compile(optimizer="adam", loss="mse")
    
    return model

model = get_model()
model.fit(train_images, train_steering_angles, nb_epoch=4)

quality = model.evaluate(train_images, train_steering_angles)
print("Train fitting: %s" % quality)

quality = model.evaluate(test_images, test_steering_angles)
print("Test error: %s" % quality)

#print("Predicted: %s" % model.predict(images))
#print("Original: %s" % steering_angles)
