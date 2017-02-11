# Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[border_feature]: ./examples/center_2017_01_28_03_59_58_715.jpg
[bridge]: ./examples/center_2017_01_28_04_00_54_674.jpg
[countryside]: ./examples/center_2017_01_28_04_01_07_383.jpg
[shadow]: ./examples/center_2017_02_01_01_01_12_441.jpg
[model]: ./model.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py carnd-p3.json
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

Input image size is 320x160. Main features that help deciding steering angle is road side, they are usually located at the bottom part of the screen and have maximum dimension of about 50-100 pixels:

![border_feature]
![bridge]
![countryside]
![shadow]

I've decided not to crop input images and instead just train the network more to ignore irrelevant features.

To decrease the number of parameters, first 3 convolution layers have 5x5 kernels and use 2x2 stripes making border features ~8 times smaller. Further convolutions don't need to include more than 7-12 pixels which can get covered by 3 additional convolution layers with 3x3 kernels. The model includes RELU layers to introduce nonlinearity.

Images are converted to HSV, resized to 160x80 and normalized prior to entering the network to save processing power when training.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 38, 78, 5)     380         convolution2d_input_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 38, 78, 5)     0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 37, 7)     882         dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 17, 37, 7)     0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 17, 8)      1408        dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 7, 17, 8)      0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 13, 12)     1452        dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 5, 13, 12)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 11, 12)     1308        dropout_4[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 3, 11, 12)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 1, 9, 12)      1308        dropout_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 108)           0           convolution2d_6[0][0]
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 108)           0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 12)            1308        dropout_6[0][0]
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 12)            0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 6)             78          dropout_7[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             7           dense_2[0][0]
====================================================================================================
Total params: 8,131
Trainable params: 8,131
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after each other layers with parameters in order to reduce overfitting. The model itself is quite small so it doesn't overfit much.

The model was trained and validated on data sets produced by driving several rounds on track 1 and 2 as well as on data set provided by Udacity.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used samples that had non-zero sterring angles at most 2 screenshots before of 5 after (model.py, 43)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use as many convolutional layers as possible keeping the number of parameters low.

I only used center image because I went for the Dominique simulator after reading that training from a mouse-controller car is much easier.

I've tried adding recovery suggestions by shifting input image and augmenting steering angle. I also tried throwing away 98% of training data where steering angle is zero. Results were mixed.

First 10 days I struggled with large networks with millions of parameters. They overfit easily while val_loss sometimes remained quite high and they kept failing in driving test. Training was taking long time and they didn't work well. Now I understand that the problem was in training data quality rather than model size.

So I started looking for a model with just a few parameters. Steering decision is mostly made based on the road curvature and finding it shouldn't need too many parameters.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-89) consisted of 6 convolution layers (that brought the number of rows in the processed image down to one row) followed by 2 fully connected layers.
Dropouts and relu is used after each layer.
![model]

#### 3. Creation of the Training Set & Training Process

I recorded several rounds of track 1 and track 2.
I only took screenshots that at at max 5 screenshots away from a turn.
To augment the data sat, I also flipped images and angles thinking that this would help the car steer in both directions equally well.

I finally randomly shuffled the data set and used `sklearn.model_selection.train_test_split` to split into training and validation sets. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as the model practically stopped training after that.

```
12000/12000 [==============================] - 29s - loss: 0.0590 - val_loss: 0.0512l_loss improved from inf to 0.05119, saving model to carnd-p3.h5
Epoch 2/20
12000/12000 [==============================] - 8s - loss: 0.0525 - val_loss: 0.0496al_loss improved from 0.05119 to 0.04958, saving model to carnd-p3.h5
Epoch 3/20
12000/12000 [==============================] - 5s - loss: 0.0479 - val_loss: 0.0477al_loss improved from 0.04958 to 0.04769, saving model to carnd-p3.h5
Epoch 4/20
12000/12000 [==============================] - 5s - loss: 0.0476 - val_loss: 0.0482al_loss did not improve
Epoch 5/20
12000/12000 [==============================] - 4s - loss: 0.0472 - val_loss: 0.0462al_loss improved from 0.04769 to 0.04622, saving model to carnd-p3.h5
Epoch 6/20
12000/12000 [==============================] - 5s - loss: 0.0472 - val_loss: 0.0461al_loss improved from 0.04622 to 0.04610, saving model to carnd-p3.h5
Epoch 7/20
12000/12000 [==============================] - 4s - loss: 0.0474 - val_loss: 0.0463al_loss did not improve
Epoch 8/20
12000/12000 [==============================] - 4s - loss: 0.0451 - val_loss: 0.0447al_loss improved from 0.04610 to 0.04468, saving model to carnd-p3.h5
Epoch 9/20
12000/12000 [==============================] - 5s - loss: 0.0462 - val_loss: 0.0458al_loss did not improve
Epoch 10/20
12000/12000 [==============================] - 4s - loss: 0.0468 - val_loss: 0.0444al_loss improved from 0.04468 to 0.04438, saving model to carnd-p3.h5
Epoch 11/20
12000/12000 [==============================] - 4s - loss: 0.0472 - val_loss: 0.0462al_loss did not improve
Epoch 12/20
12000/12000 [==============================] - 4s - loss: 0.0446 - val_loss: 0.0456al_loss did not improve
Epoch 13/20
12000/12000 [==============================] - 4s - loss: 0.0445 - val_loss: 0.0460al_loss did not improve
Epoch 14/20
12000/12000 [==============================] - 4s - loss: 0.0455 - val_loss: 0.0450al_loss did not improve
Epoch 15/20
12000/12000 [==============================] - 4s - loss: 0.0467 - val_loss: 0.0451al_loss did not improve
Epoch 16/20
12000/12000 [==============================] - 4s - loss: 0.0454 - val_loss: 0.0455al_loss did not improve
Epoch 17/20
12000/12000 [==============================] - 4s - loss: 0.0439 - val_loss: 0.0452al_loss did not improve
Epoch 18/20
12000/12000 [==============================] - 4s - loss: 0.0449 - val_loss: 0.0456al_loss did not improve
Epoch 19/20
12000/12000 [==============================] - 4s - loss: 0.0453 - val_loss: 0.0439al_loss improved from 0.04438 to 0.04393, saving model to carnd-p3.h5
Epoch 20/20
12000/12000 [==============================] - 4s - loss: 0.0464 - val_loss: 0.0443al_loss did not improve
```