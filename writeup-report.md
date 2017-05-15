# Behavioral Cloning

[//]: # (Image References)

[image1]: ./Images/center-driving.jpg  "Center Driving"
[image2]: ./Images/recovery-left.jpg "Recovery Left"
[image3]: ./Images/recovery-right.jpg  "Recovery Right"
[image4]: ./Images/camera-left.jpg "Camera Left"
[image5]: ./Images/camera-right.jpg  "Camera Right"
[image6]: ./Images/cnn-architecture.png  "Architecture"
[image7]: ./Images/center-driving-track2.jpg "Architecture"

Behavioral Cloning Project

The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road


#### Included Files

The project includes the following files:

model.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
run1.mp4 is a video of the car successfully driving a lap in the simulator

With the udacity provided simulator, the car can be driven autonomously around the track by executing
python drive.py model.h5

### Model Architecture and Training Strategy

#### Model architecture

The final model was a keras implementation of NVIDIA's autonomous driving network.

![alt text][image6] 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   					    | 
| Normalization    	    | x / 255.0 - 0.5 for each pixel 	            |
| Convolution 5x5    	| 2 x 2 stride                                  |
| RELU					|												|
| Convolution 5x5    	| 2 x 2 stride                                  |
| RELU					|												|
| Convolution 5x5    	| 2 x 2 stride                                  |
| RELU					|												|
| Convolution 3x3	    |                                               |   									
| RELU					|												|
| Convolution 3x3	    |                                               |   									
| RELU					|												|
| Flatten               |                                               |
| Fully connected		| outputs 100        						    |
| Fully connected		| outputs 50        						    |
| Fully connected		| outputs 10        						    |
| Output		        | outputs 1        						        |


This architecture was chosen because it has successfully been used for autonomous driving both in simulations and in diverse real world conditions. Multiple convolutions help achieve desirable feature extraction. More about this network can be found in this [paper by NVIDIA ](https://arxiv.org/pdf/1604.07316v1.pdf).

#### Attempts to reduce overfitting

The model was trained on both simulator tracks to prevent overfitting. It was also trained going in the opposite direction on track one.

Dropout was attempted, but it was found that while it marginally improved validation accuracy, the performance on the test track was compromised. 

#### Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually. Accuracy gains, as measured against a validation set, diminished after 5 epochs, so this was the final number of epochs selected.

#### Creation of the Training Set & Training Process

To capture good driving behavior, two laps were recorded on track one using center lane driving. The following is an example image of center lane driving:

![alt text][image1] 

Another lap was recorded with the purpose of providing training data on how to recover if the vehicle reached boundaries. The vehicle was placed on the edges of the track and the desirable action of returning to the center was input.

![alt text][image2] ![alt text][image3] 

A lap of center lane driving was also recorded on track two.

![alt text][image7] 

#### Augmentation
To augment the data set, images were recorded from the left and right edges of the car, in addition to the center images. Because there is only one steering measurement, these images had to have a correction applied to the steering measurement to compensate for their geometric offset. 
Adding these images to the training set has a twofold benefit of training the car to move towards the center of the road and simply adding more data.

![alt text][image4] ![alt text][image5] 

The data was further augmented by flipping the center image and reversing the steering angle. This helps prevent overfitting and helps the model generalize better.
