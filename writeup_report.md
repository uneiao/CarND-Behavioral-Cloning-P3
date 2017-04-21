# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/image1.jpg "Image1"
[image2]: ./images/image2.jpg "Image2"
[image3]: ./images/image3.jpg "Image3"
[image4]: ./images/figure_1.png "Image4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* statistics.py for plotting training loss summary and model structure
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file,
the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model,
and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with several convolutional layers.
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.
A cropping is used to restrict ROI after the lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after each convolutional layer in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was tuned manually as 0.0008,
a little bit slower than default value.

#### 4. Appropriate training data

Training data was collected from training mode of the simulator.
Firstly I simulated center lane driving, and save one lap data of this behavior.
Secondly I tried to recover from the left and right sides of the road, wriggling along roads in fact...
Also one lap for recovering behavior.

Then I began to train the model. The early trained model apparently failed in some scenes,
so I collected more data around the failing scenes and back to training, again and again.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try, one by one.

At first I tried LeNet structure, however the training and validation loss converged so quickly,
that I decided to increase the model capacity. So I changed to the recommended NVIDIA architecture.

The NVIDIA architecture has deeper structure,
as a result it took too long to train on my Macbook without a GPU.
I tried to simplified the model.
I applied grayscaling pre-process on the input data to have smaller inputs (N x 160 x 320 x 1).
I reduced one convolutional layer, and tune some parameters a little bit.
The first convolutional layer down-sampled with a bigger stride(4x4) to save memory.
In each convolutional layer, a batch normalization layer was added before activations to have faster converging and higher accuracy.
Along with a dropout layer after activations, in case of over-fitting the training set.
The new model worked well, it saved 50+% training time each epoch!

The data was split into training set and validation set.
I judged the model by comparing training loss and validation loss,
to see if the model got over-fitting and how well it worked.

My model was trained on first two laps of training data as a cold start.
I checked the performance of trained models in the autonomous mode simulator each time after the model was trained.
The model failed several times around some 'curvy' curves.
Incrementally I collected my behavior data around these failing scenes, including center line driving and recovering.
My idea was that more training data I added, more details were offered to the model to learn.
Repeating input images under particular low frequency scenes helped to generalize better.

It looks like that the keras model.fit method split validation data before it shuffled training data, which means,
training data was not randomly split into training and validation set.
Consequently sometimes the validation loss didn't decrease at all no matter how I edited a model.
Because some hard-case data was appended to the csv file due to my training data collection strategy.
My solution was to shuffle it using random.shuffle on my own before calling model.fit method.

Finally, a well-trained model conquered all the curves in track one. It could drive over and over without leaving the lane.

#### 2. Final Model Architecture

The final model architecture consisted of four convolutional layers and four fully connected layers.

Here is a visualization of the architecture:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 1)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 320, 1)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 17, 79, 18)    468         cropping2d_1[0][0]               
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 17, 79, 18)    72          convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 17, 79, 18)    0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 17, 79, 18)    0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 7, 38, 24)     10824       dropout_1[0][0]                  
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 7, 38, 24)     96          convolution2d_2[0][0]            
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 7, 38, 24)     0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 7, 38, 24)     0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 3, 36, 48)     10416       dropout_2[0][0]                  
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 3, 36, 48)     192         convolution2d_3[0][0]            
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 3, 36, 48)     0           batchnormalization_3[0][0]       
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 3, 36, 48)     0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 34, 64)     27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 1, 34, 64)     256         convolution2d_4[0][0]            
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 1, 34, 64)     0           batchnormalization_4[0][0]       
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 1, 34, 64)     0           activation_4[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2176)          0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           278656      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 32)            4128        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            528         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             17          dense_3[0][0]                    
====================================================================================================
Total params: 333,365
Trainable params: 333,057
Non-trainable params: 308
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center
so that the vehicle would learn to recovery from boundaries.

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would make model generalize better,
for most of curves were only left turns in track one.

Then I continuously added data from failing scenes. Especially some curves without lane line on one side.

![alt text][image3]

After the collection process, I had 4244 number of data points, 25464 training samples after augmentation.
I then preprocessed this data by gray scaling, and normalized them into [-0.5, 0.5] using lambda layer.
I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 20 as evidenced by the statistics of the losses.
I used an adam optimizer, setting 0.0008 as the learning rate.

![alt text][image4]
