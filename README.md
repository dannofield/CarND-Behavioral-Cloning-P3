# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./IMG_RESULTS/center_2016_12_01_13_30_48_287.jpg "Center 1"
[image2]: ./IMG_RESULTS/center_2016_12_01_13_31_13_037.jpg "Center 2"
[image3]: ./IMG_RESULTS/center_cropped1.jpg "center cropped 1"
[image4]: ./IMG_RESULTS/center_cropped1G.jpg "center cropped 1 Gray"
[image5]: ./IMG_RESULTS/center_cropped2.jpg "center cropped 2"
[image6]: ./IMG_RESULTS/center_cropped2G.jpg "center cropped 2 Gray"

[image8]: ./IMG_RESULTS/left_2016_12_01_13_30_48_404.jpg "Left 1"
[image9]: ./IMG_RESULTS/left_2016_12_01_13_31_13_037.jpg "Left 2"
[image10]: ./IMG_RESULTS/left_cropped1.jpg "left_cropped 1"
[image11]: ./IMG_RESULTS/left_cropped1G.jpg "left_cropped 1 Gray"
[image12]: ./IMG_RESULTS/left_cropped2.jpg "left_cropped 2"
[image13]: ./IMG_RESULTS/left_cropped2G.jpg "left_cropped 2 Gray"

[image14]: ./IMG_RESULTS/right_2016_12_01_13_30_48_287.jpg "right 1"
[image15]: ./IMG_RESULTS/right_2016_12_01_13_31_12_937.jpg "right 2"
[image16]: ./IMG_RESULTS/right_cropped1.jpg "cright_cropped 1"
[image17]: ./IMG_RESULTS/right_cropped1G.jpg "right_cropped 1 Gray"
[image18]: ./IMG_RESULTS/right_cropped2.jpg "right_cropped 2"
[image19]: ./IMG_RESULTS/right_cropped2G.jpg "right_cropped 2 Gray"

[image20]: ./IMG_RESULTS/center_2016_12_01_13_30_48_287.jpg "Center 1"

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

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After a few tests using LeNet architecture, I decided to give it a try to the architecture proposed by [NVIDIA](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

My model is defined in model.py lines 87-141 and consists of:

| Layer         		    |     Description	        					      | 
|:---------------------:|:---------------------------------------:| 
| Input         		    | 320,92,1 Gray & normalized image   			| 
| Convolution          	| 24 5x5 stride, same padding            	|
| RELU					        |												                  |
| Convolution          	| 36 5x5 stride, same padding            	|
| RELU					        |												                  |
| Convolution          	| 48 5x5 stride, same padding            	|
| RELU					        |												                  |
| Convolution          	| 64 3x3 stride, same padding            	|
| RELU					        |												                  |
| Convolution          	| 64 3x3 stride, same padding            	|
| RELU					        |												                  |
| Fully connected		    | 100 Neurons									            |
| Fully connected		    | 50 Neurons									            |
| Fully connected		    | 1  Neurons									            |


The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 136).

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the simulator for two laps at the speed I knew the autonomous car was going to drive it.

This was important because if I trained the model at low speed I was able to turn with bigger steering angles but the autonomous car could not do it.

I ended up using a convolution neural network model similar to the model proposed for NVIDIA and after a few hyperparameters being tuned I was able to achieve great results.

I thought this model might be appropriate because the paper they published expose good results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I found that my first modelwas not able to achive the best mean squared error on the training set so I incremented the number of epchos. 

After I increased the number of epchos it started to oscilate around. This was a signal that I needed to reduce the number of epchos until I found the right one. Five epchos seemed to fix it. 

The final step was to run the simulator to see how well the car was driving around track one.

There were a few spots where the vehicle fell off the track.
To improve the driving behavior in these cases, I decided to use all three cameras, center, left and right. I used the left and right cameras with a corrected steering angle of 2 at first.

The model didn´t go off the road but it was moving the steering continually. So I decreased the correction to 0.7. This time the car seemed to perform way better.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture

![alt text][image20]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]
![alt text][image2]

I used the left camera with a correction angle of 0.7

![alt text][image8]
![alt text][image9]

Then I used the right camera with a correction angle of 0.7

![alt text][image14]
![alt text][image15]

Then I repeated this process on track two in order to get more data points.

Since I used all 3 cameras I did not have to augment the data set by flipped images.

After the collection process, I had around 9000 number of data points. I then preprocessed this data by cropping the images so the model does not get distracted by things like the sky, trees, etc.:

Here is the center camera image cropped
![alt text][image3]
![alt text][image5]

Left camera image cropped
![alt text][image10]
![alt text][image12]

Right camera image cropped
![alt text][image16]
![alt text][image18]

After this I normalized the data by converting this data to gray scale.

Here is the center camera image cropped
![alt text][image4]
![alt text][image6]

Left camera image cropped
![alt text][image11]
![alt text][image13]

Right camera image cropped
![alt text][image17]
![alt text][image19]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

