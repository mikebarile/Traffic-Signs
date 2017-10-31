# **Traffic Sign Recognition**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/sample_image.png "Sample Image"
[image2]: ./examples/grayscale.jpg "Grayscale"
[image3]: ./traffic-sign-examples/test_1.png "Test image 1"
[image4]: ./traffic-sign-examples/test_2.png "Test image 2"
[image5]: ./traffic-sign-examples/test_3.png "Test image 3"
[image6]: ./traffic-sign-examples/test_4.png "Test image 4"
[image7]: ./traffic-sign-examples/test_5.png "Test image 5"
[image8]: ./traffic-sign-examples/test_6.png "Test image 6"

---

### Data Set Summary & Exploration

Below is a summary of the traffic signs dataset that I generated using the shape property of the training images / labels arrays:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

To get a sense for how the data was distributed amongst the various labels, I created and printed a counter which had the following output:
```
Counter({2: 2010, 1: 1980, 13: 1920, 12: 1890, 38: 1860, 10: 1800, 4: 1770, 5: 1650, 25: 1350, 9: 1320, 7: 1290, 3: 1260, 8: 1260, 11: 1170, 18: 1080, 35: 1080, 17: 990, 14: 690, 31: 690, 33: 599, 15: 540, 26: 540, 28: 480, 23: 450, 30: 390, 6: 360, 16: 360, 34: 360, 22: 330, 36: 330, 20: 300, 40: 300, 21: 270, 39: 270, 24: 240, 29: 240, 27: 210, 32: 210, 41: 210, 42: 210, 0: 180, 19: 180, 37: 180})
```

The dataset is skewed and the smallest label by sample size has ~1/10 the samples as the largest.

I also printed a random image with its label classification to test that the images were loaded properly.

![alt text][image1]

### Data preprocessing and visualization

At first, I tried converting the images to greyscale. Example below:

![alt text][image2]

Later on, I found that the greyscale conversion was actually hurting model performance. I ended up making the greyscale optional by allowing the model user to set the number of color channels (1 or 3).

I also tried augmenting the dataset. At first, I tried 5x'ing the number of samples using a variety of image tweaks but found that this hurt performance by ~10%. In my final implementation, I augmented the original training dataset slightly and used that for my final training dataset (without adding any samples).

As a last step, I normalized the data to values between 0 and 1. This was to simplify the input we are feeding into the tensorflow model and to boost performance.

### Model architecture

My final model consisted of the following layers:

* Input
* Convolution
* Relu
* Max pooling
* Convolution
* Relu
* Fully connected
* Softmax

The architecture is based off of LeNet but with the following tweaks:
* Removed the 2nd max pooling layer to boost performance
* Added dropout to improve generalization

### Model Training

I trained the model using the following architecture:
* Softmax logits using architecture listed above
* Cross entropy: tf.nn.softmax_cross_entropy_with_logits
* Loss operation: tf.reduce_mean
* Optimizer: tf.train.AdamOptimizer
* Training operation: optimizer.minimize

I also used the following hyperparameters:
* EPOCHS = 100
* BATCH_SIZE = 512
* KEEP_PROB = 0.80
* LEARNING_RATE = 0.001
* COLOR_CHANNELS = 3

### Iteration to achieve >93% validation set accuracy

My final model results were:
* Training set accuracy: 99.8%
* Validation set accuracy: 94.6%
* Testing set accuracy: 91.3%

At first I tried implementing the LeNet architecture with greyscale and 0.8 dropout. LeNet is an excellent configuration for training CNNs. After training, I was only able to achieve 89% accuracy on the validation set.

To iterate, I first tried moving around some of the hyperparameters. I tried increasing the batch size by multiplying it by 2 several times with little effect on training accuracy. I also tried removing grayscale which boosted the accuracy to 91%.

Next I tried augmenting the dataset. At first, I used around 5 different image augmentations with a relatively high probability of activation. I also increased the entire training dataset by ~5x. After training, my validation accuracy dropped from 91% to ~85%.

In my next iteration, I tried lowering the magnitude of image augmentation and maintained a test set that was the same size as the original training set. I also removed the 2nd max pooling layer from my LeNet architecture - this is because my GPU was able to handle the compute load. Removing the pooling layer and reducing the image augmentation gave me the final boost I needed with a final 94.6% validation set accuracy.

### Testing the model on new images

Here are six German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

All six of these images seem reasonably similar to the images found in the training set.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1. Speed limit (30km/h)      		| 1. Speed limit (30km/h)   									|
| 4. Speed limit (70km/h)     			| 7. Speed limit (100km/h) 										|
| 7. Speed limit (100km/h)					| 7. Speed limit (100km/h)											|
| 17. No entry	      		| 17. No entry					 				|
| 30. Beware of ice/snow			| 11. Right-of-way at the next intersection      							|
| 25. Road work			| 25. Road work      							|

The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.7%. This is significantly lower than the accuracy in the test set which was 91.3%. Possible reasons for why this might be include:
* A sample size of 6 is extremely low
* The images in the training set could have had elements of standardization that weren't included in these images from the web
* In the case of "Beware of ice/snow", there were significantly fewer results in the training set than there were for the other classes (390 samples)

My model was 100% certain it was correct in all six of these test cases. It's clearly an arrogant model :)
