#**Traffic Sign Recognition** 

## Rubric Points

Here is a link to my [project code](https://github.com/liz-murphy/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is ?
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.
    Class 0: Speed limit (20km/h), 180 samples



![png](DataVIsualization/output_9_1.png)


    Class 1: Speed limit (30km/h), 1980 samples



![png](DataVIsualization/output_9_3.png)


    Class 2: Speed limit (50km/h), 2010 samples



![png](DataVIsualization/output_9_5.png)


    Class 3: Speed limit (60km/h), 1260 samples



![png](DataVIsualization/output_9_7.png)


    Class 4: Speed limit (70km/h), 1770 samples



![png](DataVIsualization/output_9_9.png)


    Class 5: Speed limit (80km/h), 1650 samples



![png](DataVIsualization/output_9_11.png)


    Class 6: End of speed limit (80km/h), 360 samples



![png](DataVIsualization/output_9_13.png)


    Class 7: Speed limit (100km/h), 1290 samples



![png](DataVIsualization/output_9_15.png)


    Class 8: Speed limit (120km/h), 1260 samples



![png](DataVIsualization/output_9_17.png)


    Class 9: No passing, 1320 samples



![png](DataVIsualization/output_9_19.png)


    Class 10: No passing for vehicles over 3.5 metric tons, 1800 samples



![png](DataVIsualization/output_9_21.png)


    Class 11: Right-of-way at the next intersection, 1170 samples



![png](DataVIsualization/output_9_23.png)


    Class 12: Priority road, 1890 samples



![png](DataVIsualization/output_9_25.png)


    Class 13: Yield, 1920 samples



![png](DataVIsualization/output_9_27.png)


    Class 14: Stop, 690 samples



![png](DataVIsualization/output_9_29.png)


    Class 15: No vehicles, 540 samples



![png](DataVIsualization/output_9_31.png)


    Class 16: Vehicles over 3.5 metric tons prohibited, 360 samples



![png](DataVIsualization/output_9_33.png)


    Class 17: No entry, 990 samples



![png](DataVIsualization/output_9_35.png)


    Class 18: General caution, 1080 samples



![png](DataVIsualization/output_9_37.png)


    Class 19: Dangerous curve to the left, 180 samples



![png](DataVIsualization/output_9_39.png)


    Class 20: Dangerous curve to the right, 300 samples



![png](DataVIsualization/output_9_41.png)


    Class 21: Double curve, 270 samples



![png](DataVIsualization/output_9_43.png)


    Class 22: Bumpy road, 330 samples



![png](DataVIsualization/output_9_45.png)


    Class 23: Slippery road, 450 samples



![png](DataVIsualization/output_9_47.png)


    Class 24: Road narrows on the right, 240 samples



![png](DataVIsualization/output_9_49.png)


    Class 25: Road work, 1350 samples



![png](DataVIsualization/output_9_51.png)


    Class 26: Traffic signals, 540 samples



![png](DataVIsualization/output_9_53.png)


    Class 27: Pedestrians, 210 samples



![png](DataVIsualization/output_9_55.png)


    Class 28: Children crossing, 480 samples



![png](DataVIsualization/output_9_57.png)


    Class 29: Bicycles crossing, 240 samples



![png](DataVIsualization/output_9_59.png)


    Class 30: Beware of ice/snow, 390 samples



![png](DataVIsualization/output_9_61.png)


    Class 31: Wild animals crossing, 690 samples



![png](DataVIsualization/output_9_63.png)


    Class 32: End of all speed and passing limits, 210 samples



![png](DataVIsualization/output_9_65.png)


    Class 33: Turn right ahead, 599 samples



![png](DataVIsualization/output_9_67.png)


    Class 34: Turn left ahead, 360 samples



![png](DataVIsualization/output_9_69.png)


    Class 35: Ahead only, 1080 samples



![png](DataVIsualization/output_9_71.png)


    Class 36: Go straight or right, 330 samples



![png](DataVIsualization/output_9_73.png)


    Class 37: Go straight or left, 180 samples



![png](DataVIsualization/output_9_75.png)


    Class 38: Keep right, 1860 samples



![png](DataVIsualization/output_9_77.png)


    Class 39: Keep left, 270 samples



![png](DataVIsualization/output_9_79.png)


    Class 40: Roundabout mandatory, 300 samples



![png](DataVIsualization/output_9_81.png)


    Class 41: End of no passing, 210 samples



![png](DataVIsualization/output_9_83.png)


    Class 42: End of no passing by vehicles over 3.5 metric tons, 210 samples



![png](DataVIsualization/output_9_85.png)

![png](DataVIsualization/output_11_0.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

- I converted the images to grayscale
- I used histogram normalization (many of the images in the visualization have poor contrast. Also used in this paper).
- I augmented the data. I kept the balance of classes as is, because it may reflect the balance of classes in the test set.

Here is an example of an original image and an augmented image:



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

![png](DataVIsualization/Architecture.001.jpeg)


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

- Kept most of the LeNet lab architecture unchanged
- Learning Rate 0.001, batch size 128
- Increased Epochs to 30 - didn't see much change after 15-20.
- AdamOptimizer
- Xavier initialization to cut down number of parameters.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


