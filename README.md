**This repository presents a curated collection of end-to-end Machine Learning and Deep Learning projects developed to demonstrate practical skills across key areas of 
Artificial Intelligence, including Computer Vision, Natural Language Processing, and Predictive Analytics. The projects cover real-world problems such as object recognition, 
image super-resolution, text classification, fraud detection, and review score prediction, and showcase the complete pipeline from data preprocessing and model development to 
training, evaluation, and result visualization.**

project:**Object Recognition** 
## Description:
In this project :
* Import datasets from Keras
* Use one-hot vectors for categorical labels
* Addlayers to a Keras model
* Load pre-trained weights/trained weights(in this project i have trained weights considering only 3 epochs since 350 epochs takes approximately 10hrs for training)
* Make predictions using a trained Keras model

The dataset we will be using is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## problem Statement: 
To develop a deep learning model capable of automatically recognizing and classifying objects in images into predefined categories with high accuracy

## Dataset:
we will use cifar10 dataset which can be imported from keras as: from keras.datasets import cifar10

## Tools and Libraries 
*Keras / tensorflow
*python
*numpy

## Accuracy 
60.59% (considering only 3 epochs)

## Results: 

![objectrecog](https://github.com/user-attachments/assets/67877e59-50d4-4aa8-a8b9-a8972f0ab06d) 

project: **Image Super Resolution**

# Description: 
During this project  : 

* use the PSNR, MSE, and SSIM image quality metrics,
* process images using OpenCV,
* convert between the RGB, BGR, and YCrCb color spaces,
* build deep neural networks in Keras,
* deploy and evaluate the SRCNN network

## Problem Statement
To design and train a deep learning model that can reconstruct high-resolution images from low-resolution inputs by learning the underlying spatial details and texture patterns, thereby improving visual quality and preserving important image features.

## Dataset 
we will be using a dataset from  link http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html by downloading this zip files ensuring using the image contained in set 5 and set 14 folders only.

## Tools  and Libraries
*python 
*keras
*opencv
*matplotlib
*numpy
*skiimage

## Accuracy
used pretrained model due to problem of training in  device 

# Results 
one sample result is shown below and other saved execution is also shown
![imgsuperresol1](https://github.com/user-attachments/assets/a0bcb73e-86b2-4f98-8bfa-243b057d4825)
![imgsuperresol2](https://github.com/user-attachments/assets/d9fc855c-36d2-409f-8533-7a1502beb5c0)

project: **NLP- text classification**
## Description
 We will use:
*sentence and word tokenization
*stop words
*stemming
*part of speech tagging
*chunking
*chinking
*named identity recognition
* Regular Expressions
* Feature Engineering
* Multiple scikit-learn Classifiers
* Ensemble Methods

## problem statement
To develop a machine learning model that can automatically classify text messages as **spam** or **ham (non-spam)** by analyzing the content and patterns of the messages. 

## Datasets from online resources and added some

## Tools and Libraries
*python
*numpy
*nltk 
*pandas
*sklearn

## Accuracy
98.20%

## Results 
![classifireport](https://github.com/user-attachments/assets/07522f0a-7cd9-433f-aaec-c7b9f7126105)

project:**Credit card Fraud Detection**

## Description
In this project,  will build and deploy the following two machine learning algorithms:

* Local Outlier Factor (LOF)
* Isolation Forest Algorithm
Furthermore, using metrics suchs as precision, recall, and F1-scores, we will investigate why the classification accuracy for these algorithms can be misleading.

In addition,  will explore the use of data visualization techniques common in data science, such as parameter histograms and correlation matrices, to gain a better understanding of the underlying distribution of data in our data set. 
## Problem Statement
To build a predictive model that can accurately identify fraudulent credit card transactions from legitimate ones by analyzing transaction patterns and detecting anomalies, helping prevent financial losses and improve security.
## Dataset
used datasets including approx 29000 credit card transcations
## Tools and Libraries
*python'
*numpy
*pandas
*matplotlib
*seaborn
## Accuracy :
isolation forest: 0.997%
local outlier factor: 0.996%
## Results
![classrepo2](https://github.com/user-attachments/assets/4b932f6d-4b6b-4946-80bb-30eaa6f6f2e4)













  













