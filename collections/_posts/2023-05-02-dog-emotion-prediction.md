---
layout: post
title: "Deep Learning Neural Networks Predicting Dog's Emotions"
date: 2023-05-02T09:49:03Z
categories: ["Deep Learning", "Neural Network", "Image Prediction"]
description: Create a supervised learning model using a dataset of 16,000 dog images to accurately predict their emotions—angry, happy, relaxed, or sad—using a convolutional neural network for a web application.
thumbnail: "/assets/images/gen/content/AMLDog_f08.jpg"
---
🌐 You can reach the project code here: [code]([https://almgcs-bigdataanalytics-cse482-uqbv8c.streamlit.app/](https://github.com/yenything/CMSE890_AppliedMachineLearning/blob/master/CMSE890_AML_FinalProject.ipynb)https://github.com/yenything/CMSE890_AppliedMachineLearning/blob/master/CMSE890_AML_FinalProject.ipynb)
# Using Deep Learning Neural Networks to Predict Dog's Emotions
by Yena Hong

## 1. Abstract

The objective of this project is to develop a supervised learning model using a labeled dataset of 16,000 dog images to predict their emotional state as either angry, happy, relaxed, or sad. The approach employed for image classification is a convolutional neural network (CNN), and it is a multiclass classification problem. The workflow entails data exploration and cleansing, model training and validation, and deploying the model on a web application. The desired outcome is a high degree of accuracy in forecasting dogs' emotions.

## 2.	Introduction

Dogs are increasingly playing important roles in society, such as in rescue operations, drug detection, and therapy. While performing these duties, dogs, like humans, may experience a range of emotions. However, dogs may find it challenging to express their emotions due to their professional training, which prioritizes their job performance over their own emotional needs. This can lead to negative emotional states, and a lack of understanding of a dog's emotions and behavior can result in undesirable situations, such as dogs failing to perform their duties or even biting and attacking humans, leading to fatal outcomes in extreme cases.
Therefore, it is crucial to gain a better understanding of dogs' emotions. By doing so, we can provide them with appropriate care, communication, equipment to alleviate their stress, and breaks during their work. Additionally, dog owners can develop a strong bond with their pets, improve their well-being, and avoid any potential accidents that could arise due to a lack of emotional understanding.

## 3.	Definition
### 3.1.	Task Definition

The task of this project is to classify images of dogs' emotional states into one of four categories based on their emotions. The dataset used for this task consists of 16,000 images, each with a shape of (384, 384, 3). The author explores the use of a simple convolutional neural network (CNN), as well as more complex models such as Interception V3, AlexNet, and OverFeat. The author also investigates the use of principal component analysis (PCA) to reduce the dimensionality of the input images and compares the performance of models trained on both the original and PCA-transformed data. The models are evaluated using standard classification metrics such as accuracy, confusion matrix, and area under receiver operating characteristic (AUROC).

### 3.2.	Algorithm Definition

In this project, the author utilized Principal Component Analysis (PCA) and Convolutional Neural Network (CNN) algorithms as the primary methods of analysis.

#### 3.2.1.	Principal Component Analysis (PCA)

Principal Component Analysis, is a statistical method that reduces the dimensionality of a dataset by identifying patterns and correlations in the data and transforming it into a smaller set of uncorrelated variables called principal components. This allows for easier analysis and visualization of the data.

#### 3.2.2.	Convolutional Neural Network (CNN)

A CNN is a type of neural network that is commonly used in image classification tasks. It consists of several layers, including convolutional layers that extract features from images, pooling layers that reduce the dimensionality of the feature maps, and fully connected layers that classify the images based on the learned features.

##### 3.2.2.1.	InterceptionV3

Interception V3 is a CNN architecture used for image classification tasks, consisting of multiple convolutional and pooling layers followed by fully connected layers. It has achieved high accuracy on benchmark datasets and is widely used for various computer vision applications.

##### 3.2.2.2.	AlexNet

AlexNet is a CNN architecture developed by Alex Krizhevsky, which won the ImageNet Large Scale Visual Recognition Challenge in 2012. It consists of five convolutional layers followed by three fully connected layers, and it uses techniques such as ReLU activation, max pooling, and dropout to improve performance.

##### 3.2.2.3.	OverFeat

OverFeat is a CNN architecture that combines the concepts of object detection and classification into a single network. It uses sliding windows of different sizes to detect objects at different scales and then classifies them using fully connected layers. It was the first CNN to achieve top performance in both object detection and classification tasks.

## 4.	Machine Learning Approach

### 4.1.	Exploratory Data Analysis

#### 4.1.1.	Original Data

This original dataset is provided by Kaggle websites (1), which consist about 16,000 images of dogs. These images have been categorized as four classes: 0: Angry, 1: Happy, 2: Relaxed, and 3: Sad. Figure 1 depicts the number of images in each emotion category. However, there is an imbalance among the classes, with the angry category having half the number of images compared to the other three categories.
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f01.jpg" title="Figure 1. Number of Images in Original Data" %}

In addition to the number of images, some of the images provided in the dataset do not fully capture the faces of dogs. For instance, there are images that feature both dogs and their owners, and some images are not even of dogs, such as those of teddy bears or cats. Furthermore, some of the images are either too small, too dark, or only show a partial face, making it difficult to recognize the dog's emotion. Therefore, we need to clean and preprocess the data to ensure that the machine learning model is trained with accurate images.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f02_1.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f02_2.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f02_3.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f02_4.jpg" title="Figure 2 Sample Dog Images in Original Data" %}

#### 4.1.2.	Cleaned Data

The author carefully chose images that clearly display one dominant emotion and feature a single dog subject, while also possessing bright and clear lighting. To maintain a balanced representation of each emotional category, the author selected an equal number of 150 images from each of the four categories. Additionally, to prevent any ambiguity in the interpretation of the selected images, the author deliberately avoided choosing any images that could be interpreted as conveying multiple emotions.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f0.jpg" title="Figure 3 Number of Images After Image Selection Process" %}

In Figure 4, it can be observed that the images depict clear emotional states solely in the presence of dogs.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f04_1.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f04_2.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f04_3.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f04_4.jpg" title="Figure 4 Sample Dog Images After Image Selection Process" %}

### 4.2.	Data Preprocessing

#### 4.2.1.	Data Augmentation

The initial dataset comprised 16,000 images, but following the selection process, the number of images was reduced to 600, with 150 images for each of the four categories. While this is a relatively small dataset compared to the original, it could potentially lead to over-fitting. In order to address this issue, the author generated augmented images by applying various transformations to the original images.

To specify the augmentation parameters, the author instantiated an instance of the ImageDataGenerator (2) class and set the rotation range to 15 degrees, limiting the random rotation applied to the images. Excessive rotation could potentially hinder the ability of the machine learning algorithm to recognize the dog face accurately.

The width and height shift ranges were set to 0.1, allowing for a 10 percent horizontal and vertical shift to be applied to the images. It is worth noting that setting the shift range too large could lead to cropping of significant parts of the image, which would not be beneficial in terms of leveraging accurate input images.

The shear and zoom ranges were set to 0.1, specifying the intensity of the shear transformation and the range of random zoom to be applied to the images, respectively. Additionally, horizontal flipping was allowed on the original images, as flipping images horizontally does not change the information contained in the input data.

Finally, if image rotation resulted in empty areas, the author filled them with the nearest pixels using the fill_mode parameter. This ensured that the resulting augmented images were as informative as possible while preserving the original features of the dog faces.

Each image was augmented five times, resulting in a total of 3,000 images.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f05.jpg" title="Figure 5  Number of Images After Image Augmentation Process" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f06.jpg" title="Figure 6 Sample Images of Augmentation" %}

#### 4.2.2.	Images Rescaling and Formatting

The author preprocessed the images using ImageDataGenerator rescale function to normalize the pixel values between 0 and 1, and formatted the images to a uniform size of 244 by 244 by 3. These preprocessing steps simplified the training process of the machine learning models and enabled all input images to have the same size.

#### 4.2.3.	Principal Components Analysis

The dataset used in the analysis contains high-resolution images with dimensions of 244 by 244 by 3, resulting in a large and high-dimensional dataset with a total dimensionality of 178,608 for each image. Performing computing operations on such high-dimensional data can be computationally expensive and time-consuming. Therefore, the author explored dimensionality reduction techniques to overcome this issue.

One of the techniques used in our analysis was Principal Component Analysis (PCA) from scikit-learn (3). This technique identifies the most important components of the data and reduces its dimensionality, which makes computations more efficient. To identify the minimum number of principal components required to explain at least 90% of the total variance in the data, we set the explained variance of PCA to 90%.

This approach can lead to more efficient computing operations without significant loss of information. After applying PCA to the image dataset, the author found that retaining 256 (=16 by 16) principal components was optimal for retaining significant features of the original dataset.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f07.jpg" title="Figure 7 Relationship between PCA Components and Explained Variance" %}

After applying PCA, the author reconstructed images to compare the original images with the dimensionality-reduced images. The reconstructed images showed a decrease in quality, with some becoming blurry and losing their original color pixels, as shown below. However, despite the decrease in quality, the reduced-dimensionality images still captured 90% of the variation present in the original dataset, while discarding the remaining 10%.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f08.jpg" title="Figure 8 Original and PCA-reconstructed Images" %}

### 4.3.	Machine Learning Methodology

The authors considered different machine learning methodologies, starting with two types of input data: dimensionality-reduced images obtained through the application of PCA, and original images. They compared the performance of the CNN machine learning model using both input types. Next, the authors applied the Interception V3 model to the original image input, and compared its performance with that of the CNN model. By conducting these comparisons, the authors aimed to assess the impact of dimensionality reduction on model performance, and to determine the optimal machine learning methodology for their dataset.

| Machine Learning   | Dimensionality-Reduced Images                     | Original Images              |
|--------------------|---------------------------------------------------|------------------------------|
| Simple CNN         | Apply CNN on DR images with Hyperparameter Tuning | Apply CNN on Original Images |
| CNN InterceptionV3 | N/A                                               | Apply Interception V3 on Original Images|
| CNN AlexNet        | N/A                                               | Apply AlexNet on Original Images|
| CNN OverFeat       | N/A                                               | Apply OverFeat on Original Images|

* Table 1 Machine Learning Method




## 2. dfdfd
### 2.2. Data Cleaning
#### 2.2.2. dfdfd
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/tweets-1.jpg" title="Figure 2 Sample Dog Images in Original Data" %}
| Numerical x variable | RMSE     |
|----------------------|----------|
| song_duration_ms     | 21.1865  |
| acousticness         | 21.1357  |
| danceability         | 21.0424  |
| energy               | 21.188   |
| instrumentalness     | 21.0152  |
| liveness             | 21.1783  |
| loudness             | 21.0667  |
| speechiness          | 21.1733  |
| tempo                | 21.1833  |
| audio_valence        | 21.1528  |
