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

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f03.jpg" title="Figure 3 Number of Images After Image Selection Process" %}

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

*Table 1 Machine Learning Method*
 
#### 4.3.1.	Train, Validation, and Testing Split

In order to train and test their models, the authors split their dataset into three subsets. The largest subset, comprising 70% of the data, was used for training, while 20% was reserved for validation and the remaining 10% was set aside for testing. Each subset contained a different number of images: 2,100 for training, 600 for validation, and 300 for testing. The authors then applied both the CNN and Interception V3 algorithms to the split data to compare their performance.

#### 4.3.2.	Training Model 1: Simple CNN

The author utilized Convolutional Neural Networks (CNNs) for image and video analysis in deep learning. The CNN architecture was applied to analyze two types of input images: dimensionality-reduced images of size 16x16x3 and original images of size 244x244x3.

##### 4.3.2.1.	Using Dimensionality-Reduced Images

###### 4.3.2.1.1.	Methods

The author developed CNNs in TensorFlow (4) for dogs’ emotional states prediction. The author considered the following network architecture in an investigation:

[Conv-ReLU-MaxPool-Dropout] - [Conv-ReLU-MaxPool-Dropout] – [Flatten] - [Dense-ReLU-Dropout] – [Dense-Softmax]

The model has a 2-layer convolutional neural network with ReLU activation function and 32 and 64 filters of size 3x3 for the first and fourth layers, respectively. It also has two max-pooling layers that reduce spatial dimensions by a factor of 2. Dropout regularization is applied with a rate of 0.1 and 0.2 for the third and ninth layers, respectively.

The model has two fully connected layers, the first with one neuron and the second with four neurons that use the ReLU and softmax activation functions, respectively. And finally, a danse layer with a softmax activation function that outputs the probabilities for the four classes.

###### 4.3.2.1.2.	Hyperparameter Tuning

The author applied hyperparameters in the CNN model, including batch size, epoch, learning rate, and optimizers. To find the optimal hyperparameters, the author conducted a grid search with 5-fold cross-validation, varying the batch size with 32, 64, and 128, the number of epochs with 50 and 100, and the learning rate with 0.001 and 0.0001 for optimizers Adam and RMSprop.

After training all the hyperparameters, the model performed best with a batch size of 64, 50 epochs, and a learning rate of 0.001 for both optimizers Adam and RMSprop, with an accuracy of 23.47%.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f09.jpg" title="Figure 9 Grid Search Result for Adam Optimizer" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f10.jpg" title="Figure 10 Grid Search Result for RMSprop Optimizer" %}

###### 4.3.2.1.3.	Training and Validation

After conducting a previous Hyperparameter Tuning process, the author selected the best hyperparameter, which included using the Adam optimizer, a batch size of 64, 50 epochs, and a learning rate of 0.001. The author then proceeded to calculate accuracy and loss for both the training data and the validation data using this hyperparameter. The author used the categorical crossentropy Loss function (5) for these calculations.

The results of this evaluation were presented in a figure, which showed that the training accuracy remained in range between 20 to 30%, and the validation accuracy decreased from 25% to 21% after the fifth iteration out of 50 epochs. Moreover, both the training loss and validation loss were plateaued around 1.386, indicating that the model did not improve over time. 

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f11.jpg" title="Figure 11 Training and Validation Accuracy using PCA images with Simple CNN" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f12.jpg" title="Figure 12 Training and Validation Loss using PCA images with Simple CNN" %}

##### 4.3.2.2.	Using Original Images

The evaluation of the CNN models with different combinations of hyperparameters revealed no significant differences in accuracy, as demonstrated by the figures above. Furthermore, even the best achieved accuracy was found to be inadequate. Consequently, in an attempt to improve model performance, the author decided to train CNN models using the original images.

###### 4.3.2.2.1.	Methods

The following network architecture is used to train the original input data:

[Conv-ReLU-MaxPool-Dropout] - [Conv-ReLU-MaxPool-Dropout] – [Flatten] - [Dense-ReLU-Dropout] – [Dense-Softmax]

It consists of two convolutional layers, each followed by a max-pooling layer and a dropout layer to prevent overfitting. The flattened output is passed through a fully connected layer with ReLU activation function, followed by another dropout layer, and finally a dense layer with softmax activation function to output the probabilities for the four classes.

It begins with a convolutional layer with 32 filters and 'relu' activation function, followed by a max-pooling layer and dropout layer. The same pattern is repeated with more filters for the next two convolutional layers. After the third convolutional layer, there is another max-pooling layer followed by a dropout layer. The output is then flattened and fed into a dense layer with 512 units and 'relu' activation function, followed by another dropout layer. The final dense layer has 4 units, one for each class, and uses 'softmax' activation function to output the probabilities of each class.

###### 4.3.2.2.2.	Hyperparameters

The hyperparameters used for the CNN model with the original images are 50 epochs and the Adam optimizer.

###### 4.3.2.2.3.	Training and Validation

The model was trained with a training loss of 5.0239, and a training accuracy of 0.3313 in the first epoch. The validation loss and accuracy were 1.5123 and 0.2633, respectively. The training loss and accuracy improved gradually in the subsequent epochs, but the validation loss and accuracy remained relatively constant. This suggests that the model may be overfitting to the training data and not generalizing well to new data.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f13.jpg" title="Figure 13 Training and Validation Accuracy using Original images with Simple CNN" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f14.jpg" title="Figure 14 Training and Validation Loss using Original images with Simple CNN" %}

#### 4.3.3.	Training Model 2: Interception V3 Algorithm

The model sets the output of a pre-trained model as the input to custom layers. The custom layers include a global average pooling layer, a dense layer with 512 neurons and ReLU activation function, a dropout layer with a rate of 0.2, and finally an output layer with softmax activation function that outputs the predicted probability distribution over the classes.

##### 4.3.3.1.	Hyperparameters

The hyperparameters used for the Interception V3 model with the original images are 50 epochs and the Adam optimizer.

##### 4.3.3.2.	Training and Validation

From the output, we can see that the training accuracy starts at 0.2875 and gradually increases over the epochs, reaching 0.8375 by epoch 21. This suggests that the model is learning to classify the training data more accurately as it receives more training.

The validation accuracy starts at 0.3967 and also increases over the epochs, reaching 0.7942 by epoch 21. This suggests that the model is also improving its ability to generalize to unseen data, which is indicated by the validation data.

The training loss and validation loss both start high and gradually decrease over the epochs, indicating that the model is improving its ability to classify the data correctly.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f15.jpg" title="Figure 15 Training and Validation Accuracy using Original images with InterceptionV3" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f16.jpg" title="Figure 16 Training and Validation Loss using Original images with InterceptionV3" %}

#### 4.3.4.	Training Model 3: AlexNet

The model includes five convolutional layers, each with ReLU activation. The first convolutional layer has 96 filters and a kernel size of 11x11, and a stride of 4x4. The next four convolutional layers have 256, 384, 384, and 256 filters, respectively, and a kernel size of 5x5 or 3x3. The strides for the remaining layers are 1x1. The padding for the convolutional layers is set to 'same' for all layers except the first one.

After each convolutional layer, a max-pooling layer with a pool size of 3x3 and stride of 2 is applied. Max-pooling reduces the spatial dimensions of the output feature maps and helps to make the model more robust to small variations in the input images. Batch normalization is also applied after each convolutional layer, which helps to improve the training stability and speed up the convergence of the model.

The model includes two fully connected (dense) layers with 4096 units each and ReLU activation. Dropout regularization with a rate of 0.4 is applied after each dense layer.

The output layer is a dense layer with num_classes units and softmax activation. The softmax function normalizes the outputs of the output layer so that they represent probabilities of the input image belonging to each class.

##### 4.3.4.1.	Hyperparameters

The hyperparameters used for the AlexNet model with the original images are 50 epochs and the Adam optimizer.

##### 4.3.4.2.	Training and Validation

In general, it can be seen that the model is not performing well, as the loss is quite high and the accuracy is relatively low. There are fluctuations in both the loss and accuracy values throughout the epochs, which indicates that the model is not stable and may not be learning effectively.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f17.jpg" title="Figure 17 Training and Validation Accuracy using Original images with AlexNet" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f18.jpg" title="Figure 18 Training and Validation Loss using Original images with AlexNet" %}

#### 4.3.5.	Training Model 4: OverFeat

The model with 5 convolutional layers and 3 fully connected layers. The input to the network is an image of size 224x224 with 3 color channels. The first layer has 96 filters of size 11x11 with a stride of 4 and ReLU activation, followed by max pooling, and batch normalization. The following layers have 256, 512, 1024, and 1024 filters respectively, with smaller filter sizes and padding to keep the output size the same. The final convolutional layer is followed by max pooling and batch normalization. The output of the convolutional layers is then flattened and passed through two fully connected layers with 3072 and 4096 units respectively, each followed by ReLU activation and dropout regularization. The output layer has 4 units with softmax activation for classification.

##### 4.3.5.1.	Hyperparameters

The hyperparameters used for the OverFeat model with the original images are 50 epochs and the Adam optimizer.

##### 4.3.5.2.	Training and Validation

Looking at the results, it seems that the model's performance fluctuated throughout the training process. The initial accuracy was 0.3063 and remained relatively stable until the end. However, the loss increased, indicating that the model was not fitting the training data very well. The validation accuracy was very low and remained mostly unchanged throughout the training process, indicating that the model was not able to generalize well to new data. The validation loss increased significantly, indicating that the model was overfitting to the training data.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f19.jpg" title="Figure 19 Training and Validation Accuracy using Original images with OverFeat" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f20.jpg" title="Figure 20 Training and Validation Loss using Original images with OverFeat" %}

#### 4.3.6.	Testing

The author evaluated the accuracy and loss of four different models using testing data. InceptionV3 demonstrated the highest accuracy of 0.83 and the lowest loss of 0.4877 among the four models. Based on these results, the final model for predicting a dog's emotions will use InceptionV3.

| Model        | Accuracy | Loss   |
| ------------ | -------- | ------ |
| Simple CNN   | 0.3867   | 1.2750 |
| Inception V3 | 0.8300   | 0.4877 |
| AlexNet      | 0.5000   | 1.2707 |
| OverFeat     | 0.4133   | 1.4923 |

*Table 2 Accuracy and Loss Compare for CNNs*

### 4.4.	Prediction Results

The author used InterceptionV3 as the final model, and predicted the emotional states of dogs using the model. And, the author generated confusion matrix and Receiver Operating Characteristic (ROC) curve to visualize the result.

#### 4.4.1.	Confusion Matrix

The confusion matrix is a table that summarizes the performance of a classification algorithm. In this case, the matrix is a 4x4 table where each row represents the true class and each column represents the predicted class. The numbers in the table represent the number of samples that belong to each class.

Looking at the matrix, there are 4 classes: angry, happy, relaxed, and sad. The first row indicates that there were 21 samples of class angry that were correctly classified as angry, 20 samples of class angry that were misclassified as happy, 18 samples of class angry that were misclassified as relaxed, and 16 samples of class angry that were misclassified as sad.

The second row indicates that there were 14 samples of class happy that were correctly classified as happy, 18 samples of class happy that were misclassified as angry, 11 samples of class happy that were misclassified as relaxed, and 32 samples of class happy that were misclassified as sad.

The overall performance of the classifier can be evaluated by looking at the diagonal elements. The higher the diagonal elements, the better the classifier’s performance. In this case, the diagonal elements are not very high, indicating that the classifier may not be very accurate.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f21.jpg" title="Figure 21 Confusion Matrix of Testing Data" %}

#### 4.4.2.	Receiver Operating Characteristic (ROC)

The ROC curve provides a visual representation of a classifier's performance by plotting the true positive rate (TPR) against the false positive rate (FPR) at different classification thresholds. The area under the ROC curve (AUC) is a common metric used to assess the overall performance of the classifier, where a perfect classifier would have an AUC of 1, and a random classifier would have an AUC of 0.5.

In this case, the classifier's AUC values range from 0.41 to 0.52 for the four classes, indicating that it is better than random at distinguishing between the classes. However, the "sad" class has an AUC value of 0.41, which suggests that the classifier may struggle to differentiate between samples in this class compared to the others. This could be due to factors such as high variability within the class or similarities to other classes.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f22.jpg" title="Figure 22 ROC Curve of Testing Data" %}

#### 4.4.3.	Misclassification Sample Images

The following are examples of misclassifications from each class, which demonstrate some of the challenges faced in accurate image classification.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AMLDog_f23.jpg" title="Figure 23 Sample Images of Misclassification" %}

### 4.5.	Conclusion

While the machine learning models exhibited high accuracy and low loss during the training and validation phases, the performance of the model during real-time prediction was not as satisfactory, with a test accuracy of only 0.83. This could be attributed to the phenomenon of overfitting, where the model has become too specialized to the training data and unable to generalize well to new data. Further investigation and improvements in the training process may be necessary to enhance the model's real-time prediction capabilities.

## 5.	Summary

### 5.1.	Goal, Methods, and Recap

The goal of the project is to accurately predict dogs’ emotions from images. Predict a dog emotion is particularly beneficial to dog owners who lack experience in breeding dogs, as it will help them understand their pets' emotions and develop a strong bond with them.

The method of machine learning for this project involved exploratory data analysis, data preprocessing, and the training and testing of several models using different algorithms. The cleaned dataset of 600 dog images was preprocessed through data augmentation, rescaling, and formatting. Principal component analysis was also applied to the dataset to reduce its dimensionality.

Several models were trained and tested, including a simple CNN model using both dimensionality-reduced and original images, as well as more complex models using the Interception V3, AlexNet, and OverFeat algorithms. The models were evaluated using confusion matrices and receiver operating characteristic curves, and misclassification sample images were also examined.

Overall, the results showed promise for predicting dogs' emotions, although further improvements could be made.

### 5.2.	Future work

In the future, there are several potential avenues for improvement in this project. One possibility is to investigate various techniques for automating the image preprocessing step, without requiring manual intervention. This would increase efficiency and allow for faster model training and evaluation. 

Moreover, to improve the model's performance, it is essential to use a larger and more diverse dataset with high-quality images. The current dataset used in this project lacks good quality images, which can potentially limit the model's ability to accurately classify emotions. Furthermore, it may be worth considering the possibility of classifying emotions into multiple categories, as subtle emotional states may be present in the images that are not captured by the current classification scheme.

With regard to hyperparameter tuning, the model trained on the original data did not undergo hyperparameter tuning due to the high computational cost. Therefore, in future work, it may be beneficial to utilize a more powerful computing resource to optimize hyperparameters and potentially improve the model's performance. 

Lastly, while a web app was not created in this project due to the model's low performance on real data, this can be revisited in the future as the model continues to improve. A functional web app can be an effective tool for showcasing the model's performance to a wider audience and potentially increasing its practical utility.

## 6.	References

(1) https://www.kaggle.com/datasets/devzohaib/dog-emotions-prediction

(2) https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

(3) https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

(4) https://www.tensorflow.org/tutorials/images/cnn

(5) https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
