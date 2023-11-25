---
layout: post
title: "Android Malware Detection"
date: 2023-07-05T09:49:03Z
categories: ["Massive Data", "Feature Engineering", "Machine Learning"]
description: Develop a robust machine learning model to accurately classify Android malware into distinct types and differentiate between malignant and benign samples.
thumbnail: "/assets/images/gen/content/AndMal_PPT4.jpg"
---

🌐 You can reach the project code here: [Project Code](https://github.com/yenything/STT811_StatisticalModeling)
# Android Malware Detection
by
Yena Hong, Megha Viswanath, Lacey Hamilton

## ABSTRACT

Android malware detection and classification are essential tasks in maintaining the security of mobile devices and users' data. This project aims to develop a robust machine learning model to accurately classify Android malware into distinct types and differentiate between malignant and benign samples. Through a series of Jupyter notebooks, we perform an exploratory data analysis, binary classification between malignant and benign malware, and feature engineering on a sample dataset to understand and confirm the effectiveness of various techniques. The final notebook consolidates the insights and techniques derived from previous notebooks to create a high-performing classifier for Android malware. 

Using a diverse set of classifiers, including Decision Trees, XGBoost, LightGBM, RandomForest, and CatBoost, we ensure a comprehensive and robust model for malware detection. We also employ preprocessing techniques such as Standard Scaler for data normalization and SMOTE for addressing class imbalance, enhancing the model's performance and generalization capabilities.

The resulting machine learning model demonstrates the potential for accurate and efficient Android malware classification, providing a valuable tool for securing mobile devices and protecting users from malicious applications. The project's structured and systematic approach enables continuous improvement and adaptability, ensuring the model remains relevant and effective in the ever-evolving landscape of Android malware threats.

## 1.1. INTRODUCTION

The proliferation of mobile devices and applications has led to a surge in Android malware, posing significant security risks to users' personal data and device functionality. To address this challenge, the development of effective and efficient malware detection and classification models is crucial. In this project, we aim to create a machine learning model that accurately classifies Android malware into distinct types and differentiates between malignant and benign samples.

Our approach to the project is systematic and structured, ensuring a comprehensive understanding of the data and a robust model development process. We begin by analyzing the entire dataset, gaining insights into the features and relationships between different malware types and benign samples. This initial analysis allows us to explore the data's structure, characteristics, and statistical properties.

Next, we create a sample dataset to work on a more focused and manageable context. We perform binary classification between malignant and benign malware, serving as a foundational step in the overall malware classification process. Using the sample dataset, we also experiment with various feature engineering techniques, enabling us to understand and confirm their effectiveness before applying them to the entire dataset.

With the insights derived from the sample dataset, we proceed to develop and evaluate machine learning models for classifying Android malware into distinct types. We employ a diverse set of classifiers, including Decision Trees, XGBoost, LightGBM, RandomForest, and CatBoost, ensuring a comprehensive and robust model for malware detection. Preprocessing techniques such as Standard Scaler and SMOTE are also applied to enhance the model's performance and generalization capabilities.

This project report documents the methodology, techniques, and findings of our work, highlighting the potential of machine learning in addressing the critical task of Android malware detection and classification. Through a structured and systematic approach, we demonstrate the effectiveness of our model in securing mobile devices and protecting users from malicious applications, contributing to a safer mobile ecosystem.

## 1.2. ABOUT THE DATA

The Canadian Institute of Cybersecurity created the CIC-AndMal2017, a comprehensive Android Malware Dataset with 10,854 samples, including 4,354 malware and 6,500 benign applications from diverse sources. The data's authenticity was ensured by running applications on real smartphones instead of emulators, due to advanced malware's ability to detect and alter behavior in emulated environments.

The benign applications were collected from the Google Play market, published between 2015 and 2017. The malware samples, on the other hand, were installed on real devices and categorized into four types: Adware, Ransomware, Scareware, and SMS Malware. These samples belong to 42 unique malware families, with a varying number of captured samples for each family.

To obtain a comprehensive view of the malware samples and overcome the stealthiness of advanced malware, three distinct data capturing states were defined: Installation, Before Restart, and After Restart. A specific scenario was created for each malware category to capture the network traffic features (.pcap files) during these states. Using CICFlowMeter-V3, over 80 features were extracted from the network traffic data, providing a rich dataset for the development of machine learning models for Android malware detection and classification.

## 1.3. METHODOLOGY

Our methodology for this project is structured into four main stages, which are detailed in the corresponding notebooks. Each stage plays a crucial role in understanding the data and developing a robust machine learning model for Android malware classification.

1. Initial Data Analysis (Notebook1): The first stage involves an exploratory data analysis of the entire dataset. This analysis provides insights into the data's structure, characteristics, and statistical properties, enabling us to gain a better understanding of the features and relationships between different malware types and benign samples.
2. Binary Classification (Notebook2): In the second stage, we focus on binary classification between malignant and benign malware. We develop and evaluate machine learning models to differentiate between malicious and benign applications, serving as a foundational step in the overall malware classification process.
3. Feature Engineering on Sample Data (Notebook3): The third stage presents our initial work on a sample dataset to understand and confirm the effectiveness of various feature engineering techniques. This stage allows us to experiment with different approaches in a more focused and manageable context before applying them to the entire dataset.
4. Final Model Development and Evaluation (Notebook4): The final stage consolidates the insights and techniques derived from the previous stages to create a robust and accurate machine learning model for Android malware classification. This stage includes feature engineering, resampling, encodings, and model development to obtain the best possible classifier for distinguishing between different types of malignant malware.
   
In the following sections, we will provide a detailed description of each stage, discussing the techniques used, the rationale behind our choices, and the results obtained.

## 1.4. EXPLORATORY DATA ANALYSIS

## 1.4.1. FEATURE DESCRIPTION

Our dataset consists of 1,411,064 entries and 86 columns, providing a comprehensive view of the Android malware landscape. The dataset includes information on various aspects of network traffic and packet characteristics, captured during different states of malware behavior (installation, before restart, and after restart). Among the 86 columns, 75 are of float64 data type, and 11 are of object data type. Some of the key features in the dataset include source and destination IPs, ports, protocols, flow duration, total forward and backward packets, packet length statistics, flow rate measures, and flag counts.

The dataset captures a wide range of network traffic features, offering a rich source of information for developing machine learning models to classify Android malware effectively. The comprehensive feature set allows us to gain insights into the behavior of various malware types and their differences from benign applications, facilitating the development of robust and accurate classification models.

## 1.4.2. DATA CLEANING

The following bar plot illustrates the count of missing values for each column within the dataset. From the visualization, it is evident that the number of NaN values is limited to 12 or fewer for all columns. Subsequently, we investigated the distribution of these NaN values across the rows and discovered that they are not uniformly dispersed. Given the considerable size of the original dataset, which contains 1,411,063 entries, we have chosen to remove the 12 rows containing NaN values. This decision is based on the minimal impact that the removal of such a small number of rows will have on our model's performance and overall results.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AndMal_01.jpg" %}

## 1.4.3. FEATURE VARIABLE EXPLORATION

In this section, we delve into the process of exploring the feature variables of our dataset to better understand their properties and potential impact on our malware classification model. As we examined the columns, we encountered a few issues that required further investigation and clarification. For instance, we noticed that the 'Protocol' column, which was expected to be categorical in nature, contained numerical data. Upon closer examination, we discovered that these numbers represented various protocols such as TCP, UDP, and others, with values 6, 17, and 0 corresponding to each protocol respectively. This assignment of values is based on the standards set by the Internet Assigned Numbers Authority (IANA).

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AndMal_02.jpg" %}

Upon further investigation of the dataset, we discovered two columns with the same name: 'Fwd_Header_Length' and 'Fwd_Header_Length.1'. We found that these columns contained duplicate values, which added redundancy to our analysis. To streamline the process, we removed one of these redundant columns.

Furthermore, we observed that some rows had object data types, while their counterparts had float data types for the same type of data but in the opposite direction of flow. We identified the following columns with object data types: 'Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp', 'Flow_IAT_Min', 'Packet_Length_Std', 'CWE_Flag_Count', 'Down/Up_Ratio', 'Fwd_Avg_Bytes/Bulk', 'Label', and 'Malware_Type'. Utilizing object data types in models can be cumbersome, especially when most columns already have numerical data types. Consequently, we decided to explore these rows further.

We found that the 'Flow_ID' column was simply a combination of 'Destination_IP', 'Source_IP', 'Destination_Port', 'Source_Port', and 'Protocol'. Thus, we removed the 'Flow_ID' column. Additionally, we recognized the potential of source and destination ports to help identify certain types of traffic, such as well-known protocols. Therefore, we changed their column type to float.

Regarding the IP address columns, we realized that the IP address could only help us trace the location if the traffic was originating from a public network. If traffic was originating from a private network, tracing its location using APIs was not possible. Moreover, tracing the location using API calls for such a large dataset was limited. Therefore, we decided to drop the 'Source_IP' and 'Destination_IP' columns. However, we did extract unique destination IP addresses to test the function of tracing back IP addresses and to gain insights into the locations where data from malware devices was being downloaded. The map below depicts the cities included in these destinations.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AndMal_map.jpg" %}

Lastly, we decided to drop the 'Timestamp' column, as it only provided the date and time when the data was collected, which did not pertain to our goal of malware detection. We also noticed that several columns, including 'Flow_IAT_Min', 'Packet_Length_Std', 'CWE_Flag_Count', and 'Down/Up_Ratio', had object data types but were of float type. As a result, we changed the data type of these columns to float.

## 1.4.4. TARGET VARIABLE EXPLORATION

We delved into the analysis of the target variable within the malware classification project. Our initial step was to explore the distribution of Benign and Malignant Malware, the two primary categories under consideration. The Malignant category encompasses four distinct subtypes: Adware, Ransomware, Scareware, and SMSmalware. A bar graph representation of the data revealed that the Malignant category dominates the dataset, accounting for the majority of the observations.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AndMal_map.jpg" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AndMal_03.jpg" %}

To obtain a more comprehensive understanding, we proceeded to subdivide the Malignant category into its four constituent subtypes, aiming to uncover any disparities in their distribution. The resulting graph exhibited a discernible contrast among the various malware types. Adware, Ransomware, and Scareware demonstrated a relatively balanced distribution, while SMSmalware stood out, constituting roughly two-thirds of the combined size of the other three subtypes.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AndMal_04.jpg" %}

In order to delve deeper, we performed an in-depth analysis of each malware type, pinpointing the specific families associated with them. This information proved valuable for visualizing the distribution of malware families within each subtype. Consequently, we generated four distinct figures to effectively represent the family distributions for Adware, Ransomware, Scareware, and SMSmalware respectively.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/AndMal_04.jpg" %}





