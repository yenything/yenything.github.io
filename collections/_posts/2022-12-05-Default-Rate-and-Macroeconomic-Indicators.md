---
layout: post
title: "Default Rate and Macroeconomic Indicators"
date: 2022-12-05
categories: ["EDA", "time-series", "Streamlit"]
description: I will guide financial institutions in managing their credit risk (the risk of default on a debt that may arise from a borrower failing to pay their loan) during future crises by focusing on select indices from among the many variables available.
thumbnail: "/assets/images/gen/content/20221205_0.png"
---

{% include framework/shortcodes/youtube.html id='WOGxT7hkpKU' %}

You can reach the application here: 🌐[Web App]([https://yenything-cmse830-ml-project-cmse-ml-prj-rws7mg.streamlit.app/](https://yenything-cmse830-datascience-project-cmse-hw6-fldvbh.streamlit.app/)


# 1. Introduction
The main goal of this project is to identify the most affected macroeconomic indices during past global economic recessions. In this project, I will analyze the relationship between the historical default rate in the United States and various macroeconomic indices, and select the top three indices that should be carefully monitored in the future. I anticipate this project to be helpful to companies (e.g., banks) in the financial industry. The project will allow the banks to manage their credit risk, the risk of default on a debt that may arise from a borrower failing to pay their loan, for the upcoming crisis by focusing on only selective indices among myriads of variables.  

# 2. Dataset
## 2.1. Description
The dataset used in this project consists of time-series data from 2002 to 2022, including two economic recessions: the Great Recession in 2008 and the COVID-19 recession in 2020. It includes both annual default rates and monthly macroeconomic indices. The type of recessions and the corresponding time periods were identified using data from the National Bureau of Economic Research, while the macroeconomic indices were obtained from Kaggle. The annual default rates used in this project were sourced from the S&P Global Ratings report. However, to derive more relevant indices, it is recommended that users utilize their own company's default rates instead of relying solely on data from S&P Global.

## 2.2. Sources
The dataset used in this project was obtained from various sources, resulting in differences in the measurement periods or announcement times among the indices. For instance, the Consumer Price Index (CPI) is released monthly by the U.S. Bureau of Labor Statistics, whereas the Gross Domestic Product Index (GDP) is based on quarterly measurements according to the Bureau of Economic Analysis. Moreover, the S&P Global Ratings report provides an annual default rate. As a result, missing data is inevitable, given that each row represents a single month.

## 2.3. Missingness
I classified the missingness type in this dataset as MNAR (Missing Not at Random), as the reason for the missingness is apparent - the release schedule for each index varies. There are two possible solutions to address the missingness: either discard all rows with missing values or impute new values for the missing elements. Since the dataset spans 20 years, dropping rows with missing values would result in a dataset with only 20 rows, which is insufficient for analysis. Therefore, to maintain the original dataset's number of rows (241), I replaced the missing values by adding values of the gap between years divided by 12 months. For example, if the default value for 2021 is 1% and for 2022 is 1.12%, I added 0.01% for each month of 2021, such as Jan 2021: 1%, Feb 2021: 1.01%, Mar 2021: 1.02%, and so on.

# 3. Web Application
## 3.1. The Value of App: Enhancing Financial Decision Making
This project holds significant value despite the need to impute some values in the dataset and the use of a common analyzing methodology. The Basel Committee on Banking Supervision mandates stress testing for banks, requiring them to estimate credit risk by considering economic recession scenarios and prepare buffers accordingly. To create models for predicting default rates, banks mainly rely on macroeconomic variables. By automating and visualizing the procedures of collecting macroeconomic variables and conducting correlation analysis, my project can help banks make informed decisions on macroeconomic indicators. This will be a valuable tool for banks in meeting regulatory requirements and managing credit risk.

## 3.2. Exploring the App

> Step 1 - Explore historical trend of economic indicators and default rate of companies in the United States.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/20221205_1.png" %}
You can select any index from the dropdown menu to see its time-series trend. However, since the units of each index vary, it may be difficult to compare patterns if the values are relatively small. Therefore, I have scaled the dataset to maintain the fluctuation of the lines regardless of the variables you choose.
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/20221205_2.png" %}
After scaling the data, you can see the trend of the lines. In this graph, when you focus on the gray boxes representing the periods of recession over the past two decades, you can see that some indices skyrocketed or suddenly dropped.

Here are my findings:
1.	The default rate increased right after the recessions.
2.	During the recessions, stock prices (S&P and NASDAQ) and GDP decreased, but eventually, they rose again.
3.	The consumer price index (CPI) and the producers' purchase index (PPI) peaked slightly during the Great Recession, but there were no significant changes during the COVID-19 recession.
4.	The mortgage rate and the corporate bond yield rate fluctuated significantly, and they moved in the same direction.
5.	The pattern of unemployment rate is similar to that of the default rate.
6.	The pattern of inflation rate is similar to that of the import price index.
7.	Disposable income soared right after COVID-19.

> Step 2 - Analyze the relationship between the default rate and various indicators.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/20221205_0.png" %}
This heatmap shows the correlation between the default rate and indices. In the first column of the heatmap, we can observe that unemployment and the corporate bond yield rate have a positive correlation with the default rate.

> Step 3 - Select the top three indicators that might be carefully monitored in the future

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/20221205_3.png" %}
These three correlation graphs automatically plot the correlations between the default rate and the top three indices with the highest correlations. What I've found is that the inflation rate, unemployment rate, and import price index show the highest correlation with the default rate, whether it is a recession period or not.

