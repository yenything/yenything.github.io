---
layout: post
title: "Spotify Song Popularity Prediction"
date: 2023-02-20T16:49:03Z
categories: ["R", "EAD", "Regression"]
description: "The aim of this initiative is to develop a forecasting algorithm that can estimate the popularity of a song by utilizing other variables in the song data."
thumbnail: "/assets/images/gen/content/spotify.png"
image: "/assets/images/gen/content/spotify_white.png"
---
# Spotify Song Popularity Prediction

The aim of this initiative is to develop a forecasting algorithm that can estimate the popularity of a song by utilizing other variables in the song data. The dataset contains 15 columns and 18,835 rows. Since the initial column denotes song names, it will be eliminated. Additionally, we will filter the dataset rows with song popularity values ranging from 0 to 100, as popularity scores of zero are not common, and values over 100 are considered outliers in this dataset. Consequently, the refined dataset will have 14 columns and 18,563 rows.

## 1. Exploratory Data Analysis

Our exploration will focus on the cleaned dataset, which contains fourteen columns divided into two variable types: numerical and categorical. We will examine the numerical variables by analyzing the shape of the histogram and calculating the mean and median values. As for the categorical variables, we will explore the frequency of each category, but we won't calculate the mean and median values.

### 1.1. EDA on Song Popularity
The histogram presented illustrates the distribution of song popularity values in a cleaned dataset. The popularity values range from 1 to 100 and follow a normal distribution. The box plot indicates that 50% of the data falls within the interquartile range, which spans from 40 to 70.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-111.jpg" %}

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-12.jpg" %}

### 1.2. EDA on the numerical variables

The graphs below illustrate that the columns 'song_duration_ms', 'tempo', and 'audio_valence' have similar mean and median values, indicating that they are normally distributed. Conversely, the histograms of 'acousticness', 'instrumentalness', 'liveness', and 'speechiness' show mean values greater than the medians, indicating that these columns are right-skewed. In contrast, 'danceability', 'energy', and 'loudness' are left-skewed as their mean values are less than their medians, resulting in tails on the left side.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-20.jpg" %}

### 1.3. EDA on the categorical variables

The categorical variables in this dataset include key, audio_mode, and time signature. The category with the highest frequency occurs when the key is 0, audio_mode is 1, and time_signature is 4. If we want to build a more reliable regression model, we can consider including the key and audio_mode variables, but we should avoid using the time_signature variable since the majority of the data points are for time_signature = 4, which indicates bias.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-21.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-22.jpg" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-23.jpg" %}

### 1.4. Correlations and scatter plots

The variables that have the highest correlation with song_popularity are instrumentalness (-0.1299), danceability (0.1131), and loudness (0.1060). However, even when taking into account the categorical variables of key and audio_mode, it is difficult to discern any clear patterns from the scatter plots of these variables with song_popularity.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-30.jpg" %}

The variables 'energy and loudness' (0.7563), 'acousticness and energy' (-0.6651), and 'acousticness and loudness' (-0.5586) exhibit the strongest correlations among the x variables. Since these three variables have strong correlations with each other, it may be useful to combine them when creating the regression model.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-40.jpg" %}

## 2. Simple Regression

We will compare the naïve model's error of 21.15945 for song popularity with the residuals of simple regression models. If the RMSEs of the simple regression models are less than the naïve model's error, we will consider using the variables in our final regression model.

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


To improve the regression model for song popularity, we compared the RMSE values of simple regression models with the naïve model's error of 21.15945. All the RMSE values were found to be greater than the naïve model's error, indicating the need to use at least one independent variable in the regression model. Interestingly, the variables with the top three lowest RMSE values were danceability, instrumentalness, and loudness, which also happen to be the top three variables with the strongest correlations with song popularity. Thus, we can use these three variables in the regression model, resulting in a residual standard error of 20.86 and improved performance compared to the previous simple models. Therefore, we can confidently include instrumentalness, danceability, and loudness in the final regression model for song popularity.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_1.png" %}

The adjusted R-squared value suggests that only about 2.8% of the variability in song popularity can be accounted for by the independent variables of instrumentalness, danceability, and loudness. Therefore, it may be necessary to apply some transformation to the input variables to enhance the performance of the model.

## 3. Variable transformations attempted
### 3.1. Transforming the variable loudness
To take advantage of the strong correlations between loudness, energy, and acousticness, we can transform the loudness variable into three new variables: (1) loudness multiplied by energy, (2) loudness multiplied by acousticness, and (3) loudness multiplied by both energy and acousticness. These transformed variables can then be included in the regression model to see if they improve the predictive power of the model.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_21.png" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_22.png" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_23.png" %}

After transforming the variable loudness, the RMSE values decreased and the adjusted R-squared values increased compared to the model with non-transformed loudness. Therefore, the model with (loudness * energy * acousticness) was chosen instead of using the original loudness variable for the regression model, since it had the lowest RMSE value of 20.54 and the greatest adjusted R-squared value of 0.06057 among the three models considered.

### 3.2. Transforming the variable instrumentalness:
When the power of the instrumentalness variable is decreased, the RMSE value decreases and the adjusted R-squared value increases. However, when the power is set to a value less than 1/6, the RMSE value starts to increase again.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_31.png" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_32.png" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_33.png" %}

### 3.3. Not transforming the variable danceability:
The transformation of the variable danceability does not improve the RMSE and adjusted R-square values of the regression model or leads to a decrease in these values. Therefore, we exclude the transformed variable from the regression model.

## 4. Final regression model.

The final regression model for predicting the song popularity is:

`song popularity = -10.4188 × √(6&instrumentalness) + 8.6063 × danceability + 2.9404 × loudness - 38.8873 × energy - 41.3020 × acousticness - 2.4798 × loudness × energy - 3.6369 × loudness × acousticness + 45.4606 × energy × acousticness + 5.1890 × loudness × energy × acousticness + 88.9532`

The feature engineering techniques used in the model are transforming the variable instrumentalness to √(6&instrumentalness) and creating a feature interaction for the variable loudness with the energy and acousticness. These techniques lower the RMSE of the model to 20.42 and increase the adjusted R-squared to 0.06852.

### 4.1. Summary of the model fit and comparisons

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/r_result_4.png" %}

In our model, the RMSE of train and test data are 20.42, 20.22 respectively, while the RMSE of the naïve model is 21.15

### 4.2. Description of parameters signs

A parameter estimate (coefficient) signifies the change in the predictor's value, while keeping all other predictors constant. The coefficient reflects how much the predictor contributes; a high coefficient indicates that it strongly affects the likelihood of an outcome, whereas a coefficient near zero suggests that it has little impact on it. Positive coefficients imply that an explanatory variable increases the chance of an outcome, whereas negative coefficients suggest a decrease.

In our model, the coefficients of energy and acousticness are -38.8873 and -41.3020, respectively, indicating that they decrease the probability of the outcome. However, after feature engineering, the combined coefficient of energy and acousticness is 45.4606, which is a high positive value. This suggests that the combination of these variables will increase the likelihood of the outcome. Similarly, the coefficient of loudness is 2.9404, but the combined coefficient of loudness, energy, and acousticness is 5.1890, which will increase the probability of the outcome.

### 4.3. Analysis of residuals

The section summarizes the residuals, which represent the difference between the predicted values of the model and the actual results. Smaller residuals are indicative of better model performance.

When the residuals of the regression model are plotted against the non-transformed variables (instrumentalness, danceability, and loudness) and the transformed variables (√(6&instrumentalness), danceability, and (loudness * energy * acousticness)), they appear to be more normally distributed when the variables are transformed.

As can be seen in the first two plots, the bump in the right corner disappears after transforming the instrumentalness variable. Additionally, in the last two plots, the residuals are more evenly spread out horizontally after the transformation. Although the danceability variable was not transformed in the regression model, the dots that were previously lined up at the bottom of the third plot (regression model with non-transformed variables) become more spread out in the fourth plot (regression model with transformed variables instrumentalness and loudness).

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-50.jpg" %}

### 4.4. Sensitivity analysis

Sensitivity analysis is a process that quantifies the relationship between the uncertainty in a model's output and the uncertainty in its inputs. It helps to determine the factors that have the greatest impact on the model's output and allows us to focus on the most important aspects. This can save time, reduce frustration, and increase efficiency.

The bars in the plot are ordered from bottom to top as follows: instrumentalness_sens, danceability_sens, loudness_sens, energy_sens, acousticness_sens, loudness:energy_sens, loudness:acousticness_sens, energy:acousticness_sens, and loudness:energy:acousticness_sens.

From the graph, we can see that fluctuations in loudness, energy, and acousticness are the most influential factors that contribute to uncertain output values.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-60.jpg" %}

### 4.5.	Graphs of fit (Predicted vs. actual).
The majority of the data points are not located close to the fitted line, indicating that the current model may not be the best fit for the data. To improve the fit, we may need to consider using a different type of model or incorporating additional variables that better explain the variation in the data. Alternatively, we could explore the possibility of transforming the existing variables to better capture the underlying patterns in the data. It's important to note that points located further away from the mean or vertically distant from the line may have a significant impact on the fit of the model, and should be carefully examined to determine if they are genuine outliers or simply part of the underlying variability in the data.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/spotify-70.jpg" %}

