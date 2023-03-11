---
layout: post
title: "World Cities - Cost of Living"
date: 2023-01-20
categories: ["Clustering", "Python", "Streamlit"]
description: The project aims to cluster world cities based on living costs and explore the characteristics of each cluster while observing their geographic location on a world map.
thumbnail: "/assets/images/gen/content/cities_0.png"
image: "/assets/images/gen/content/cities_0.png"
---

{% include framework/shortcodes/youtube.html id='wcBpxVWDv0s' %}

You can reach the application here: [Web App](https://yenything-cmse830-ml-project-cmse-ml-prj-rws7mg.streamlit.app/)


# 1. Introduction
## 1.1. Introduction
The goal of this project is to apply unsupervised learning to cluster the world cities based on their living costs. With the optimal number of clusters discovered in this web application, we will label each city and explore the characteristics of each cluster. Also, we will locate each cluster on the world map with different colors to observe whether the results of clustering are related to geographic location.

## 1.2. Dataset
The original data set consists of country names, city names, and categories for cost of living. I added continent names, latitude, and longitude on the data set to locate each of city on the world map.

# 2. Methodology
As you see in the video, there are more than 60 columns in the dataset. In the real world, it is impossible to make a 60-dimensional graph. To understand the K-mean clustering method which we are going to explore in this web app, we will use two columns instead to visualize them on a 2-dimensional graph.

Let's see how we get the optimal number of clusters and what the clustering result looks like!

## 2.1. Scatter Plot

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_1.png" %}

Can you identify any clusters in the graph above?

## 2.2. Optimal Number of Clusters

I've tried the column 'Meal, Inexpensive Restaurant' on the X-axis and 'Bottle of Wine (Mid-Range)' on the Y-axis. And, to be honest, I cannot see any good clusters on the graph. Then, how do we know how many clusters we need for unsupervised learning? The answer is...

Take a look at these elbow curve and silhouette score!

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_2.png" %}

## 2.3. K-means Clustering Result

We checked that the optimal number of clusters is 3. When we set the slider to 3, the data points are categorized into the three clusters which share certain similarities.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_3.png" %}

# 3. Unsupervised Learning

There is no scatter plot this time because we are going to use the whole dataset with 60 columns. But, don't worry! We will follow the exactly same steps what we've just did in the '2.Methodology' tab.

## 3.1. Optimal Number of Clusters
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_4.png" %}

The optimal number of clusters is 4.

## 3.2. K-means Clustering Result
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_5.png" %}

We can generate four clusters when k equals 4. Each cluster's name is 0, 1, 2, and 3. The interesting fact I have found from the table above is that the cluster label 2 has only one data point. It turned out to be Singapore, which means that Singapore has unique characteristics different from the other three clusters.

Let's find out the characteristics of each cluster.

{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_61.png" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_62.png" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_63.png" %}
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_64.png" %}

# 4. Geolocation & Findings
## 4.1. Geolocation
{% include framework/shortcodes/figure.html src="/assets/images/gen/content/cities_0.png" %}
Label 0: Yellow | Label 1: Blue | Label 2: Green | Label 3: Orange | Label 4: Purple 

## 4.2. Findings

[Cluster: Label 0]

** The cluster of the absolute low living cost if you don't have a mortgage loan.**
- This cluster is known for its incredibly low cost of living, especially for those who do not have a mortgage loan. In fact, the average cost of living in this cluster is the lowest among all the clusters, except for the category of 'Mortgage Interest Rate', which is the highest in comparison to the other clusters. However, if you're not burdened with a mortgage, you can enjoy a very affordable lifestyle in this cluster.

[Cluster: Label 1]
** The cluster of the relative low living cost. **
- For those looking for a relatively low cost of living, this cluster is a great option. The average cost of living in this cluster is similar to that of cluster label 3, but what sets it apart is the 'Average Monthly Net Salary', which is twice as much as the latter. This means that you can enjoy a comfortable lifestyle without breaking the bank.

[Cluster: Label 2]
** The cluster of the absolute high living cost. **
- If money is not an issue and you're looking for the highest standard of living, then this cluster is the one for you. It is the most expensive cluster of them all, with the average cost of living in most categories being higher than the other clusters. In particular, it is worth noting that Singapore is the most expensive city in the world to buy a car, with cars costing approximately four times more than they do in the other clusters. So, if you're looking for luxury and can afford it, this is the perfect cluster for you.

[Cluster: Label 3]
** The cluster of the relative high living cost. **
- This cluster is similar to cluster label 1 in terms of the average cost of living. However, what sets it apart is the 'Average Monthly Net Salary', which is twice as small as that of cluster label 1. This means that living in this cluster may require a bit more budgeting and financial planning to maintain a comfortable standard of living. However, if you are willing to be a bit more frugal, you can still enjoy a relatively comfortable lifestyle in this cluster.






