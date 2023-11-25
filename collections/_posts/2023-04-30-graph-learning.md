---
layout: post
title: "Graph Classification Problem"
date: 2023-04-30T09:49:03Z
categories: ["Graph", "Classification"]
description: Given a graph dataset, we aim to predict the label of the nodes in the graph. Hence, this is a classification problem.
thumbnail: "/assets/images/gen/content/twitter_logo.jpg"
---

### 1. Project Goal  
The goal of this project is to develop a classification model to predict the label of nodes in a given graph dataset. This is a competition-based project where different groups will compete against each other, and the performance of the final submission will determine the grades for the project. The focus will be on developing a robust and accurate model that can accurately classify the nodes in the graph, and the success of the project will depend on the ability to build a model that performs well on the test dataset.  
  
### 2. Data Description  

The given graph dataset consists of 2,480 nodes and 10,100 edges, with 7 possible classes and 1,390 features. The features for each node are stored in the "features.npy" file, while the labels for each node are stored in the "labels.npy" file. The adjacency matrix for the graph is stored in the "adj.npz" file.
The data is split into a training set of 496 nodes and a test set of 1984 nodes, with the splits defined in the "splits.json" file.

### 3. Workflow
#### a. Data Preparation
The code starts by importing the necessary libraries, including numpy and PyTorch Geometric. It assumes that the following variables have already been defined:
  
* splits: a Python dictionary containing the data splits (train and test)
* feat: a NumPy array containing the node features
* adj: a sparse NumPy array containing the adjacency matrix
* labels: a NumPy array containing the node labels
  
First, we create a PyTorch geometric Data object to store our graph data. We start by converting the feature matrix (feat) to a PyTorch tensor and assign it to the "x" attribute of our Data object. We then initialize a tensor of zeros called "y" to store the labels for each node in the graph. The tensor has a shape of (number of nodes in the graph, 1) and a data type of "long".
  
We assign the training set labels to the corresponding nodes using the "idx_train" indices, and convert the labels to a PyTorch tensor with dtype "long". We assign this tensor to the "y" attribute of our Data object.
  
We create the edge index for the graph using the adjacency matrix (adj). We first convert the nonzero indices of the adjacency matrix to a PyTorch tensor using the "nonzero()" method, and assign this to the "edge_index" attribute of our Data object.
  
We then create boolean masks to indicate which nodes are in the training and test sets. We initialize these masks with zeros and set the corresponding nodes to True using the "idx_train" and "idx_test" indices.
  
Finally, we assign the feature matrix, edge index, label tensor, and masks to our Data object, and set the "num_classes" attribute to the number of unique labels in our dataset. This resulting PyTorch geometric Data object can be used as input for various machine learning models, allowing for the prediction of node labels in the graph.
  
#### b. Baseline Model

To classify graph data, we implement a Graph Convolutional Network (GCN) architecture using the PyTorch Geometric library. The GCN consists of two GCNConv layers with sigmoid activation and dropout functions, followed by a final GCNConv layer with a softmax activation function. We use this architecture to produce a probability distribution over the output classes for the input graph data. The sigmoid activation and dropout functions are used to prevent overfitting to the training data.
  
In our experimentation with different optimization algorithms and their hyperparameters, we test Adam, SGD, and RMSprop. Among the six models we evaluate, we find that RMSprop with a learning rate of 0.01, an alpha value of 0.8, and a weight decay of 0.0005 achieves the highest accuracy of 0.9819.
  
| Optimizer | Learning Rate | Momentum/Alpha | Weight Decay | Accuracy |
|-----------|--------------|----------------|--------------|----------|
| Adam      | 0.001        | N/A            | 0.001        | 0.9677   |
| SGD       | 0.01         | 0.9            | 0.0001       | 0.2923   |
| Adam      | 0.001        | N/A            | 0.005        | 0.2923   |
| RMSprop   | 0.01         | 0.9            | 0.0005       | 0.9718   |
| RMSprop   | 0.01         | 0.8            | 0.0005       | 0.9819   |
| RMSprop   | 0.1          | 0.5            | 0.001        | 0.5020   |

#### c. Hyperparameter tuning
To find the best hyperparameters, we use ‘itertools’  to generate all possible combinations of a set of hyperparameters. The set of hyperparameters that we use are the following:

* 'num_hidden': [16, 32, 64, 128],

* ‘lr’: [0.003, 0.002, 0.001, 0.04, 0.03, 0.02, 0.01, 0.1],

* 'weight_decay': [0.0005, 0.0001, 0.001, 0.01]  
  
The code below performs a hyperparameter search for the GCN model on the given graph dataset. For each combination, it trains a GCN model with the specified hyperparameters using the train split of the dataset. The model with the highest accuracy on the train set is selected as the best model, and its hyperparameters are reported. Finally, the best model is evaluated on the train set and its accuracy is reported. This code is useful for finding the best hyperparameters to use when training a GCN on the given graph dataset.

### 4. Results and Discussions
After completing the tuning process, we observe that the best accuracy is 0.9959, and we use the hyperparameters {'num_hidden': 64, 'lr': 0.04, 'weight_decay': 0.0001} to achieve this accuracy. Moreover, we notice that by using these hyperparameters, our model achieves a train accuracy of 0.9959. This accuracy is better than our baseline model, which has a training accuracy of 0.9819.
Our model achieves a train accuracy of 0.9940, indicating that it performs well on the training set. To evaluate its performance on new, unseen data, we also measure the test accuracy, which we obtain to be 0.847278.

### 5. The learned lessons and Further Improvements
During our project, we experimented with various classifiers like GAT, GraphSAGE, and GCN. Our findings showed that GCN outperformed the other classifiers. To further improve our results, we could implement cross-validation and ensemble models to ensure our model is robust and can generalize well on unseen data. In addition, we could use other hyperparameter tuning techniques such as Bayesian Optimization to fine-tune our model's performance. By implementing these improvements, we can continue to learn and enhance our machine learning models to produce better results.

### 6. Code
You can reach the project here: [code](https://github.com/yenything/CSE881_DataMining/blob/main/Code.ipynb)
