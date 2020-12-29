# Machine Learing - Basics


## Overview

+ [Machine learner](../Notes/a12-MLArticles.md#making-sense-of-machine-learner): computer algorithm designed for 
  + pattern recognition
  + curve fitting
  + classification
  + clustering

+ [Type of machine learner](../Notes/a12-MLArticles.md#making-sense-of-machine-learner)
  + supervised methods
  + unsupervised methods
  + time-series methods
    + data collected at many points of time
    + cross-sectional research for marketing
    + utilizing discriminant analysis, regression and factor analysis commonly
  + pattern mining: used for rationalize self placement and for recommender system
  + special methods
    + text analysis
    + social network analysis
    + web analysis
    + mining stream data
    + anomaly detection

+ [Popular machine learner](../Notes/a12-MLArticles.md#making-sense-of-machine-learner)
  + Artificial Neural Network (ANN)
  + Support Vector Machine (SVM)
  + Random Forest
  + Adaboost / boosting


## Decision Tree Classifiers

+ [Decision tree](../Notes/a12-MLArticles.md#decision-tree-classifiers-a-concise-technical-overview)
  + a type of flow chart to assist in the decision making process
  + internal node: tests on particular attributes
  + branches: a single test outcome
  + leaf nodes: class labels

+ [Decision tree classifiers](../Notes/a12-MLArticles.md#decision-tree-classifiers-a-concise-technical-overview)
  + most important classifiers
    + [Iteractive Dichotimiser 3 (ID3)](https://tinyurl.com/jpxx8c3) - Ross Quinlan's precursor to the C4.5
    + [C4.5](https://tinyurl.com/7dr6b28) - one of the [most popular classifiers](https://tinyurl.com/y7slvaad) of all time, also from Quinlan
    + CART - independently invented around the same time as C4.5, also still very popular
  + all adopt a top-down, recursive, divide-and-conquer approach to decision tree induction
  + C4.5: a benchmark against which the performance of newer classification algorithms are often measured
  + main tasks of decision tree classification algorithms
    + tree induction
    + tree pruning

+ [Decision tree induction](../Notes/a12-MLArticles.md#decision-tree-classifiers-a-concise-technical-overview)
  + the process of constructing a decision tree from a set of training data and computations of attribute selection measures
  + taking a set of pre-classified instances as input, deciding which attributes are best to split on, splitting the dataset, and recursing on the resulting split datasets until all training instances are categorized
  + goal
    + split on the attributes which create the purest child node possible
    + minimize the number
  + Decision Tree Induction Algorithm

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick="window.open('https://tinyurl.com/y8kljks4')"
        src    ="https://tinyurl.com/yc9tjzv2"
        alt    ="Decision Tree Induction Algorithm"
        title  ="Decision Tree Induction Algorithm"
      />
    </figure>

+ [Attribute Selection Measures](../Notes/a12-MLArticles.md#decision-tree-classifiers-a-concise-technical-overview)
  + 3 prominent decision tree classifiers.
    + [Information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) - used in the ID3 algorithm
    + [Gain ratio](https://en.wikipedia.org/wiki/Information_gain_ratio) - used in the C4.5 algorithm
    + [Gini index](https://en.wikipedia.org/wiki/Gini_coefficient) - used in the CART algorithm
  + information gain
    + based on information theory from Claude Shannon
    + how much would need to be known about a previously-unseen instance in order for it to be properly classified
    + measured by comparing entropy, or the amount of information needed to classify a single instance of a current dataset partition, to the amount of information to classify a single instance if the current dataset partition were to be further partitioned on a given attribute
    + the required number of information gain comparisons enough to tell us how much is actually gained
      + given attribute
      + the expected reduction in the info requirements caused by knowing the value of attribute
    + max info gain $\to$ attribute upon which to split this partition

+ [Decision Tree Raising](../Notes/a12-MLArticles.md#decision-tree-classifiers-a-concise-technical-overview)
  + completed decision tree model
    + overlay complex
    + containing unnecessary structure
    + difficult to interpret
  + tree pruning: the process of removing the unnecessary structure from a decision tree in order to make it more efficient
  + overfitting
    + models build from algorithms too specifically-tailored to the particular training dataset that was used to generrate them
    + perform poorly on another set of unseeing test data
    + applied to nearly all ML classification algorithm
  + C4.5 pruning methods
    + post-pruning, with subtree raising
    + subtree raising entails raising entire subtrees to replace nodes closer to the root $\to$ reclassifying leaves of subtrees closer to the root which may have been replaced during this process
  + Decision Tree Raising Algorithm

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick="window.open('https://tinyurl.com/y8kljks4')"
        src    ="https://www.kdnuggets.com/wp-content/uploads/dt-raising-algo.jpg"
        alt    ="Decision Tree Raising Algorithm"
        title  ="Decision Tree Raising Algorithm"
      />
    </figure>

  + procedure of decision tree
    1. build tree via induction
    2. solidified tree w/ pruning
    3. classifying data

+ [Decision tree](../Notes/a12-MLArticles.md#decision-tree-classifiers-a-concise-technical-overview)
  + classification strategy as oppose to some single well-known classification algorithm
  + any algorithms seeks to classify data, and take a top-down, recursive. divide-conquer approach to crafting a tree-based graph for subsequent instance classification, regardless of any other particulars (including attribution split selection methods and optional tree-pruning approach) would be consider a decision tree


## Clustering Techniques

+ [Clustering](../Notes/a12-MLArticles.md#comparing-clustering-techniques-a-concise-technical-overview)
  + used fro analyzing data not including pre-loaded classes
  + data distance: grouped together using the concept of maximizing inter-class similarity and minimizing the similarity btw differing classes
  + techniques
    + k-means clustering
    + hierarchical clustering
    + fuzzy clustering
    + density clustering
    + centroid-style clustering

+ K-means clustering algorithm

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick="window.open('https://www.kdnuggets.com/2016/09/comparing-clustering-techniques-concise-technical-overview.html')"
        src    ="https://www.kdnuggets.com/wp-content/uploads/kmeans-pseudocode.jpg"
        alt    ="k-means Clustering Algorithm"
        title  ="k-means Clustering Algorithm"
      />
    </figure>

  + converge: the re-calculated centroids match the previous iteration's centroids or are within some present margin
  + Euclidean distance: one measure of distance

    \[ \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} \]

  + iterative clustering k-means in serial, however, the distance calculations within an iteration need not be

+ Expectation-Maximization (EM)
  + probabilistic clustering: determine the most likely set of cluster, given a set of data
  + EM: a probabilistic clustering algorithm
  + 2 steps
    + __Expectation Step (E-Step):__ particular objects to clusters based on parameter $\to$ cluster probability calculation step the cluster probabilities being the 'expected' class values
    + __Maximization Step (M-Step):__ calculate the distribution parameters, maximizing expected likelihood
  + parameter estimation equations: cluster probabilities known for each cluster
  + mean for a cluster

    \[ \mu_c = \frac{w_1x_1 = w_2x_2 + \cdots + w_nx_n}{w_1 + w_2 + \cdots + w_n} \]

  + standard deviation of a cluster

    \[ \sigma_c = \frac{w_1(x_1 - \mu)^2 + w_2(x_2 - \mu)^2 + \cdots + w_n(x_n - \mu)^2}{w_1 + w_2 + \cdots + w_n} \]

    + $w_i$: probability of an instance $i$ as a member of a cluster $c$
    + $x_1$: all of the dataset's instances



