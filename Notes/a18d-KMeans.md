# Clustering with K-Means

Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/clustering-with-k-means)

[Local notebook](src/a18c-creating-features.ipynb.ipynb)


## Introduction

+ Unsupervised learning algorithms
  + not making use of a target
  + purpose:
    + learning some property of the data
    + representing the structure of the features in a certain way
  + a "feature discovery" technique in terms of feature engineering

+ Clustering
  + the assigning of data points to groups
  + group based on how similar the points to each other
  + making "birds of a feather flock together"
  + used for feature engineering: an attempt to discover
    + groups of customers representing a market segment
    + geographic areas sharing similar weather patterns
  + adding a feather of cluster labels $\to$ untangle complicated relationships of space and proximity


## Cluster Labels as a Feature

+ Feature w/ clustered labels
  + clustering: like a traditional "binning" or "[discretization](https://bit.ly/2PBQ1VB)" transform
  + multiple features:
    + a.k.a. vector quantization
    + multi-dimensional binning
  + motivation for adding cluster labels
    + clusters breaking up complicated relationships across features in simple chunks
    + applying divided and conquer strategy to handle different clusters
    + learning the simpler chunks one-by-one instead learning the complicated one
  + example: clustering w/ single and two features

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/clustering-with-k-means')"
        src    = "https://i.imgur.com/sr3pdYI.png"
        alt    = "Left: Clustering a single feature. Right: Clustering across two features."
        title  = "Left: Clustering a single feature. Right: Clustering across two features."
      />
    </figure>

  + example: a new feature of cluster labels
    + `Cluster`: categorical variable w/ a label encoding
    + typically produced w/ a clustering algorithm
    + depending on the model, one-hot encoding probably more appropriate

    <table align=center>
    <thead>
    <tr>
      <th>Longitude</th><th>Latitude</th><th>Cluster</th>
    </tr>
    </thead>
    <tbody>
      <tr style="text-align: center;"><td>-93.619</td><td>42.054</td><td>3</td></tr>
      <tr style="text-align: center;"><td>-93.619</td><td>42.053</td><td>3</td></tr>
      <tr style="text-align: center;"><td>-93.638</td><td>42.060</td><td>1</td></tr>
      <tr style="text-align: center;"><td>-93.602</td><td>41.988</td><td>0</td></tr>
    </tbody>
    </table>
  
  + example: relationship btw `YearBuilt` and `SalePrice`
    + curved relationship: too complicated $\to$ underfit
    + smaller chunks $\to$ almost linear relationship $\to$ easier to learn

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/clustering-with-k-means')"
        src    = "https://i.imgur.com/rraXFed.png"
        alt    = "Clustering the YearBuilt feature helps this linear model learn its relationship to SalePrice."
        title  = "Clustering the YearBuilt feature helps this linear model learn its relationship to SalePrice."
      />
    </figure>



## k-Means Clustering

+ Clustering algorithms
  + classification
    + how they measure "similarity" or "proximity"
    + what kinds of features working with
  + k-means: intuitive and easy to apply in a feature engineering context
  + selection of algorithm: depending on application

+ K-means clustering
  + measuring similarity using ordinary straight-line distance (Euclidean distance)
  + creating clusters by placing a number of points, called centroids, inside the feature space
  + each point assigning to the cluster whatever centroid it closest to
  + $k$: the parameter about how many centroids
  + Voronoi tessallation
    + imaging each centroid capturing points through a sequence of radiating circles
    + a line formed w/ the overlapped sets of circles from competing centroids
    + analogy: which cluster to assigned w/ future data

+ K-means w/ scikit-learn's implementation
  + hyperparameters: `n_clusters`, `max_iter`, `n_init`
  + procedure
    + init: randomly initializing `n_cluster` centroids
    + assign points to the nearest cluster centroid
    + move each centroid to minimize the distance to its points
    + repeat the above 2 steps until the centroids converged or reaching the maximum iteration (`max_iter`)
  + issue: initial random position of the centroids $\to$ poor clustering
  + solution:
    + repeat the algorithm a number of times (`n_init`)
    + return the clustering w/ the least total distance btw each point and its centroid, the optimal clustering
  + increasing the `max_iter` for a large number of clusters
  + increasing `n_init` for a complex dataset
  + best partitioning for a set of features depending on
    + model used
    + what to predict




## Example - California Housing





## Exercise






