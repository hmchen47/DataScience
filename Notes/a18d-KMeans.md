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
  + sensitive to scale:
    + rescale or normalize data w/ extreme values
    + depending on domain knowledge and predicting target
    + rule of thumb: feature
      + ready directly comparable, e.g., test result at different time $\to$ not rescale
      + not on comparable scales, e.g., height and weight $\to$ usually benefit for rescale
      + not clear $\to$ use common sense
    + features w/ larger values weighted more heavily
    + comparing different schemes through cross-validation probably helpful
    + examples:
      + `Latitude` and `Longitude` of cities in California: rescaling distort the natural distances $\to$ not rescaling
      + `Lot Area` and `Living Area` of houses in Ames, Iowa: not clear, living area tending to be more valuable while sale price as the prediction target
      + `Number of Door` and `Housepower` of a 1989 model car: rescaling, he number of doors in a car will be negligible comparing to its horsepower (usually in the hundreds)
  + best partitioning for a set of features depending on
    + model used
    + what to predict

+ Examples: SalePrice of Ames and Airbnb in NYC

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://www.kaggle.com/ryanholbrook/clustering-with-k-means" ismap target="_blank">
      <img style="margin: 0.1em;" height=250
        src   = "https://i.imgur.com/KSoLd3o.jpg.png"
        alt   = "K-means clustering creates a Voronoi tessallation of the feature space."
        title = "K-means clustering creates a Voronoi tessallation of the feature space."
      >
      <img style="margin: 0.1em;" height=250
        src   = "https://i.imgur.com/tBkCqXJ.gif"
        alt   = "The K-means clustering algorithm on Airbnb rentals in NYC.s"
        title = "The K-means clustering algorithm on Airbnb rentals in NYC.s"
      >
    </a>
  </div>


## Example - California Housing

+ Example: California housing
  + data set: [California Housing](https://www.kaggle.com/camnugent/california-housing-prices)
    + `Latitude` and `Longitude`: natural candidates for k-means clustering
    + `MedInc`: creating economic segments in different regions of California
  + training w/ K-means

    ```python
    # Create cluster feature
    kmeans = KMeans(n_clusters=6)
    X["Cluster"] = kmeans.fit_predict(X)
    X["Cluster"] = X["Cluster"].astype("category")
    #   MedInc  Latitude  Longitude Cluster
    # 0 8.3252  37.88     -122.23   1
    #    ...
    ```

  + geographic plotting 
    + observing the geographic distribution of the clusters
    + separating segments for high-income areas on the coasts

    ```python
    sns.relplot(
        x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
    );
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/clustering-with-k-means')"
        src    = "https://bit.ly/2PH3XxA"
        alt    = "Geographic clusters"
        title  = "Geographic clusters"
      />
    </figure>

  + median house values and box-plots
    + box-plot: the distribution of the target within each cluster
    + informative clustering $\to$ distributions separating across `MedHouseVal`

    ```python
    X["MedHouseVal"] = df["MedHouseVal"]
    sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 10vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/clustering-with-k-means')"
        src    = "https://bit.ly/3nCnKLd"
        alt    = "Distributions of clusters w/ box-plot"
        title  = "Distributions of clusters w/ box-plot"
      />
    </figure>



## Exercise

+ Exercise: California housings
  + [original exercise](https://www.kaggle.com/hmchen47/exercise-clustering-with-k-means/edit)
  + dataset: [Ames data set](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
  + loading data and utility for cross-validation

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor

    # Set Matplotlib defaults
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )


    def score_dataset(X, y, model=XGBRegressor()):
        # Label encoding for categoricals
        for colname in X.select_dtypes(["category", "object"]):
            X[colname], _ = X[colname].factorize()
        # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
        score = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_squared_log_error",
        )
        score = -1 * score.mean()
        score = np.sqrt(score)
        return score

    # Prepare data
    df = pd.read_csv("data/a18/ames.csv")
    ```




