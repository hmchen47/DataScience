# Principal Components Analysis


Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/principal-component-analysis)

[Local notebook](src/a18e-principal-component-analysis.ipynb)


## Introduction

+ Principal Component Analysis and feature engineering
  + a partitioning of the variation in the data
  + a great tool to help to discover important relationship in the data
  + used to create more informative features
  + typically applied to [standardized](https://bit.ly/2S9yMM2) data
  + variation meaning
    + standardized data: correlation
    + unstandardized data: covariance


## Principal Component Analysis

+ Visualization for Principal Component Analysis
  + axes of variation
    + describing the ways the abalone tend to different from one another
    + axes: perpendicular lines along the natural dimensions of the data
    + each axis for one original feature
  + idea of PCA: instead of describing the data w/ the original features, describing it w/ axes of variation
  + dataset: [Abalone data set](https://www.kaggle.com/rodolfomendes/abalone-dataset)
    + physical measurements taken from several thousand Tasmanian abalone
    + only focusing on `Height` and `Diameter` of their shells
  + axes of variation for abalone
    + Size component
      + the longer axis
      + small height and small diameter (lower left) contrasted w/ large height and large diameter (upper right)
    + Shape component
      + the shorter axis
      + small height and large diameter (flat shape) contrasted w/ large height and small diameter (round shape)

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://www.kaggle.com/ryanholbrook/principal-component-analysis" ismap target="_blank">
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/rr8NCDy.png"
          alt   = "Axes of variation with abalone"
          title = "Axes of variation with abalone"
        >
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/XQlRD1q.png"
          alt   = "The principal components become the new features by a rotation of the dataset in the feature space."
          title = "The principal components become the new features by a rotation of the dataset in the feature space."
        >
      </a>
    </div>

+ PCA as new features
  + new features PCA: liner combinations (weighted sums) of the original features

    <code> df["Size"] = 0.707 * X["Height"] + 0.707 * X["Diameter"]</code><br>
    <code> df["Shape"] = 0.707 * X["Height"] - 0.707 * X["Diameter"] </code>

    + principal components of the data: `Size`, `Shape`
    + loadings: weights, 0.707
  + number of principal components = features in the original dataset
  + component's loadings expressed through signs and magnitudes
    + table of loadings

      <table>
        <thead>
          <tr><th>Features \ Components</th><th>Size (PC1)</th><th>Shape (PC2)</th></tr>
        </thead>
        <tbody>
          <tr style="text-align: right;"><td>Height</td><td>0.707</td><td>0.707</td></tr>
          <tr style="text-align: right;"><td>Diameter</td><td>0.707</td><td>-0.707</td></tr>
        </tbody>
      </table>

    + `Size` component: `Height` and `Diameter` varying in the same direction (same sign)
    + `Shape` component: `Height` and `Diameter` varying in opposite direction (opposite sign)
    + all loadings w/ the same magnitude $\to$ features contributing equally

+ Percent of explained variance
  + PCA represents the amount of variation in each component
  + more variation in the data along the `Size` component than along the `Shape` component
  + making the precise comparison though each component's percent of explained variation

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/principal-component-analysis')"
        src    = "https://i.imgur.com/xWTvqDA.png"
        alt    = "Size accounts for about 96% and the Shape for about 4% of the variance between Height and Diameter."
        title  = "Size accounts for about 96% and the Shape for about 4% of the variance between Height and Diameter."
      />
    </figure>

  + `Size` component: the majority of variation btw `Height` and `Diameter`
  + the amount of variance in a component
    + not necessarily correspond to how good it is as a predictor
    + depending on what to predict


## PCA for Feature Engineering

+ Ways to use PCA for feature engineering
  + use as a __descriptive technique__
    + computing the MI scores for the components
    + what kind of variation most predictive of the target
    + ideas for kinds of features to create
      + `Size`: product of `Height` and `Diameter`
      + `Shape`: ratio of `Height` and `Diameter`
    + try clustering on one or more of the high scoring components
  + use __components__ themselves as features
    + the components exposing the variational structure of the data directly
    + often more informative than the original features
    + use cases
      + __dimensionality reduction__
        + highly redundant features, in particular, multicolinear
        + partitioning out the redundancy into one or more near-zero variance components
      + __anomaly detection__
        + unusual variation often w/ the low-variance components
        + unusual variation: not apparent from the original features
        + components highly informative in an anomaly or outlier detection task
      + __noise reduction__
        + sensor reading often w/ common background noise
        + able to collect the (informative) signal into a smaller number of features while leaving out the noise
        + boosting the signal-to-noise ratio
      + __decorrelatio__
        + ML sometimes struggling w/ highly-correlated features
        + transforming correlated features into uncorrelated components

+ PCA best practices
  + only working w/ numeric features, including continuous quantities or counts
  + sensitive to scale: standardizing data before applying PCA
  + removing or constraining outliers for undue influence on the results


## Example - 1985 Automobiles

+ Example: 1985 automobiles
  + dataset: [Automobile](https://www.kaggle.com/toramky/automobile-dataset)
  + task: descriptive technique to discover features
  + load data and prepare utility functions

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import display
    from sklearn.feature_selection import mutual_info_regression

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

    def plot_variance(pca, width=8, dpi=100):
        # Create figure
        fig, axs = plt.subplots(1, 2)
        n = pca.n_components_
        grid = np.arange(1, n + 1)
        # Explained variance
        evr = pca.explained_variance_ratio_
        axs[0].bar(grid, evr)
        axs[0].set(
            xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
        )
        # Cumulative Variance
        cv = np.cumsum(evr)
        axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
        axs[1].set(
            xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
        )
        # Set up figure
        fig.set(figwidth=8, dpi=100)
        return axs

    def make_mi_scores(X, y, discrete_features):
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    df = pd.read_csv("data/a18/autos.csv")
    ```

  + selecting features
    + features w/ high MI score on the target, `price`: `highway_mpg`, `engine_size`, `horsepower`, `curb_weight`
    + standardizing the data $\gets$ features naturally not on the same scale

    ```python
    features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]

    X = df.copy()
    y = X.pop('price')
    X = X.loc[:, features]

    # Standardize
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    ```

  + fit PCA and create the principal components

    ```python
    from sklearn.decomposition import PCA

    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    ```




## Exercise








