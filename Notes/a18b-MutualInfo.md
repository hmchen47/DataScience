# Mutual Information

Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/mutual-information)

[Local notebook](src/a18b-mutual-information.ipynb)


## Introduction

+ Handling features
  + issue: hundreds and thousands of features w/o description
  + procedure to resolve
    + constructing a ranking w/ a __feature utility metric__, a function measuring associatiions btw a feature and a target
    + choosing a smaller set of the most useful features to develop initially and having more confidence to spend time on them

+ Mutual information
  + metric used to measure associations btw a feature and a target
  + a lot like correlation to measure a relationship btw two quantities
  + MI detecting any kind of relationship while correlation only detecting linear relationship
  + a great general-purpose metric and specially useful at the start of feature development
  + advantages
    + easy to use and interpret
    + computationally efficient
    + theoretically well-founded
    + resistant to overfitting
    + able to detect any kind of relationship

## Mutual Information and What it Measures

+ Mutual information and measurement
  + MI describing relationships in terms of _uncertainty_
  + __mutual information (MI)__ btw two quantities: a measure of the extent to which knowledge of one quantity reduces uncertainty about the other
  + scikit-learn algorithm for MI
    + two mutual information metrics in `feature_selection` module
    + continuous features
      + `float` dtype
      + real value targets: `mutual_info_regression`
    + categorical features
      + `object` or `categorical` dtype
      + treated as discrete by giving them a label encoding
      + categorical targets: `mutual_info_classif`
  + data visualization: a great toolbox for feature ranking, e.g., bar chart
  + example: Ames Housing data
    + the relationship btw the exterior quality of a house and the price it sold for
    + diagram
      + knowing the value of `ExterQual` to make more certain about the corresponding `SalePrice`
      + MI (`ExterQual` w/ `SalePrice`): the average reduction of uncertainty in `SalePrice` taken over the four values of `ExterQual`
    + entropy: uncertainty measured using a quantity from information theory
    + the entropy of a variable (rough): how many yes-or-no questions required to describe an occurrence of that variable, on average

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/mutual-information')"
        src    = "https://i.imgur.com/X12ARUK.png"
        alt    = "Knowing the exterior quality of a house reduces uncertainty about its sale price."
        title  = "Knowing the exterior quality of a house reduces uncertainty about its sale price."
      />
    </figure>

## Interpreting Mutual Information Scores

+ Mutual information scores
  + MI = 0.0
    + least possible value
    + independent: unable to tell anything about the other
  + MI maximum value
    + theory: no upper bound
    + practice: MI > 2.0 uncommon
    + MI: a logarithm quantity

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/mutual-information')"
        src    = "https://i.imgur.com/Dt75E1f.png"
        alt    = "Left: Mutual information increases as the dependence between feature and target becomes tighter. Right: Mutual information can capture any kind of association (not just linear, like correlation.)"
        title  = "Left: Mutual information increases as the dependence between feature and target becomes tighter. Right: Mutual information can capture any kind of association (not just linear, like correlation.)"
      />
    </figure>

+ Considerations when using mutual information
  + relative potential:
    + MI helping to understand the relative potential of a feature
    + the potential as a predictor of the target
  + univariate metric
    + possible for a feature very informative when interacting w/ other features
    + not so informative for the feature itself
    + MI unable to detect interaction btw features
  + feature and model
    + the usefulness of feature depending on the model it used w/
    + feature probably only useful to the extent related to the target
    + a feature w/ high MI score $\nRightarrow$ model able to do anything w/ that information


## Example - 1985 Automobiles

+ Example: mutual information  w/ 1985 automobiles
  + dataset: [Automobile dataset](https://www.kaggle.com/toramky/automobile-dataset)
  + goal: predicting a car's `price` (the target) from 23 pf the car's features
  + task: ranking the features w/ mutual information and investigate the results by data visualization
  + python snippet to import libraries and read the CSV data

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.style.use("seaborn-whitegrid")

    df = pd.read_csv("data/Automobile_data.csv")
    df.head()
    ```

  + python snippet to process categorical encoding

    ```python
    X = df.copy()
    y = X.pop("price")

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int
    ```

  + python snippet for mutual information scores

    ```python
    from sklearn.feature_selection import mutual_info_regression

    def make_mi_scores(X, y, discrete_features):
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    mi_scores = make_mi_scores(X, y, discrete_features)
    mi_scores[::3]  # show a few features with their MI scores

    # curb_weight          1.526026
    # highway_mpg          0.958583
    # length               0.615287
    # bore                 0.496247
    # stroke               0.375345
    # num_of_cylinders     0.330281
    # compression_ratio    0.133210
    # fuel_type            0.047279
    # Name: MI Scores, dtype: float64
    ```

  + python snippet to plot bar chart for comparison

    ```python
    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")

    plt.figure(dpi=100, figsize=(8, 5))
    plot_mi_scores(mi_scores)
    
    # plot for high score 'curb_weight'
    sns.relplot(x="curb_weight", y="price", data=df);
    ```

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://www.kaggle.com/ryanholbrook/mutual-information" ismap target="_blank">
        <img style="margin: 0.1em;" height=250
          src   = "https://bit.ly/3vjAkBA"
          alt   = "Mutual information scores"
          title = "Mutual information scores"
        >
        <img style="margin: 0.1em;" height=250
          src   = "https://bit.ly/3nuAIuc"
          alt   = "Plot for curb weight and price"
          title = "Plot for curb weight and price"
        >
      </a>
    </div>

  + `feul_type` feature (see diagram)
    + w/ a fair low MI score
    + two price populations within the `horsepower` feature
    + probably not unimportant according to MI score
    + good to further investigate any possible effects
    + python snippet: `sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);`

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/mutual-information')"
        src    = "https://bit.ly/2R8kGdg"
        alt    = "Hosepower vs. Price"
        title  = "Hosepower vs. Price"
      />
    </figure>

