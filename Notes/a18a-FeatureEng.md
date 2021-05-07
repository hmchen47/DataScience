# What is Feature Engineering

Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/what-is-feature-engineering)

[Local notebook](src/a18a-what-is-feature-engineering.ipynb)


## The Goal of Feature Engineering

+ Feature engineering
  + simply making data better suited to the problem at hand
  + example: apparent temperature
    + perceived temperature: temperature to human based on air temperature, humidity, and wind speed, things able to measure directly
    + apparent temperature:
      + temperature measured like the heat index, and the wind chill
      + the result of a kind of feature engineering
    + making the observed data more relevant what actually care about, how it actually feels outside

+ Reasons for feature engineering
  + improving a model's predictive performance
  + reducing computational or data needs
  + improving interpretability of the results


## A Guiding Principle of Feature Engineering

+ Principle of feature engineering
  + useful feature: relationship to the target that your model is able to learn
  + linear model: transforming the features to make features' relationship to the target linear
  + key idea: a transformation applied to a feature becoming in essence a part of model itself
  + example: price of land
    + linear model w/ Length:
      + predicting the Price of square plots of land from the Length of one side
      + fitting a linear model directly to Length  $\to$ not linear $\to$ poor results
    + Linear model w/ $\text{Length}^2$:
      + square the Length feature to get Area $\to$ creating linear relationship
      + adding Area to the feature set $\to$ fitting a parabola w/ linear model
  + high return on time invested in feature engineering

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://www.kaggle.com/ryanholbrook/what-is-feature-engineering" ismap target="_blank">
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/5D1z24N.png"
          alt   = "A linear model fits poorly with only Length as feature."
          title = "A linear model fits poorly with only Length as feature."
        >
      </a>
      <a href="url" ismap target="_blank">
        <img style="margin: 0.1em;" height=200
          src   = "https://i.imgur.com/BLRsYOK.png"
          alt   = "Left: The fit to Area is much better. Right: Which makes the fit to Length better as well."
          title = "Left: The fit to Area is much better. Right: Which makes the fit to Length better as well."
        >
      </a>
    </div>


## Example - Concrete Formulations

+ Example: concrete formulations
  + task: illustrating how adding a few synthetic to dataset to improve the predictive performance of a random forest model
  + dataset: [Concrete](https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength)
    + containing a variety of concrete formulations and the resulting product's compressive strength
    + compressive strength: a measure of how much load that kind of concrete can bear
  + python snippet: read CSV data

    ```python
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    df = pd.read_csv("data/a18/concrete.csv")
    df.head()
    ```

    |	Cement | Blast<br>Furnace<br>Slag |	Fly<br>Ash | Water | Super<br>plasticizer | Coarse<br>Aggregate | Fine<br>Aggregate | Age | Compressive<br>Strength |
    |-------|-------|------|-----|-----|------|------|-------|------|
    | 540.0 | 0.0	  | 0.0 | 162.0 | 2.5 | 1040.0 | 676.0 | 28 | 79.99 |
    | 540.0 | 0.0	  | 0.0 | 162.0 | 2.5 | 1055.0 | 676.0 | 28 | 61.89 |
    | 332.5 | 142.5 | 0.0 | 228.0 | 0.0 | 932.0	 | 94.0	 | 70 | 40.27 |
    | 332.5 | 142.5 | 0.0 | 228.0 | 0.0 | 932.0	 | 94.0	 | 65 | 41.05 |
    | 198.6 | 132.4 | 0.0 | 192.0 | 0.0 | 978.4	 | 25.5	 | 60 | 44.30 |

  + establishing baseline
    + training the model on the un-augmented dataset $\to$ determining whether new features actually useful
    + good practice at the feature engineering process
    + baseline score: help to decide whether the new features worth keeping
    + python snippet

      ```python
      X = df.copy()
      y = X.pop("CompressiveStrength")

      # Train and score baseline model
      baseline = RandomForestRegressor(criterion="mae", random_state=0)
      baseline_score = cross_val_score(
          baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
      )
      baseline_score = -1 * baseline_score.mean()

      print(f"MAE Baseline Score: {baseline_score:.4}")
      # MAE Baseline Score: 8.232
      ```

  + feature added:
    + good predictor of `CompressiveStrength`: ratios of the features
    + adding 3 new ratio features to the dataset

    ```python
    X = df.copy()
    y = X.pop("CompressiveStrength")

    # Create synthetic features
    X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
    X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
    X["WtrCmtRatio"] = X["Water"] / X["Cement"]

    # Train and score model on dataset with additional ratio features
    model = RandomForestRegressor(criterion="mae", random_state=0)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    score = -1 * score.mean()

    print(f"MAE Score with Ratio Features: {score:.4}")
    # MAE Score with Ratio Features: 7.948
    ```

## Python Reference

+ `sklearn.model_selection.cross_val_score¶` method
  + syntax: `sklearn.model_selection.cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan)`
  + docstring: evaluate a score by cross-validation
  + Parameters
    + `estimator`: estimator object implementing ‘fit’<br>The object to use to fit the data.
    + `X`: array-like of shape (n_samples, n_features)<br>The data to fit. Can be for example a list, or an array.
    + `y`: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None<br>The target variable to try to predict in the case of supervised learning.
    + `groups`: array-like of shape (n_samples,), default=None<br>Group labels for the samples used while splitting the dataset into train/test set. Only used in conjunction with a “Group” cv instance (e.g., GroupKFold).
    + `scoring`: str or callable, default=None<br>
      + A str (see model evaluation documentation) or a scorer callable object / function with signature `scorer(estimator, X, y)` which should return only a single value.
      + Similar to [cross_validate](https://bit.ly/3bcA0wJ) but only a single metric is permitted.
      + If `None`, the estimator’s default scorer (if available) is used.
    + `cv`: int, cross-validation generator or an iterable, default=None<br> Determines the cross-validation splitting strategy. Possible inputs for cv are:
      + `None`, to use the default 5-fold cross validation,
      + `int`, to specify the number of folds in a (Stratified)KFold,
      + [CV splitter](https://scikit-learn.org/stable/glossary.html#term-CV-splitter),
      + An iterable yielding (train, test) splits as arrays of indices.

      For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used. These splitters are instantiated with shuffle=False so the splits will be the same across calls.
    + `n_jobs`: int, default=None<br>Number of jobs to run in parallel. Training the estimator and computing the score are parallelized over the cross-validation splits. `None` means 1 unless in a joblib.parallel_backend context. `-1` means using all processors.
    + `verbose`: int, default=0<br>The verbosity level.
    + `fit_params`: dict, default=None<br>Parameters to pass to the fit method of the estimator.
    + `pre_dispatch`: int or str, default=’2*n_jobs’<br>Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. This parameter can be:
      + `None`, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
      + An `int`, giving the exact number of total jobs that are spawned
      + A `str`, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
      + `error_score`: ‘raise’ or numeric, default=np.nan<br>Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised. If a numeric value is given, FitFailedWarning is raised.
  + Returns
    + `scores`: ndarray of float of shape=(len(list(cv)),)<br>Array of scores of the estimator for each run of the cross validation.


