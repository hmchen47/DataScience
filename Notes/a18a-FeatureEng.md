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
    + linear model w. Length:
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
    + compressive strength: a measure of how much much load that kind of concrete can bear
  + python snippet: read CSV data

    ```python
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    df = pd.read_csv("../input/fe-course-data/concrete.csv")
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
    + good predictor of CompressiveStrength: ratios of the features
    + adding 3 new ratio features tot he dataset

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


