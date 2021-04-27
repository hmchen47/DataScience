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




