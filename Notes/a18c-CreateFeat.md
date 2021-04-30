# Creating Features

Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/creating-features)

[Local notebook](src/a18c-creating-features.ipynb.ipynb)


## Introduction

+ Tips to discovering new features
  + understand the features: referring to data documentation if available
  + acquire domain knowledge: research the problem domain
  + study previous work
  + use data visualization:
    + revealing pathologies in the distribution of a feature
    + simplifying complicated relationships
    + a must step for feature engineering process

+ Example: load datasets and set plotting environment
  + datasets
    + [US Traffic Accidents](https://www.kaggle.com/sobhanmoosavi/us-accidents)
    + [1985 Automobiles](https://www.kaggle.com/toramky/automobile-dataset)
    + [Concrete Formulations](https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength)
    + [Customer Lifetime Value](https://www.kaggle.com/pankajjsh06/ibm-watson-marketing-customer-value-data)
  + python snippet

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold", labelsize="large",
        titleweight="bold",
        titlesize=14, titlepad=10,
    )

    accidents = pd.read_csv("data/a18/accidents.csv")
    autos = pd.read_csv("data/a18/autos.csv")
    concrete = pd.read_csv("data/a18/concrete.csv")
    customer = pd.read_csv("data/a18/customer.csv")
    ```

## Mathematical Transforms

+ Representing feature relationships
  + relationship among numerical features usually expressed mathematical formulas
  + ratio:
    + features describing a car's engine in Automobile dataset
    + a variety of formulas for creating potentially useful new feature
    + e.g., `stroke ratio`: a measure of how efficient an engine vs how performant
  + combination
    + complicated formulation among features
    + the more complicated combination is, the more difficult it will ne for a model learn
    + e.g., engine;s "displacement" as a measure of its power
  + data visualization
    + able to suggest transformations
    + often a "reshaping" of a feature through powers or logarithms
    + e.g., highly skewed distribution of `Windspeed` in US Accidents
  + python snippet

    ```python
    autos["stroke_ratio"] = autos.stroke / autos.bore

    autos["displacement"] = (
        np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
    )

    # If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
    accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

    # Plot a comparison
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
    sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);
    ```

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 30vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/creating-features')"
        src    = "https://bit.ly/3e0Yjjc"
        alt    = "Data visualization with "
        title  = "Data visualization with "
      />
    </figure>


## Counts

+ Counting features
  + features describing presence or absence
  + representing such features w/ binary (1 for presence , 0 for Absence) or Boolean (True or False)
  + dealing such features in sets
  + new "counts" features: aggregating such features
  + able to create Boolean values w/ dataframe built-in methods
  + example: Traffic Accidents
    + features indicating whether some roadway near the accident
    + creating a count of the total number of roadway features

    ```python
    roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
        "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
        "TrafficCalming", "TrafficSignal"]
    accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
    ```

  + example: Concrete
    + features: the components in concrete formulation
    + many formulations w/o some components, e.g., value w/ 0
    + count how many components used in formulations

    ```python
    components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
              "Superplasticizer", "CoarseAggregate", "FineAggregate"]
    concrete["Components"] = concrete[components].gt(0).sum(axis=1)
    ```


## Building-Up and Breaking-Down Features

+ Manipulating structure data
  + complex strings usually broken into simpler pieces
  + common examples of structure data
    + ID numbers: `'123-45-6789'`
    + Phone numbers: `'(999) 555-0123'`
    + Street addresses: `'8241 Kaggle Ln., Goose City, NV'`
    + Internet addresses: `'http://www.kaggle.com'`
    + Product codes: `'0 36000 29145 2'`
    + Dates and times: `'Mon Sep 30 07:06:05 2013'`
  + able to apply string methods, like `split`, directly to columns

    ```python
    customer[["Type", "Level"]] = ( # Create two new features
        customer["Policy"]          # from the Policy feature
        .str                        # through the string accessor
        .split(" ", expand=True)    # by splitting on " "
                                    # and expanding the result into separate columns
    )
    #     Policy        Type        Level
    # 0   Corporate L3  Corporate   L3
    #     ...
    ```

  + able to join simple features into a composed feature

    ```python
    autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
    # 0   alfa-romero   convertible   alfa-romero_convertible
    #     ...
    ```

## Group Transforms

+ Group transforms
  + aggregating information across multiple rows grouped by some category
  + good practice: category interaction $\to$ group transform over the category
  + aggregation function to combine two features
    + grouping categorical feature
    + aggregating feature values
  + built-in dataframe method as aggregation function, e.g., `mean`, `max`, `min`, `median`, `var`, `std`, `count`
  + preventing inappropriate data splitting
    + using training and validation splits to preserve their independence
    + best practice
      + creating a grouped feature using only the training set
      + joining it to the validation set
      + using the validation set's `merge` set after creating a unique set of values w/ `drop_duplicates` on the training set
  + example: average income by state
    + `state` as grouping feature
    + `mean` as aggregation function
    + `Income` for the aggregated feature

    ```python
    customer["AverageIncome"] = (
        customer.groupby("State")  # for each state
        ["Income"]                 # select the income
        .transform("mean")         # and compute its mean
    )
    #   State       Income    AverageIncome
    # 0 Washington  56274     38122.733083
    # 1 Arizona     0         37405.402231
    #   ...
    ```

  + example: frequency w/ which state occurs

    ```python
    customer["StateFreq"] = (
        customer.groupby("State")
        ["State"]
        .transform("count")
        / customer.State.count()
    )
    #   State       StateFreq
    # 0 Washington  0.087366
    #   ...
    ```

  + 

  + example: handling data splitting

    ```python
    # Create splits
    df_train = customer.sample(frac=0.5)
    df_valid = customer.drop(df_train.index)

    # Create the average claim amount by coverage type, on the training set
    df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

    # Merge the values into the validation set
    df_valid = df_valid.merge(
        df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
        on="Coverage",
        how="left",
    )

    #  Coverage  AverageClaim
    #0 Extended  482.887836
    #   ...
    ```

+ Tips for creating features
  + linear models
    + learning sums and differences naturally
    + unable to learn anything more complex
  + ratio:
    + difficult for most models to learn
    + ratio combinations leading to some easy performance gains
  + normalization
    + linear models and Neural Nets generally doing better w/ normalized features
    + NN: features scaled to values not too far from 0
    + tree-based models also beneficial from normalization but limited
  + tree models
    + learning to approximate almost any combination of features
    + combination especially important when limited data
  + counts:
    + especially helpful for tree models
    + tree models w/o natural way of aggregating information across many features at once


## Exercise

+ Exercise: creating features
  + task: developing mathematical transforms
    + features describing areas
    + same units (square-feet)
    + using `XGBoots` (a tree-based model) $\to$ focus on ratios and sums
  + create mathematical transforms
    + `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
    + `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
    + `TotalOutsideSF`: the sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `Threeseasonporch`, and `ScreenPorch`

    ```python
    X_1 = pd.DataFrame()  # dataframe to hold new features

    X_1["LivLotRatio"] = X.GrLivArea/X.LotArea
    X_1["Spaciousness"] = (X.FirstFlrSF + X.SecondFlrSF) / X.TotRmsAbvGrd
    X_1["TotalOutsideSF"] = X[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", \
                              "Threeseasonporch", "ScreenPorch"]].sum(axis=1)
    ```

  + interaction w/ a categorical
    + discovering and interaction btw `BldgType` and `GrLivArea`
    + using one-hot encoding
    + example python snippet w/ one-hot encoding

      ```python
      # One-hot encode Categorical feature, adding a column prefix "Cat"
      X_new = pd.get_dummies(df.Categorical, prefix="Cat")

      # Multiply row-by-row
      X_new = X_new.mul(df.Continuous, axis=0)

      # Join the new features to the feature set
      X = X.join(X_new)
      ```

    + python snippet

      ```python
      # One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
      X_2 = pd.get_dummies(X.BldgType, prefix="Bldg")
      # Multiply
      X_2 = X_2.mul(X.GrLivArea, axis=0)
      ```

  + count feature
    + creating a feature to describe how many kinds of outdoor areas a dwelling has
    + `PorchTypes`: count how many of the following greater than 0.0
      + `WoodDeckSF`
      + `OpenPorchSF`
      + `EnclosedPorch`
      + `Threeseasonporch`
      + `ScreenPorch`

    ```python
    X_3 = pd.DataFrame()
    components = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch", "ScreenPorch"]

    X_3["PorchTypes"] = X[components].gt(0).sum(axis=1)
    ```

  + breaking down a categorical feature
    + `MSSubClass` features:
      + display the set of categories: `df.MSSubClass.unique()`
      + more generalized categorization by the first word of each category
    + create a feature containing only the first words of the feature

    ```python
    X_4 = pd.DataFrame()

    X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
    ```

  + using a group transform
    + creating feature `MedNhbdArea`
    + `MedNhbdArea`: the median of `GrLivArea` grouped on `Neighborhood`

    ```python
    X_5 = pd.DataFrame()

    X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    ```

  + training w/ `XGBoost` and evaluate the score

    ```python
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

    X_new = X.join([X_1, X_2, X_3, X_4, X_5])
    score_dataset(X_new, y)
    # 0.13847331710099203
    ```
