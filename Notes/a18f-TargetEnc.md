# Target Encoding


Author: R. Holbrook

Organization: Kaggle

[Original](https://www.kaggle.com/ryanholbrook/target-encoding)

[Local notebook](src/a18f-target-encoding.ipynb)


## Target Encoding

+ Supervised feature encoding engineering
  + a method of encoding categories as integer number
  + example: one-hot or label encoding

+ Target encoding
  + any kind of encoding replacing a feature's categories w/ some number derived from the target
  + simple and effect version: applying a group aggregation, like the mean
  + Automobiles: average price of each vehicle's make

    ```python
    autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
    ```

  + mean encoding: applying a group aggregation w/ mean
  + other encodings: likelihood encoding, impact encoding, and leave-one-out encoding


## Smoothing

+ Issues of encoding
  + unknown categories
    + creating a special risk of overfitting
    + required to be trained on an independent "encoding" split
    + imputation: filling in missing values for any categories
  + rare categories
    + any statistics on this group unlikely very accurate
    + example: Automobiles
      + Mercurcy make only occurred once
      + mean price not very representative of any Mercurcies
      + making overfitting more likely
    + solution: smoothing

+ Smoothing technique
  + blending the in-category average w/ the overall average
  + rare categories: less weight on their category average
  + missing categories: the overall average
  + pseudocode

    <code>encoding = weight * in_category + (1 - weight) * overall </code>

  + weight
    + a value btw 0 and 1 calculated from the catgory frequency
    + determining weight by computing __m-estimate__

      <code>weight = n / (n + m) </code>

    + $n$: the total number of times the category occurred in the data
    + $m$: hyperparameter to determine the "smoothing factor"
    + value for $m \to$ how noisy expecting the categories to be
      + target values varying a great deal $\implies$ choosing a larger value for $m$
      + target values relatively stable $\implies$ choosing a smaller value
  + larger values of $m$ $\to$ more weight on the overall estimate

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 20vw;"
        onclick= "window.open('https://www.kaggle.com/ryanholbrook/target-encoding')"
        src    = "https://i.imgur.com/1uVtQEz.png"
        alt    = "M-estimate w/ categories count and smoothing factor"
        title  = "M-estimate w/ categories count and smoothing factor"
      />
    </figure>

  + example: Automobiles
    + 3 cars w/ the make Chevrolet, $n = 3$
    + $m = 0.2$: 

      <code>chevrolet = 0.6 * 6000.00 + 0.4 * 13285.03</code>

+ Use cases for target encoding
  + high-cardinality features:
    + a feature w/ large number of categories: troublesome to encode
    + one-hot encoding:
      + generating too many features and alternative
      + not appropriate for that feature
    + target encoding: deriving numbers for the categories w/ the relationship w/ the target
  + domain-motivated feature
    + prior experience: categorical feature probably not so important even if scored poorly w/ a feature metric
    + target encoding revealing a feature's true informative 

## Example - MovieLens1M

+ Example: MovieLens1M
  + dataset: [MovieLens1M](https://www.kaggle.com/grouplens/movielens-20m-dataset)
    + 1 million movie rating by users of the MovieLens website
    + features describing each user and movie
  + loading data and preparing plotting

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import warnings

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
    warnings.filterwarnings('ignore')

    df = pd.read_csv("data/a18/movielens1m.csv")
    df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
    print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))
    # Number of Unique Zipcodes: 3439
    ```

  + preparing the training dataset
    + a good candidate for target encoding/: > 3000 categories
    + size of the dataset: over one-million rows
    + creating a 25\% split to train the target encoder

    ```python
    X = df.copy()
    y = X.pop('Rating')

    X_encode = X.sample(frac=0.25)
    y_encode = y[X_encode.index]
    X_pretrain = X.drop(X_encode.index)
    y_train = y[X_pretrain.index]
    ```

  + encoding w/ MEstimate encoder
    + utilizing m-estimate encoder w/ `category=encoder` lib
    + encoding `Zipcode` feature

    ```python
    from category_encoders import MEstimateEncoder

    # Create the encoder instance. Choose m to control noise.
    encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

    # Fit the encoder on the encoding split.
    encoder.fit(X_encode, y_encode)

    # Encode the Zipcode column to create the final training data
    X_train = encoder.transform(X_pretrain)
    ```

  + 


