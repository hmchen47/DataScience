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





## Example - MovieLens1M





