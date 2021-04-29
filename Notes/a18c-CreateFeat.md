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





## Counts





## Building-Up and Breaking-Down Features





## Group Transforms





## Exercise





