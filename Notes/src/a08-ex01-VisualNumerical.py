#!/usr/bin/env python3
# -*-coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.set_theme()

# create two simple continuous variables
x = np.random.normal(size=100)
y = np.random.normal(size=100)

data_df = pd.DataFrame({"x": x, "y": y})

sns.set_theme(style="darkgrid")

# plot the distribution of the data
sns.scatterplot(x="x", y="y", data=data_df, ax=ax[0, 0]);
ax[0, 0].set_title("Scatter Plot")

# create a histogram plot of x variable w/ red color
sns.histplot(data=data_df, x="x", color="r", ax=ax[0, 1])
ax[0, 1].set_title("Histogram")

# create a density plot of x variable
sns.kdeplot(x="x", data=data_df, ax=ax[1, 0])
sns.rugplot(x="x", data=data_df, ax=ax[1, 0])
ax[1, 0].set_title("Density Plot")

# create a box plot of x variable w/ green color
sns.boxplot(x="x", data=data_df, color="g", ax=ax[1, 1])
ax[1, 1].set_title("Box Plot")

plt.show();

