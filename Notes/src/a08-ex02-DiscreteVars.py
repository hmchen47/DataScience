#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# loading the titanic dataset
titanic_df = sns.load_dataset("titanic")

# print(titanic_df.info())
# print(titanic_df.head())

# creating a count plot for the class variable
sns.countplot(x="class", data=titanic_df, ax=ax1)
ax1.set_title("Count Plot")

# geting the count of each class
values = titanic_df["class"].value_counts().values

# getting the labels of each class
labels = titanic_df["class"].value_counts().index

# creating the pie count
ax2.pie(values, labels=labels, shadow=True, startangle=90)
ax2.set_title("Pie Chart")


plt.show();

