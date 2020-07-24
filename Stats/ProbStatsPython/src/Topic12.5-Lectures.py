#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt

def load_hw_data(printing=False):
    """lodaing data w/ pandas"""

    hw_df = pd.read_csv('./Topic12-Lectures/data/HW25000.csv')
    hw_df = hw_df.iloc[:, 1:]

    hw_df.columns = ['Height', 'Weight']

    if printing:
        print("\n\nHeight-Weight regression:")
        print("\nhw_df: \n{}".format(hw_df.head()))
        print("\nhw_df.describe: \n{}".format(hw_df.describe()))

    return hw_df

def get_hw_reg(df, printing=False):
    """calculate parameter vector for Height-Weight regression"""
    print(df.head())
    A = np.array(df['Height'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(df['Weight'])

    w = np.linalg.lstsq(A.T, y)[0]

    if printing:
        print("\n\nCalculating the parameter vector for Height-Weight regression:")
        print("\nAw = b w/ w = linalg.lstsq(A.T, b): A= \n{}, \nb.T= {}, w.T= {}")

    return w

def df_average(df, x_name, y_name):
    """calculate mean of x and y axis"""
    # calculate the mean weight for each 1-inch interval of height
    df['round'+x_name] = df[x_name].round()

    per_height_means = df.groupby('round'+x_name).mean()[[y_name]]

    return per_height_means


def f(x, w):
    return w[0]+w[1]*x



def plot_average(df, x_name, y_name):
    per_height_means = df_average(df, x_name, y_name)
    ax = df.plot(kind='scatter', s=1, x=x_name, y=y_name, figsize=[10, 8])
    per_height_means.plot(y=y_name, style='ro', ax=ax, legend=False)

    x0, x1 = plt.xlim()
    y0, y1 = plt.ylim()

    # plot vertical line for grid
    for _x in np.arange(x0+0.5, x1+1, 1):
        ax.plot([_x, _x], [y0, y1], 'g')

    x0, x1 = plt.xlim()
    ax.plot([x0, x1], [f(x0, w1), f(x1, w1)], 'k')
    plt.show()

    return None
    

if __name__ == "__main__":

    print("\nStarting Topic 12.5 Lecture Notes Python code ......")

    hw_df = load_hw_data()

    w1= get_hw_reg(hw_df)

    # A linear graph of averages
    # plot_average(hw_df, 'Height', 'Weight')


    # non-linear graph of averages
    hw_df['P2'] = hw_df['Weight'] + (hw_df['Height']-68)**2
    plot_average(hw_df, 'Height', 'P2')




    print("\nEnd of Topic 12.5 Lecture Notes Python code ......\n")


