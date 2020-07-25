#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt

def f(x, w):
    return w[0]+w[1]*x

def plot_fs_reg(df, debug=False):
    """Plot regression for Father-Son Height

    Args:
        df (dataframe): dataset to analysis
    """
    A = np.array(df['Father'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(df['Son'])

    w1 = np.linalg.lstsq(A.T, y)[0]

    if debug:
        print("\nPearson Father-Son height regression: w.T={}".format(w1.T))
        print("  son's height = {} + {} * father's height".format(w1[0], w1[1]))

    ax = df.plot(kind='scatter', s=1, x='Father', y='Son', figsize=[6, 5])
    x0, x1 = plt.xlim()
    ax.plot([x0, x1], [f(x0, w1), f(x1, w1)], 'r')
    plt.title("Regression of Father's height to Son's height")
    plt.show()

    return None    


def plot_fsd_reg(df, debug=False):
    """Plot regression w/ Father & (Son - Father) height

    Args:
        df (dataframe): data set to analyze
        debug (bool, optional): turn on/off debug print. Defaults to False.
    """
    A = np.array(df['Father'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(df['Son-Father'])

    w2 = np.linalg.lstsq(A.T, y)[0]

    if debug:
        print("\ndimension of A & y: A.shape= {}, y.shape= {}".format(A.shape, y.shape))
        print("\nregression parameter: w.T= {}".format(w2.T))

    if debug:
        print("\nregression parameter of father & difference btw son and father: {}"\
            .format(w2.T))

    ax = df.plot(kind='scatter', s=1, x='Father', y='Son-Father', figsize=[6, 5])
    x0, x1 = plt.xlim()
    ax.plot([x0, x1], [f(x0, w2), f(x1, w2)], 'r')
    plt.title("Regression of Father's height to the difference btw Son and Father")

    plt.show()

    return None


if __name__ == "__main__":

    print("\nStarting Topic 12.6 Lecture Notes Python code ......")

    debug = False

    # load data
    fs_df = pd.read_csv('./Topic12-Lectures/data/Pearson.csv')

    if debug:
        print("\n\nPearson - Father and Son Height data:")
        print("\nfs_df.head: \n{}".format(fs_df.head()))
        print("\nfs_df.describe(): \n{}".format(fs_df.describe()))

    
    # plot regression for Pearson data
    # plot_fs_reg(fs_df, True)
    plot_fs_reg(fs_df)

    # add difference column
    fs_df['Son-Father'] = fs_df['Son'] - fs_df['Father']

    # plot father & (son-father) height regression
    plot_fsd_reg(fs_df)


    print("\nEnd of Topic 12.6 Lecture Notes Python code ......\n")


