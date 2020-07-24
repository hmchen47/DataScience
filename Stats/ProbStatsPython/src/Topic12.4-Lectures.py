#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt


def plot_reg(x, y, w):
    """plot data and regression line

    Args:
        x (ndarray): list of x value
        y (ndarray): list of y values
        w (ndarray): parameter vectors
    """
    plt.figure(figsize=[8, 6])
    line = w[0] + w[1]*x
    plt.plot(x, line, 'r-', x, y, 'o')
    plt.title("Differences of data and regression line")
    plt.xlabel("x")
    plt.ylabel("y value")

    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], w[1]*x[i]+w[0]], 'g')
    plt.grid()
    plt.show();

    return None

def f(x, w):
    return w[0]+w[1]*x


def plot_reg_wh():
    """Plot data and its regression line for wright-height relation
    """
    hw_df = pd.read_csv('./Topic12-Lectures/data/HW25000.csv')
    hw_df = hw_df.iloc[:, 1:]
    hw_df.columns = ['Height', 'Weight']

    print("\n\nHeight-Weight regression:")
    print("\nhw_df first 5 rows: \{}".format(hw_df.head()))
    print("\nhw_df statistic info: \n{}".format(hw_df.describe()))

    A = np.array(hw_df['Height'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(hw_df['Weight'])

    w1 = np.linalg.lstsq(A.T, y)[0]
    print("\ndimension: A.shape= {}, y.shape= {}".format(A.shape, y.shape))
    print("\nsolution: w.T = {}".format(w1.T))

    ax = hw_df.plot(kind='scatter', s=1, x='Height', y='Weight', figsize=[10, 8])
    x0, x1 = plt.xlim()
    ax.plot([x0, x1], [f(x0, w1), f(x1, w1)], 'r')
    plt.show();

    return None


if __name__ == "__main__":

    print("\nStarting Topic 12.4 Lecture Notes Python code ......")

    # linear regression small example
    x = np.arange(0, 9)
    y = np.array([[19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]]).T 

    A = np.array([np.ones(9), x]).T 
    print("\n\nA.T = \n{} \nx= {} \ny.T= {}".format(A.T, x, y.T))
    print("\ndimensions: A.shape= {}, y.shape = {}\n".format(A.shape, y.shape))

    w = np.linalg.lstsq(A, y)[0]
    print("\nregression: {}".format(w.T))

    # plot_reg(x, y, w)

    # Height & weight regression
    plot_reg_wh()


    print("\nEnd of Topic 12.4 Lecture Notes Python code ......\n")


