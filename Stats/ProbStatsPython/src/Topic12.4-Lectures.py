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

def load_hw_data(printing=False):
    hw_df = pd.read_csv('./Topic12-Lectures/data/HW25000.csv')
    hw_df = hw_df.iloc[:, 1:]
    hw_df.columns = ['Height', 'Weight']

    if printing:
        print("\n\nHeight-Weight regression:")
        print("\nhw_df first 5 rows: \{}".format(hw_df.head()))
        print("\nhw_df statistic info: \n{}".format(hw_df.describe()))

    return hw_df

def plot_reg_wh():
    """Plot data and its regression line for wright-height relation
    """
    hw_df = load_hw_data(printing=True)

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


def plot_average():

    hw_df = load_hw_data()
    hw_df['round_height'] = hw_df['Height'].round()
    hw_df['round_weight'] = hw_df['Weight'].round()

    A = np.array(hw_df['Height'])
    A = np.array([np.ones(len(A)),A])
    y = np.array(hw_df['Weight'])
    
    w1 = np.linalg.lstsq(A.T,y)[0]
    
    per_height_means = hw_df.groupby('round_height').mean()[['Weight']]

    ax = hw_df.plot(kind='scatter', s=1, x='Height', y='Weight', figsize=[10, 8])
    per_height_means.plot(y='Weight', style='ro', ax=ax, legend=False)

    _xlim = plt.xlim()
    _ylim = plt.ylim()

    for _x in np.arange(_xlim[0]+0.5, _xlim[1], 1):
        plt.plot([_x, _x], [_ylim[0], _ylim[1]], 'g')

    x0, x1 = plt.xlim()
    ax.plot([x0, x1], [f(x0, w1), f(x1, w1)], 'k')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('The Height-Weight data plot and their regression line: y-intercept= {:.2f} & slope= {:.2f}'\
        .format(w1[0], w1[1]))
    plt.show()

    return None

def plot_two_reg():

    hw_df = load_hw_data()

    A = np.array(hw_df['Height'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(hw_df['Weight'])
    w1 = np.linalg.lstsq(A.T, y)[0]

    A = np.array(hw_df['Weight'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(hw_df['Height'])

    w2 = np.linalg.lstsq(A.T, y)[0]

    ax = hw_df.plot(kind='scatter', s=1, x='Height', y='Weight', figsize=[10,8])
    x0, x1 = plt.xlim()
    ax.plot([x0, x1], [f(x0, w1), f(x1, w1)], 'r', label="Weight from Height")

    y0, y1 = plt.ylim()
    ax.plot([f(y0, w2), f(y1, w2)], [y0, y1], 'k', label='Height from Weight')
    plt.legend()
    plt.title('Two regression problem for Height and Weight')
    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    plt.grid()
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

    plot_reg(x, y, w)

    # Height & weight regression
    plot_reg_wh()

    # the graph of average
    plot_average()

    # two regression lines
    plot_two_reg()



    print("\nEnd of Topic 12.4 Lecture Notes Python code ......\n")


