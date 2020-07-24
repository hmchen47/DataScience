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

def get_hw_reg(df, x_name, y_name, printing=False):
    """calculate parameter vector for Height-Weight regression"""
    A = np.array(df[x_name])
    A = np.array([np.ones(len(A)), A])
    y = np.array(df[y_name])

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

def plot_average(df, x_name, y_name, title, regline=False):
    per_height_means = df_average(df, x_name, y_name)
    ax = df.plot(kind='scatter', s=1, x=x_name, y=y_name, figsize=[8, 6])
    per_height_means.plot(y=y_name, style='ro', ax=ax, legend=False)

    x0, x1 = plt.xlim()
    y0, y1 = plt.ylim()

    # plot vertical line for grid
    for _x in np.arange(x0+0.5, x1+1, 1):
        ax.plot([_x, _x], [y0, y1], 'g')

    if regline:
        w1 = get_hw_reg(df, x_name, y_name)
        x0, x1 = plt.xlim()
        ax.plot([x0, x1], [f(x0, w1), f(x1, w1)], 'k')
    
    plt.title(title, fontsize=15)
    plt.show()

    return None
    

def f2(x, w):
    return w[0]+w[1]*x+w[2]*x**2


def get_hw_reg2(df, x_name, y_name, printing=False):
    """calculate parameter vector for Height-Weight 2nd degree regression"""
    A = np.array(df[x_name])
    A = np.array([np.ones(len(A)), A, A**2])
    y = np.array(df[y_name])

    w = np.linalg.lstsq(A.T, y)[0]

    if printing:
        print("\n\nCalculating the parameter vector for Height-Weight regression:")
        print("\nAw = b w/ w = linalg.lstsq(A.T, b): \nA= \n{}, \nb.T= {}, w.T= {}"\
            .format(A, y.T, w.T))

    return w
    
def plot_hw_reg2(df, x_name, y_name, title):

    w2 = get_hw_reg2(hw_df, 'Height', 'P2')

    per_height_means = df_average(df, x_name, y_name)
    ax = df.plot(kind='scatter', s=1, x=x_name, y=y_name, figsize=[8, 6])
    per_height_means.plot(y=y_name, style='ro', ax=ax, legend=False)

    x0, x1 = plt.xlim()
    y0, y1 = plt.ylim()

    # plot vertical line for grid
    for _x in np.arange(x0+0.5, x1+1, 1):
        ax.plot([_x, _x], [y0, y1], 'g')

    X = np.arange(x0, x1, (x1-x0)/100)
    Y = f2(X, w2)

    ax.plot(X, Y, 'k')
    plt.title(title, fontsize=15)

    plt.show()

    return None

def F(X, w):
    accum = w[0]*np.ones(len(X))

    for i in rnge(1, len(w)):
        accum += w[i]*i**i

    return accum

def gen_data():
    np.random.seed(0)

    # generate data
    X = np.arange(-1, 1.6, 0.25)
    Y = X + np.random.rand(len(X))

    data = pd.DataFrame({'x': X, 'y': Y})

    return data

def plot_data(df):
    ax = df.plot(kind='scatter', s=30, c='r', x='x', y='y', figsize=[6, 5])
    plt.grid()

    return ax
    


if __name__ == "__main__":

    print("\nStarting Topic 12.5 Lecture Notes Python code ......")

    hw_df = load_hw_data()

    # A linear graph of averages
    title = 'Scattered data and average height w/ regression line'
    # plot_average(hw_df, 'Height', 'Weight', title, regline=True)


    # non-linear graph of averages
    title = 'Scattered data and 2nd-degree polynomial height average'
    hw_df['P2'] = hw_df['Weight'] + (hw_df['Height']-68)**2
    # plot_average(hw_df, 'Height', 'P2', title, regline=False)

    # limits of linear regression
    title = 'Scattered data and 2nd polynomial average w/ regression line'
    # plot_average(hw_df, 'Height', 'P2', title, regline=True)

    # 2nd degree polynomial fit
    title = 'Scattered data and average w/ 2nd degree polynomial fit'
    # plot_hw_reg2(hw_df, 'Height', 'P2', title)

    # overfitting, underfitting amn model selection

    data_df = gen_data()

    # plot data in x-y coordinate
    plot_data(data_df)
    plt.show()



    print("\nEnd of Topic 12.5 Lecture Notes Python code ......\n")


