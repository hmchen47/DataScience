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



if __name__ == "__main__":

    print("\nStarting Topic 12.4 Lecture Notes Python code ......")

    # linear regression small example
    x = np.arange(0, 9)
    y = np.array([[19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]]).T 

    A = np.array([np.ones(9), x]).T 
    print("\n\nA.T = \n{} \nx= {} \ny.T= {}".format(A.T, x, y.T))
    print("\ndimensions: A.shape= {}, y.shape = {}".format(A.shape, y.shape))

    w = np.linalg.lstsq(A, y)[0]
    print("\nregression: {}".format(w))

    plot_reg(x, y, w)


    print("\nEnd of Topic 12.4 Lecture Notes Python code ......\n")


