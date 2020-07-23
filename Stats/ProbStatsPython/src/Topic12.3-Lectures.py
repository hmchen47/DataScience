#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt

def F(x, w):
    """get dependent variable values by parameter vector and 

    Args:
        x (float): variable value
        w (float): parameter values
    """
    return w[0]+w[1]*x


def plot_through_line(p, w):
    """plot a line through 2 given points

    Args:
        p (list): list of two points w/ [x, y]
        w (ndarray): a parameter vector w/ w[0] = intercept of y-axis and w[1] = slope
    """
    plt.figure(figsize=[8, 6])
    plt.plot(p[:, 0], p[:, 1], 'ro')
    _xlim = [-1.1, 1.1]
    plt.xlim(_xlim)
    plt.ylim([0, 2.1])
    plt.plot(_xlim, [F(_xlim[0], w), F(_xlim[1], w)])
    plt.grid()
    plt.title('plotting a line passing through the two points')
    plt.show()

    return None

def plot_3_pts():
    """plot three points and unable to find straight line
    """
    plt.figure(figsize=(8, 6))
    p = np.array([[-1, 2], [1,1], [0, 1.25]])

    plt.plot(p[:, 0], p[:, 1], 'ro')
    plt.xlim([-1.1, 1.1])
    plt.ylim([0, 2.1])
    plt.grid()
    plt.,title("No straight line goes through these 3 points")

    return None


if __name__ == "__main__":

    print("\nStarting Topic 12.3 Lecture Notes Python code ......")

    # finding line passing through 2 points
    print("\n\nFinding a line passing through a point")
    p = np.array([[-1, 2], [1, 1]])
    print("\nthe points: \n{}".format(p))

    A = np.array([[1,-1], [1,1]])
    b = np.array([[2], [1]])
    print("\nwritting equations in matrix form: Aw = b")
    print("coefficient matrix: A = \n{}".format(A))
    print("ordinate or dependenable variable vector: b = \n{}".format(b))
    print("goal: finding w for Aw = b")

    A_inv = np.linalg.inv(A)
    w = A_inv.dot(b)
    print("\n\ninverse matrix of A: inv(A)= {}".format(A_inv))
    print("\nfinding w with inverse A w/ A_inv.dot(b): w= inv(A) * b = \n{}".format(w))

    w = np.linalg.solve(A, b)
    print("\nalternatively, get solution w/ np.linalg.solve(A, b): \n{}".format(w))

    plot_through_line(p, w)

    # no straight lin ethrough 3 points
    plot_3_pts()



    print("\nEnd of Topic 12.3 Lecture Notes Python code ......\n")

