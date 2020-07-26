#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt

def PCA(A, debug=False):
    """Principal Component Analysis to get mean, eigenvalues and eigenvectors

    Args:
        A (ndarray): input matrix for analysis
        debug (bool, optional): turn on/off debuging msg. Defaults to False.
    """
    mean = np.mean(A.T, axis=1) # compute the location of the mean
    M = (A - mean).T            # subtract the mean (along column)
    [eigvals, eigvecs] = np.linalg.eig(np.cov(M))

    # ording vectors so that eigenvalues decreasing order
    order = np.argsort(eigvals)[-1::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[order]
    eigvecs = eigvecs.T

    if debug:
        print("\nPrincipal Component Analysis (PCA) w/ A: \n{}\n".format(A))
        print("order= {} \nmean= {} \neigvals= {}\neigvecs= \n{}"\
            .format(order, mean, eigvals, eigvecs))

    return mean, eigvals, eigvecs


def project(x, u, m):
    """Projection of a vector to a unit vector

    Args:
        x (ndarray): input vector
        u (ndarray): unit vector
        m (ndarray): mean values
    """
    return m+(x - m).dot(u)*u


def plot_regression_eigvecs(x, y, mean, eigvals, eigvecs):
    """plot regression and eigenvectors w/ difference of data and regression

    Args:
        x (ndarray): variables
        y (ndarray): dependent variable vector
        mean (ndarray): mean values of columns
        eigvals (ndarray): eigenvalues of the system
        eigvecs (ndarray): eigenvectors of the system
    """
    fig = plt.figure(figsize=[10,8])

    # fig = plot_PCA(x, y, mean, eigvals, eigvecs, fig)
    marker = 'o'
    markersize = 8

    plt.plot(x, y, 'o', markersize=markersize, label="Data")
    plt.plot(mean[0], mean[1], 'kx', markersize=20, mew=10)

    colors = ['r', 'm']
    labels = ['Primary EigenVector', 'Secondary Eigenvector']
    for i in range(2):
        principle = eigvecs[i,:]
        stdev = np.sqrt(eigvals[i])*2
        p1 = mean - principle*stdev
        p2 = mean + principle*stdev
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], colors[i], label=labels[i])

    # fig = plot_projections(A, eigvals, mean, fig)
    for i in range(A.shape[0]):
        pt = A[i, :]
        proj = project(pt, eigvecs[0,:], mean)
        if i == 0:
            plt.plot([pt[0], proj[0]], [pt[1], proj[1]], 'g', \
                label="Projection to \nPrimary Eigenvector")
        else:
            plt.plot([pt[0], proj[0]], [pt[1], proj[1]], 'g')


    # fig = plot_regress_projections(x, y, w, fig)

    line = w[0]+w[1]*x  # regression line
    plt.plot(x, line, 'k-', label='Regression')

    # plot difference btw data and regression line
    for i in range(len(x)):
        if i == 0:
            plt.plot([x[i], x[i]], [y[i], w[1]*x[i]+w[0]], 'y', label="Diff. of y & Reg")
        else:
            plt.plot([x[i], x[i]], [y[i], w[1]*x[i]+w[0]], 'y')

    plt.axis('equal')
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.grid()
    plt.legend()
    plt.title("Data, Regression Line, Eigenvectors and the Projections",\
        fontsize=15)

    plt.show()

    return None




if __name__ == "__main__":

    print("\nStarting Topic 12.7 Lecture Notes Python code ......")

    # small example
    # generate data for analysis
    x = np.arange(0, 9)
    y = [21, 19, 23, 21, 25, 22, 25, 23, 24]

    A = np.array([np.ones(len(x)), x])
    w = np.linalg.lstsq(A.T, y)[0]

    print("\n\nRegression line from Aw = y with \nA= \n{} \ny= {} \nto get w={}"\
        .format(A, y, w))

    # PCA analysis
    A = np.array(list(zip(x, y)))

    mean, eigvals, eigvecs = PCA(A, debug=True)

    # example of projection
    prjvec = project(A[3], eigvecs[0,:], mean)
    print("\nProjection vecctor of x({}) on \nu= {} \nprjection vector: {}"\
        .format(A[3], eigvecs[0,:], prjvec))


    # plot data and mean
    plot_regression_eigvecs(x, y, mean, eigvals, eigvecs)


    print("\nEnd of Topic 12.7 Lecture Notes Python code ......\n")


