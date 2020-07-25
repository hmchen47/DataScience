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
    return mean+(x - mean).dot(u)*u



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
    prjvec = project(A[3], eigvecs[0:], mean)
    print("\nProjection vecctor of x({}) on \nu= \n{} \nprjection vector: \n{}"\
        .format(A[3], eigvecs[0:], prjvec))


    print("\nEnd of Topic 12.7 Lecture Notes Python code ......\n")


