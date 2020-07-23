#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt




if __name__ == "__main__":

    print("\nStarting Topic 12.2 Lecture Notes Python code ......")

    # transposing matrix
    print("\n\ncreate a 2x3 matrix by reshape:")
    A = np.array(range(6))
    print("\ninitial A as a 1x6 vector w/ np.array(range(6)): {}  w/ shape= {}".format(A, A.shape))
    B = A.reshape(2, 3)
    print("\ncreate B as a 2x3 matrix w/ A.reshape(2, 3): \n{}  w/ shape= {}".format(B, B.shape))
    print("\ntransposing B matrix w/ B.T: \n{} w/ shape= {}".format(B.T, B.T.shape))

    input("\nPress Enter to continue ...")

    # matrix as a collection of vectors
    print("\n\nDemo the relationship btw matrix and vectors:")
    A = np.array(range(6)).reshape(2, 3)
    print("\ncreate A w/ np.array(range(6)).reshape(2, 3)= \n{}".format(A))
    print("\nSplitting A into columns w/ np.split(A, 3, axis=1)")
    columns = np.split(A, 3, axis=1)
    for i in range(len(columns)):
        print("column {}\n{}".format(i, columns[i]))

    # reconstruct matrix from vectors
    A_recon = np.concatenate(columns, axis=1)
    print("\nReconstruct matrix from columns w/ np.concatenate(columns, axis=1): \n{}".format(A_recon))
    print("\nchecking the reconstruction = original w/ A_recon == A: \n{}".format(A_recon == A))

    # split matrix into row vectors
    print("\n\nSplitting matrix into row vectors w/ np.split(A, 2, axis=0)")
    rows = np.split(A, 2, axis=0)
    for i in range(len(rows)):
        print("row {}: {}".format(i, rows[i]))

    A_recon = np.concatenate(rows, axis=0)
    print("\nreconstruct rows into matrix w/ np.concatenate(rows, axis=0): \n{}".format(A_recon))
    print("\nchecking the reconstruction = original w/ A_recon == A: \n{}".format(A_recon == A))

    input("\nPress Enter to continue ...")

    # matrix scalar operations
    print("\n\nDemo for matrix scalar operations: A =\n{}".format(A))
    print("\naddition A + 3 = 3 + A = \n{}".format(A+3))
    print("\nsubtraction A - 3 = \n{}".format(A-3))
    print("\nmultiplication A x 3 = 3 x A = \n{}".format(A*3))
    print("\ndivision (integer) A / 2 = \n{}".format(A/2))
    print("\ndivision (float) A / 2.0 = \n{}".format(A/2.0))


    # adding & subtraction w/ 2 matrices must be consistent of dimension
    B = np.random.randn(2, 2)
    print("\n\ncreate a random value m2x2 matrices B w/ np.random.randn(2, 2): \n{}".format(B))
    print("\nadding and subtraction on A & B required dimesion checking code\n")
    try:
        rlt = A + B
    except Exception as e:
        print(e)

    # matrix-matrix product
    print("\n\nMatrix-matrix products::")
    A = np.arange(6).reshape((3,2))
    C = np.array([-1, 1])
    print("\nA= {} ]\nC={}".format(A, C))

    print("\ndot product for matrix and vector: np.dot(A, C): \n{}".format(np.dot(A, C)))

    A = np.arange(6).reshape((3,2))
    C = np.random.randn(2, 2)
    print("\n\nA= {}\nC={}".format(A, C))

    print("\ndot product for matrix and matrixvector: A.dot(C): \n{}".format(A.dot(C)))
    print("\ndot product for matrix and matrixvector: np.dot(A, C): \n{}".format(np.dot(A, C)))





    print("\nEnd Topic 12.2 Lecture Notes Python code ......\n")


