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

    # matrix as a collection of vectors
    print("\n\nDemo the relationship btw matrix and vectors:")
    A = np.array(range(6)).reshape(2, 3)
    print("\ncreate A w/ np.array(range(6)).reshape(2, 3)= \n{}".format(A))
    print("\nSplitting A into columns w/ np.split(A, 3, axis=1)")
    columns = np.split(A, 3, axis=1)
    for i in range(len(columns)):
        print("column {}\n{}".format(i, columns[i]))




    print("\nEnd Topic 12.2 Lecture Notes Python code ......\n")


