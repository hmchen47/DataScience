#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 



def main():

    # Lists vs Numpy Arrays
    x_vec = np.array([1, 2, 3])
    print("\n  Create vector x_vec w/ 'np.array([1, 2, 3]'): {}".format(x_vec))

    lst_c = [1, 2]
    print("\n  the list c_lst= {} w/ length= {}".format(lst_c, len(lst_c)))
    vec_c = np.array(lst_c)
    print("  convert list to vector vec_c w/ 'np.array(lst_c)' and shape (vec_c.shape): {}".format(vec_c, vec_c.shape))

    z = [5, 6]
    ary_z = np.array(z)
    print("\n  the lst z = {} w/ type {}".format(z, type(z)))
    print("  array ary_z = {} w/ type {}".format(ary_z, type(ary_z)))

    input("\nPress Enter to continue ......")

    # array as vector
    v1 = np.array([[1, 2], [3, 4]])
    print("\ncreate a 2D matrix w/ 'np.array([[1, 2], [3, 4]])': \n{}".format(v1))



    return None


if __name__ == "__main__":

  print("\nStarting Topic 12.1 Lecture Note for Python code ......")

  main()

  print("\nEnd of Topic 12.1 Lecture Note for Python code ......\n")
