#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


def plot_arrow(L, scale=4, text_loc=0.2, fontsize=12, title=None):
    """plot a list of arrow.  each arrow defined by start point, end point, color, and optional text

    Args:
        L (list): [[[s_x, s_y], [e_x, e_y], color, 'string'], ...], s = start point, e = end point
        scale (int, optional): limit of the x- ands y-axis. Defaults to 4.
        text_loc (float, optional): a translated value to shift text location. Defaults to 0.2.
        fontsize (int, optional): the size of font for label of the arrow text. Defaults to 12.
    """
    plt.figure(figsize=([6,6]))
    plt.xlim([-scale, scale])
    plt.ylim([-scale, scale])

    ax = plt.axes()
    plt.xlabel('1st cood (x)')
    plt.ylabel('2nd cood (y)')
    if title != None:
        plt.title(title, fontsize=fontsize+3)

    for A in L:
        s, e, c = A[:3]
        ax.arrow(s[0], s[1], e[0], e[1], head_width=0.05*scale, head_length=0.1*scale,\
            fc=c, ec=c, length_includes_head=True)
        if len(A) == 4:
            t = A[3]
            _loc = 1 + text_loc/np.linalg.norm(e)
            ax.text(_loc*e[0], _loc*e[1], t, fontsize=fontsize)

    plt.grid()
    plt.show()

    return ax




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

    # array as vector
    v1 = np.array([[1, 2], [3, 4]])
    print("\ncreate a 2D matrix w/ 'np.array([[1, 2], [3, 4]])': \n{}".format(v1))


    input("\nPress Enter to continue ......")

    # Visualizing 2D vectors

    orig = np.array([0, 0])
    v1 = np.array([1, 2])
    v2 = np.array([-1, 1])
    v3 = np.array([0, 2])
    title = "Vector arrow representation in 2D coordinate"
    plot_arrow([[orig, v1, 'r', str(v1)], [orig, v2, 'k', str(v2)], \
        [orig, v3, 'b', str(v3)]], scale=3, title=title)


    return None


if __name__ == "__main__":

  print("\nStarting Topic 12.1 Lecture Note for Python code ......")

  main()

  print("\nEnd of Topic 12.1 Lecture Note for Python code ......\n")
