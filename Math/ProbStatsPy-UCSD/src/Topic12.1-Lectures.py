#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 


def plot_arrows(L, scale=4, text_loc=0.2, fontsize=12, title=None):
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
    # plt.show()

    return ax


def main():

    # Lists vs Numpy Arrays
    print("\nCreating vector and matrix by np.array")
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

    # input("\nPress Enter to continue ......")

    # Visualizing 2D vectors
    print("\nVisualizing vector in 2D plane")
    orig = np.array([0, 0])
    v1 = np.array([1, 2])
    v2 = np.array([-1, 1])
    v3 = np.array([0, 2])
    title = "Vector arrow representation in 2D coordinate"
    # plot_arrows([[orig, v1, 'r', str(v1)], [orig, v2, 'k', str(v2)], \
    #     [orig, v3, 'b', str(v3)]], scale=3, title=title)
    # plt.show()

    # input("\nPress Enter to continue ......")
    
    # Operations in vectors w/ visualization
    print("\nOperations on vectors w/ visualization:")
    print("  v1 + v2: {} + {} = {}".format(v1, v2, v1+v2))
    # plot_arrows([[orig, v1, 'r', '$\\vec{v}_1$'], [orig, v2, 'k', '$\\vec{v}_2$'], [v1, v2, 'k'], \
    #     [orig, v1+v2, 'b', '$\\vec{v}+1 + \\vec{v}_2$']], fontsize=15)
    # plt.show()

    # two vectors summed only if w/ the same dimesion
    print("\nv1 = {}, v2 = {} --> v1+v2".format(np.array([1, 1]), np.array([1, 1, 2])))
    try:
        np.array([1, 1]) + np.array([1, 1, 2])
    except:
        print("these two vectors w/ different dimension")

    # inner product
    print("\nways to calculate innerproduct: v1={} v2={}".format(v1, v2))
    print("  np.dot(v1, v2) = {}".format(np.dot(v1, v2)))
    print("  v1[0]*v2[0]+v1[1]*v2[1]= {}".format(v1[0]*v2[0]+v1[1]*v2[1]))
    print("  sum of comprehensive list= {}"\
        .format(np.sum([v1[i]*v2[i] for i in range(len(v1))])))

    # norm of a vector
    print("\nways to calculating the norm of the vector: v1={}".format(v1))
    print("  sqrt(np.dot(v1, v1)= {}".format(math.sqrt(np.dot(v1, v1))))
    print("  np.linalg.norm(v1)=  {}".format(np.linalg.norm(v1)))

    # unit vector
    print("\nnormalizing any vector to unit vector: v1= {}, norm= {}"\
        .format(v1, np.linalg.norm(v1)))
    print("  unit vector w/ u1=v1/norm(v1): {}".format(v1/np.linalg.norm(v1)))
    print("  and its norm: {}".format(np.linalg.norm(v1/np.linalg.norm(v1))))

    # projection and visualization
    u1 = v1/np.linalg.norm(v1)
    p = u1*np.dot(u1,v2)
    print("\nProjects: u1= {}".format(u1))
    print("  projection a vector w/ u1*np.dot(u1, v2) = {}".format(p))

    # ax = plot_arrows([[orig, u1, 'r', '$\\vec{u}_1$'],[orig, v2, 'k', '$\\vec{v}_2$'],\
    #             [orig, p, 'g', "$\\vec{u}_1(\\vec{u}_1\\cdot \\vec{v}_2)$"]],\
    #            scale=1.3,text_loc=0.05)
    # ax.plot([v2[0],p[0]],[v2[1],p[1]])
    # plt.show();


    # orthogonal vector
    a = np.array([2, 2])
    b = np.array([-1, 1])
    print("\northogonal vectors: a={} b={}".format(a, b))
    print("  dot product np.dot(a, b)= {}".format(np.dot(a, b)))
    # plot_arrows([[orig, a, 'r'], [orig, b, 'k']])
    # plt.show()


    # orthonormal basis
    v = np.array([5, 6, 3, 4])
    e2 = np.array([0, 1, 0, 0])
    print("\northonormal basis w/ v={} e={}".format(v, e2))
    print("  their np.dot(v, e2) = {}".format(np.dot(v, e2)))


    # visualizing change of basis
    print("\nvisualizing change of basis")
    v = np.array([-1, -2])
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    angle = np.pi/6.0
    u1 = np.array([math.cos(angle), math.sin(angle)])
    u2 = np.array([-math.sin(angle), math.cos(angle)])
    print("  v= {} e1= {} e2= {} u1={} u2= {}".format(v, e1, e2, u1, u2))
    p1 = u1*np.dot(u1, v)
    p2 = u2*np.dot(u2, v)
    print("  projection on the new basis: p1= {} p2= {}".format(p1, p2))
    ax = plot_arrows([[orig, u1, 'r', '$\\vec{u}_1$'], [orig, u2, 'r', '$\\vec{u}_2$'], \
        [orig, e1, 'k', '$\\vec{e}_1$'], [orig, e2, 'k', '$\\vec{e}_2$'], \
        [orig, v, 'b', '$\\vec{v}$'],\
        [orig, p1, 'g', '$\\vec{u}_1(\\vec{u}_1 \\cdot\\vec{v}$'],\
        [orig, p2, 'g', '$\\vec{u}_2(\\vec{u}_2 \\cdot\\vec{v}$']], scale=2.2, text_loc=0.2)
    ax.plot([v[0], p1[0]], [v[1], p1[1]], 'm')
    ax.plot([v[0], p2[0]], [v[1], p2[1]], 'm')
    plt.show();


    return None


if __name__ == "__main__":

  print("\nStarting Topic 12.1 Lecture Note for Python code ......")

  main()

  print("\nEnd of Topic 12.1 Lecture Note for Python code ......\n")
