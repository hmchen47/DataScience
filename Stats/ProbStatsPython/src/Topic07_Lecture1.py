#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def multivariates_dist():

    # starting w/ positive weights that don't sum to 1
    P = np.array([[2.0, 2, 4], [1, 1, 2]])
    P2 = P

    print("\nThe initial weight for a probability: \n{}".format(P))

    # examine the address of P and P2
    print("\nThe original address for numpy array w/ P2 = P:\n  id(P) = {}  id(P2) = {}".format(id(P), id(P2)))
    P[0, 0] = 0
    print("\nDisplay values of P2 w/ assigning value on P[0, 0] = 0: \n{}".format(P2))
    print("\nDisplay the address after changing P[0, 0] value:\n  id(P) = {} id(P2) = {}".format(id(P), id(P2)))

    input("\nPress Enter to continue .............................\n")

    # initial np.array w/ copy than address assignment
    P = np.array([[2.0, 2, 4], [1, 1, 2]])
    P2 = np.copy(P)

    print("\nThe initial weight for a probability again: \n{}".format(P))

    # examine the address of P and P2
    print("\nThe original address for numpy array w/ P2 = np.copy(P):\n  id(P) = {}  id(P2) = {}".format(id(P), id(P2)))
    P[0, 0] = 0
    print("\nDisplay values of P w/ assigning value on P[0, 0] = 0: \n{}".format(P))
    print("\nDisplay values of P2 w/ assigning value on P[0, 0] = 0: \n{}".format(P2))
    print("\nDisplay the address after changing P[0, 0] value:\n  id(P) = {} id(P2) = {}".format(id(P), id(P2)))

    input("\nPress Enter to continue ............................\n")

    # normalizing the weights
    P = np.copy(P2)

    total = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            total += P[i, j]

    print("\nSum of np.array P w/ loops: {}".format(total))
    print("\nSum of np.array P w/ np.sum(P)): {}".format(np.sum(P)))

    # dividing the elements w/ sum
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            P[i, j] /= total
    
    print("\nNormalized P values w/ elementwise operation: \n{}".format(P))

    P2 /= np.sum(P2)
    print("\nNormalized P2 values w/ P2 /= np.sum(P2): \n{}".format(P2))
    print("\nDisplaying the dimensions of an np.array w/ P.shape: {}".format(P.shape))

    input("\nPress Enter to continue ............................\n")


    # assign values for random variables
    x = np.array([1, 2, 6])
    y = np.array([-1, 1])

    # computing marginal 
    Px = [0.]*P.shape[1]
    Py = [0.]*P.shape[0]

    for i in range(len(Px)):
        for j in range(len(Py)):
            Px[i] += P[j, i]
            Py[j] += P[j, i]

    print("\nThe marginal dist w/ loops: \n  Px = {}\n  Py = {}".format(Px, Py))

    Px = np.sum(P, axis=0)
    Py = np.sum(P, axis=1)

    print("\nThe margin dist w/ np.sum(P, axis=0/1): \n  Px= {} \n  Py= {}".format(Px, Py))


    input("\nPress Enter to continue ............................\n")

    return None

def independence_test():

    # two r.v. X and Y independence, the outer product of the marginals should be equal P

    # assign values for random variables
    P = np.array([[2.0, 2, 4], [1, 1, 2]])
    x = np.array([1, 2, 6])
    y = np.array([-1, 1])

    Px = np.sum(P, axis=0)
    Py = np.sum(P, axis=1)

    # the pure python way
    print("\nX, Y independence --> outter product of the margin = P")
    for i in range(Px.size):
        for j in range(Py.size):
            if Px[i]*Py[j] != P[j, i]:
                print("  Px[{:d}]*Py[{:d}] ! = P[{:d}, {:d}] :::: {:5.3f}*{:5.3f} = {:5.3f} ! = {:5.3f}".format(\
                    i, j, j, i, Px[i], Py[j], Px[i]*Py[j], P[j, i]))
                # print("Px[%d]*Py[%d] != P[%d,%d] ::::: %5.3f*%5.3f = %5.3f != %5.3f"%(i,j,j,i,Px[i],Py[j],Px[i]*Py[j],P[j,i]))

    input("\nPress Enter to continue ............................\n")

    # the numpy way
    print("\nIndependence test w/ np.outer(Px, Py).T - P = 0:\n{}".format(np.outer(Px, Py)))

    input("\nPress Enter to continue ............................\n")

    return None



def main():

    # join distribution of two discrete random variables
    multivariates_dist()

    # Testing independence
    independence_test()

    return None


if __name__ == "__main__":

    print("\nStarting Topic 7 Lecture NB 1 Python code ...")

    main()

    print("\nEnd Topic 7 Lecture NB 1 Python code ...\n")

