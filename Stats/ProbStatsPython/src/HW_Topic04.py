#!/usr/bin/env python3
# -*- coding: utf--8 -*-

import sys
import numpy as nProblem
import scipy as sp
from scipy.special import *

def print_compositions(func_out):
    for x in func_out:
        string = ""
        for i in x:
            string = string + str(i)+ " + "
        print(string[0:-2])
    return None

def compositions(k, n, debug=False):
    """list all possible combinations of k positive integers that sum to n

    A k-compsotion of an integer n is a k-tuple integer that sum to n.

    the function takes two natural numbers k and n as input and returns the set of all
    tuples of size k that sum to n

    Arguments:
        k {int} -- a natural number that k integers sum to n
        n {int} -- a natural number that 
    """
    if k==1:
        return [(n,)]
    # elif n-k < 1:
    #     return [tuple()]

    comp = []
    for x in range(1, n):
        for new in compositions(k-1, n-x, debug):
            if ((x,)+new) not in comp:
                comp.append((x,)+new)
            if debug: print("n= {}, new_comp = {}, \t comp= {}".format(x, new, comp))
    return comp



def main():
    """main function to consolidate the individual functions for the Assignment problems"""

    print("\nInteger Compositions ...")
    print("\nProblem 1 ...")

    k, n = 2, 8

    func_out = compositions(k, n, False)
    print("all possible combinations: ")
    print_compositions(func_out)
    print("\nActual Output from combinations({}, {}):\n{}".format(k, n, func_out))

    # input("Press Enter to continue...")


    return None


if __name__ == "__main__":

    print("\nEntering Topic 4: Permutations & Combinations Assignment ...")

    main()

    print("\nEnd Topic 4: Permutations & Combinations Assignment ...\n")