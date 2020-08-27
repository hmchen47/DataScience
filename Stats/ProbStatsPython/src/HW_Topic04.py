#!/usr/bin/env python3
# -*- coding: utf--8 -*-

import sys
import numpy as np
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

    comp = set()
    for x in range(1, n):
        for new in compositions(k-1, n-x, debug):
            # if ((x,)+new) not in comp:
            comp.add((x,)+new)
            if debug: print("n= {}, new_comp = {}, \t comp= {}".format(x, new, comp))
    return comp

import math

def binom(n, k):
    """Binomial formula: Binom(n, k) = n!/(k! * (n-k)!)

    Arguments:
        n {int} -- an integer as the objects to choose
        k {int} -- an integer as the the number of objects to choose 
    """

    return math.factorial(n)/math.factorial(k)/math.factorial(n-k)


def composition_formula(k, n):
    """take two positive integers, k an dn and return the number of k-compositions of n

    using binomial formula C(n-1, k-1) to get the number of ways to composite n

    Arguments:
        k {int} -- a positive integer that the number of positive integers sum to n
        n {int} -- a positive integer as the target sum
    """

    return (len(compositions(k, n)), int(binom(n-1, k-1)))


def constrained_compositions(n, cond, debug=False):
    """constrain a compositions of a given number w/ the condition

    Arguments:
        n {int} -- an integer for
        cond {list} -- a list of conditions that the composition not greater than
    """
    k = len(cond)

    comps = compositions(k, n)

    constrained_comps = set()
    for comp in comps:
        if debug: print(comp, tuple(cond), comp<=tuple(cond))
        if all([a <= b for a, b in zip(comp, cond)]):
            constrained_comps.add(comp)

    return constrained_comps


def main():
    """main function to consolidate the individual functions for the Assignment problems"""

    print("\nInteger Compositions ...")
    print("\nProblem 1 ...")

    k, n = 2, 8
    # k, n = 3, 4
    k, n = 2, 5

    func_out = compositions(k, n, False)
    print("all possible combinations: ")
    print_compositions(func_out)
    print("\nActual Output from combinations({}, {}):\n{}".format(k, n, func_out))


    print("\nProblem 2 ...")

    k, n = 3, 4
    k, n = 4, 12
    k, n = 9, 16

    print(len(compositions(k, n)))
    print(int(binom(n-1, k-1)))
    print(composition_formula(k, n))


    print("\nProblem 3 ...")

    n, cond = 7, [3, 2, 5]
    n, cond = 7, [1,4,4]
    n, cond = 8, [3,2,4]
    n, cond = 20,[12, 15, 5]

    func_out = constrained_compositions(n, cond)

    print("all possible combinations: ")
    print_compositions(func_out)
    print("\nActual Output from compositions({},{}) w/ len= {}:\n  {}".format(len(cond), n, len(func_out), func_out))

    # input("Press Enter to continue...")


    return None


if __name__ == "__main__":

    print("\nEntering Topic 4: Permutations & Combinations Assignment ...")

    main()

    print("\nEnd Topic 4: Permutations & Combinations Assignment ...\n")