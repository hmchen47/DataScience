#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import itertools


def permute(A, debug=False):
    """Customized permutation by recursion

    Arguments:
        A {list} -- a given list
    """
    if len(A) == 1:
        if debug: print(A, [tuple(A)])
        return [tuple(A)]

    permutations = []

    for x in A:
        for y in permute(A-{x}, debug):
            permutations.append((x,) + y)
            if debug: print(x, y, permutations)
            
    return permutations

def factorial_iter(n):
    """compute factorial w/ iterative

    Arguments:
        n {int} -- the n!
    """
    fact = 1
    for i in range(n):
        fact *= (i+1)

    return fact

def factorial_recursive(n):
    """compute factorial recursively

    Arguments:
        n {int} -- an integer value to compute its factorial
    """
    if n == 0: 
        return 1
    else:
        return n * factorial_recursive(n-1)


def main():

    print("\n... Permutations ...")
    print("\ncustomized permutation\n")
    A = {'a', 'b', 6}
    print("  Permutations of {} w/ length= {}:\n  {}".format(A, len(permute(A)), permute(A, False)))

    print("\nitertools permutation\n")
    A = 'a1c3'
    perm = set(itertools.permutations(A))
    print("  Permutations of {} w/ length= {}:\n  {}".format(A, len(perm), perm))

    input("\nPress Enter to continue ...")

    print("\n... Factorials ...")
    print("\nbuilt-in math module:  math.factorial(|{}|)= {}".format(A, math.factorial(len(A))))

    print("\niteratively w/ customized function (|{}|): {}".format(A, factorial_iter(len(A))))

    print("\nrecursively w/ customized function (|{}|): {}".format(A, factorial_recursive(len(A))))

    input("\nPress Enter to continue ...")





    return None


if __name__ == "__main__":

    print("\nStarting Topic 4: Combinatorics Python code ...")

    main()

    print("\nEnd Topic 4: Combinatorics Python code ...\n")