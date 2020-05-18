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

def partial_permute(A, k, debug=False):
    """Compute values of partial permutation

    Arguments:
        n {int} -- an integer for a given groups
        k {int} -- an integer to choose from a group
    """
    if k == 1: return [(x,) for x in A]
    permutations = []

    for x in A:
        if debug: print(x, A, permutations)

        for y in partial_permute(A-{x}, k-1, debug):
            permutations.append((x,)+y)
    return permutations


def combine_recur(A, k, debug=False):
    """Generate a list of combinations

    Arguments:
        A {Set} -- a set of elements to form its combinations
        k {int} -- an integer for the number of elements of each combination
    """
    if k == 1: return [{x} for x in A]

    combinations = []
    for x in A:
        for y in combine_recur(A-{x}, k-1, debug):
            if {x}|y not in combinations:
                combinations.append({x} | y)
            if debug: print(x, y, combinations);

    return combinations


def main():

    # Permutation

    # print("\n... Permutations ...")
    # print("\ncustomized permutation\n")
    # A = {'a', 'b', 6}
    # print("  Permutations of {} w/ length= {}:\n  {}".format(A, len(permute(A)), permute(A, False)))

    # print("\nitertools permutation\n")
    # A = 'a1c3'
    # perm = set(itertools.permutations(A))
    # print("  Permutations of {} w/ length= {}:\n  {}".format(A, len(perm), perm))

    # input("\nPress Enter to continue ...")


    # # Factorial

    # print("\n... Factorials ...")
    # print("\nbuilt-in math module:  math.factorial(|{}|)= {}".format(A, math.factorial(len(A))))

    # print("\niteratively w/ customized function (|{}|): {}".format(A, factorial_iter(len(A))))

    # print("\nrecursively w/ customized function (|{}|): {}".format(A, factorial_recursive(len(A))))


    # # Partial permutation

    # input("\nPress Enter to continue ...")
    # print("\n... Partial Permutation ...")
    # A, k = {1, 2, 3, 4}, 2

    # print("\ncustomized partial permutation function ({}, {}) w/ length= {}:\n  {}".format(A, k, \
    #     len(partial_permute(A, k)), partial_permute(A, k, False)))

    # print("\nbuilt-in math module:  itertools.permutations({}, {}) w/ length= {}\n  {}".format(A, k, \
    #     len(list(itertools.permutations(A, k))), list(itertools.permutations(A, k))))

    # combinations
    input("\nPress Enter to continue ...")
    print("\n... Combinations ...")
    A, k = {'a', 'b', 'c', 'd', 'e'}, 3

    print("\ncustomized combination function ({}, {}) w/ length= {}:\n  {}".format(A, k, \
        len(combine_recur(A, k)), combine_recur(A, k, False)))


    return None


if __name__ == "__main__":

    print("\nStarting Topic 4: Combinatorics Python code ...")

    main()

    print("\nEnd Topic 4: Combinatorics Python code ...\n")