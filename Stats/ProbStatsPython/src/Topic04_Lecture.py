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





    return None


if __name__ == "__main__":

    print("\nStarting Topic 4: Combinatorics Python code ...")

    main()

    print("\nEnd Topic 4: Combinatorics Python code ...\n")