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


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import pandas as pd

def plot_stirling_approx(n):
    """calculate and plot Stirling approximation

    Arguments:
        n {int} -- compute the factorial n
    """
    n_fact = []
    n_fact_approx = []

    # compute the factorial and Stirling approximation
    for i in range(1, n+1):
        n_fact.append(np.double(math.factorial(i)))
        n_fact_approx.append(np.double(np.sqrt(2*np.pi*i)*(i/np.e)**i))

    plt.figure(figsize=(12, 9))
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()

    plt.title("Comparison btw n! and Stirling approximation", fontsize=20)
    host.semilogy(range(1, n+1), n_fact, linewidth=2, label="Factorial")
    host.semilogy(range(1, n+1), n_fact_approx, linewidth=2, label="Stirling's approximation")
    
    host.set_xlabel("n", fontsize=16)
    host.set_ylabel("n!", fontsize=16)

    error = (np.asarray(n_fact)-np.asarray(n_fact_approx))/np.asarray(n_fact)

    p1, = par1.semilogy(range(1, n+1), error*100, linewidth=2, label="Error %")
    par1.set_ylabel("Error %", fontsize=16)
    par1.axis["right"]
    par1.axis["right"].label.set_color(p1.get_color())
    host.legend(fontsize=14)

    plt.show()

    return None


def factorial_stirling_error(n=20):
    """compute the error of Stirling's approximation and factorial

    Arguments:
        n {int} -- a integer to compute the factorial
    """

    n_fact = [math.factorial(i) for i in range(n)]
    n_fact_approx = [np.double(np.sqrt(2*np.pi*i)*(i/np.e)**i) for i in range(n)]

    error = (np.asarray(n_fact)-np.asarray(n_fact_approx, dtype=np.float))/np.asarray(n_fact)
    data = {"n": range(1, n+1), "n!": n_fact, "Stirling Approximation": n_fact_approx, "Error %": error*100}
    df = pd.DataFrame(data=data, columns=["n", "n!", "Stirling Approximation", "Error %"])
    df.set_index("n")
    print("Comparison btw n! and Stirling Approximation")
    print(df.to_string(index=False))

    return None



def main():

    # Permutation

    print("\n... Permutations ...")
    print("\ncustomized permutation\n")
    A = {'a', 'b', 6}
    print("  Permutations of {} w/ length= {}:\n  {}".format(A, len(permute(A)), permute(A, False)))

    print("\nitertools permutation\n")
    A = 'a1c3'
    perm = set(itertools.permutations(A))
    print("  Permutations of {} w/ length= {}:\n  {}".format(A, len(perm), perm))

    input("\nPress Enter to continue ...")


    # Factorial

    print("\n... Factorials ...")
    print("\nbuilt-in math module:  math.factorial(|{}|)= {}".format(A, math.factorial(len(A))))

    print("\niteratively w/ customized function (|{}|): {}".format(A, factorial_iter(len(A))))

    print("\nrecursively w/ customized function (|{}|): {}".format(A, factorial_recursive(len(A))))


    # Partial permutation

    input("\nPress Enter to continue ...")
    print("\n... Partial Permutation ...")
    A, k = {1, 2, 3, 4}, 2

    print("\ncustomized partial permutation function ({}, {}) w/ length= {}:\n  {}".format(A, k, \
        len(partial_permute(A, k)), partial_permute(A, k, False)))

    print("\nbuilt-in math module:  itertools.permutations({}, {}) w/ length= {}\n  {}".format(A, k, \
        len(list(itertools.permutations(A, k))), list(itertools.permutations(A, k))))

    combinations

    input("\nPress Enter to continue ...")
    print("\n... Combinations ...")
    A, k = {'a', 'b', 'c', 'd', 'e'}, 3

    print("\ncustomized combination function ({}, {}) w/ length= {}:\n  {}".format(A, k, \
        len(combine_recur(A, k)), combine_recur(A, k, False)))

    print("\nbuilt-in math module:  itertools.combinations({}, {}) w/ length= {}\n  {}".format(A, k, \
        len(list(itertools.combinations(A, k))), list(itertools.combinations(A, k))))

    print("\ncounting combinations directly C(n, k) = n!/(k! (n-k)!):\n  C({}, {}) = {}".format(len(A),\
        k, math.factorial(len(A)) / math.factorial(k) / math.factorial(len(A) - k)))

    # concatnate characters
    
    permute_k = partial_permute(A, k)
    permute_k = [''.join(x) for x in permute_k]
    print("\npartial permutations w/ A= {}, k= {} w/ len= {}:\n  {}".format(A, k, len(permute_k), permute_k))

    print("\npartial permutations counting directly P(n, k) = n!/(n-k)!:\n  {}".format(math.factorial(len(A)) / math.factorial(len(A)-k)))

    combine_k = combine_recur(A, k)
    combine_k = [''.join(x) for x in combine_k]
    print("\ncombinations w/ A= {}, k= {} w/ len={}:\n  {}".format(A, k, len(combine_k), combine_k))


    # plotting Sterling approximation
    n = 30
    # plot_stirling_approx(n)

    factorial_stirling_error(n)
   

    return None


if __name__ == "__main__":

    print("\nStarting Topic 4: Combinatorics Python code ...")

    main()

    print("\nEnd Topic 4: Combinatorics Python code ...\n")