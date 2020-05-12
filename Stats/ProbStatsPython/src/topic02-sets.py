#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def set_elts():

    # set of numbers
    print("\ndefine a set: set([1, 2, 30]))\n  {}".format(set([1, 2, 30])))
    A = {10, 20, 30}
    print("define a set: A = {{10, 20, 30}}\n  {}".format(A))

    # order in set not preserved
    print("\norder not preserved: set(['this', 'that', 'the other']))\n  {}".format(set(['this', 'that', 'the other'])))

    ## element appears only once
    print("\nelement only appears once: set([1, 2, 3, 2]))\n  {}".format(set([1, 2, 3, 2])))

    # create empty set
    A = set()
    print("\ncreate empty set: A = set())\n  A = {}".format(A))

    # add elements to a set
    A.update({0,10})
    print("\nadd elements to set:A.update({{0,10}})):\n  A = {}".format(A))

    # delete an element
    A = {1, 2, 3}
    print("\ndefine a set: A = {{1, 2, 3}}\n  {}".format(A))
    A.remove(2)
    print("delete an element - must exist: A.remove(2)\n  {}".format(A))
    A.discard(4)
    print("delete an element - might not exist: A.discard(4)\n  {}".format(A))
    A.discard(3)
    print("delete an element - might not exist: A.discard(3)\n  {}".format(A))

    # randomly remove from a set
    A = {1, 2, 3}
    print("\ndefine a set: A = {{1, 2, 3}}")
    A.pop()
    print("remove an element randomly: A.pop()\n  {}".format(A))
    A = {-1, 2, 1, 3}
    print("\ndefine a set: A = {{-1, 2, 1, 3}}")
    A.pop()
    print("remove an element randomly: A.pop()\n  {}".format(A))

    # sort a given set
    A = {1, 10, 4, -9, 7, 8, -6, 3, 2}
    print("\ndefine a set: A = {{1, 10, 4, -9, 7, 8, -6, 3, 2}}")
    print("sort a given set: sorted(A)\n  {}".format(sorted(A)))

    # define a set w/ tuple
    A = set([(1,2),(1,3),(3,1)])
    print("\ndefine a set by converting a list w/ tuple: set([(1,2),(1,3),(3,1)])\n  {}".format(A))
    print("\ndefine a set w/ list - not allowed: set([[1,2],[1,3],[3,1]])".format(A))


    return None


def set_operate():

    # define a set w/ a given range
    A = set(range(0,3))   # all integers between 0 and 2
    B = set(range(0,6,2)) # even integers between 0 and 2
    C = set(range(0,6))   # all integers between 0 and 5
    print("\ndefine a set w/ range: A = set(range(0,3))\n  {}".format(A))
    print("define a set w/ range: B = set(range(0,6,2))\n  {}".format(B))
    print("define a set w/ range: C = set(range(0,6))\n  {}".format(C))
    input("\nPress enter to continue...")


    # element existence checking
    print("\nchecking existence of the element in set:\n    1 in A -> {},   1 in B -> {},   3 not in B -> {}".format(1 in A, 1 in B, 3 not in B))

    print("\nchecking subset existence of a set:\n   A.issubset(C)  -> {},     A <= C --> {}".format(A.issubset(C), A<=C))

    print("\nchecking superset existence of a set:\n   C.issuperset(B)  -> {},     C >= B --> {}".format(C.issuperset(B),C>=B))

    print("\nunion sets:\n   A.union(B)  -> {},      A | B --> {}".format(A.union(B), A | B))

    print("\nintersect sets:\n   A.intersection(B)  -> {},    A & B --> {}".format(A.intersection(B), A&B))

    print("\nset difference:\n   A.difference(B)  -> {},    A - B --> {}".format(A.difference(B), A-B))

    print("\nset symmetric difference:\n   A.symmetric_difference(B)  -> {},    A^B --> {}".format(A.symmetric_difference(B), A^B))

    input("\nPress enter to continue...")

    return None

import math

def find_prime(k=100):

    I = set(range(2, k+1))

    for j in range(2, int(math.sqrt(k) + 1)):
        I -= set(range(2*j, k+1, j))
        print("iteration={:4d}, remaining={:4d}".format(j, len(I)))

    lstI = sorted(list(I))
    print("\nprime number list:")
    for i in range(len(lstI)):
        if i%20 == 19 or i == len(lstI) - 1:
            print("{:5d}".format(lstI[i]))
        else:
            print("{:5d}".format(lstI[i]), end='')

    input("\nPress enter to continue...")

    return None

def cartesian():
    """Computing the Cartesian product
    """
    A = set(['a', 'b', 'c'])
    B = set([1, 2])

    C = set()
    for x in A:
        for y in B:
            C.add((x, y))
    
    print("\nCartesian product:\n")
    print("  A = {}".format(A))
    print("  B = {}".format(B))
    print("  C = {}".format(C))
    
    return None


def main():
    """The code to set and test set related function and operators
    """

    set_elts()

    set_operate()
    
    # finding primes
    find_prime(500)

    # Cartesian product
    cartesian()

    
    return None



if __name__ == "__main__":

    print("\nStarting Topic 02 - sets Python code ...\n")

    main()

    print("\n\nEnd of Topic 02 - sets Python code ...\n")