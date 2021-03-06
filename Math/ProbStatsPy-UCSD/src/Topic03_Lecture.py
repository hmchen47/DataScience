#!/usr/bin/env python3
# -* coding: utf-8 -*-

import itertools

def main():

    # size of an object
    NewSet = {1, 2}
    print("\nSize of an object: NewSet, len(NewSet)\n  NewSet= {}, size= {}".format(NewSet, len(NewSet)))

    input("\nPress Enter to continue ...")

    # smallest and greatest elements in an object
    NewSet = {-2,-5,9,2, 1, 3}
    print("\nmax and min of an object: max(NewSet), min(NewSet)\n  NewsSet = {}, max = {}, min = {}".format(NewSet, max(NewSet), min(NewSet)))

    input("\nPress Enter to continue ...")

    # max and min  also applied to strings
    Set = {'c','b','a','z'}
    print("\nmax and min of an object: max(Set), min(Set)\n  NewsSet = {}, max = {}, min = {}".format(Set, max(Set), min(Set)))

    input("\nPress Enter to continue ...")

    # sum of elements of a set
    A = {1, 5, 2, -10, 19}
    print("\nsum of elements of a set\n  A = {}, sum= {}".format(A, sum(A)))

    input("\nPress Enter to continue ...")

    # zip function
    a = [83,59,92]
    b = ['Harry','Paul','Grace']
    print("\nzip of two sets: a= {}, b= {}, zip(a, b) w/ for loop ".format(a, b))
    for a_i, b_i in zip(a, b):
        print("  {} {}".format(a_i, b_i))

    input("\nPress Enter to continue ...")

    # enumerate function
    print("\nenumerate function pproviding an index number")
    for idx, (a_i, b_i) in enumerate(zip(a, b)):
        print("  {} {} {}".format(idx, a_i, b_i))

    input("\nPress Enter to continue ...")

    # set operations
    A = {1, 2, 3}
    B = {1, 3, 5}
    print("\nset operations w/ the sets: A={},    B={}".format(A, B))
    print("  Intersection  (A & B)= {}    w/ |A&B|= {}".format(A&B, len(A&B)))
    print("  Difference    (A - B)= {}    w/ |A-B|= {}".format(A-B, len(A-B)))
    print("  General union (A | B)= {}    w/ |A|B|= {} = |A| + |B| - |A&B|".format(A|B, len(A|B)))
    
    input("\nPress Enter to continue ...")


    # Cartesian products - two ways
    print("\nCartesian product:")
    A = {1, 2, 3}
    B = {4, 5}
    print("\nOrdered pairs in {} x {}".format(A, B))
    print("Cartesian product: \n set([(a,b) for a in A for b in B]) =\n  {}".format(set([(a,b) for a in A for b in B])))
    print("Cartesian product: \n set([i for i in itertools.product(A, B)]) =\n  {}".format(set([i for i in itertools.product(A, B)])))

    # Cartesian power: two ways
    A = {1, 2, 3}
    k = 2

    # initialize every element as a tuple
    print("\nCartesian power:")
    cartesian_powers = [(a, ) for a in A]
    for j in range(k-1):
        cartesian_powers = [ i+(a, ) for i in cartesian_powers for a in A]
    
    print(" {}^{}: {}".format(A, k, cartesian_powers))
    print(" size= {}".format(len(cartesian_powers)))


    return None

if __name__ == "__main__":

    print("\nEntering Topic 3 Lecture Counting ...")

    main()

    print("\nEnd Topic 3 Lecture Counting ...\n")