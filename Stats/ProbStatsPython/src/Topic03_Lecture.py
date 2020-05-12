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

    return None

if __name__ == "__main__":

    print("\nEntering Topic 3 Lecture Counting ...")

    main()

    print("\nEnd Topic 3 Lecture Counting ...")