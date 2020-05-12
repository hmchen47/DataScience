#!/usr/bin/env python3
# -* coding: utf-8 -*-

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


    return None

if __name__ == "__main__":

    print("\nEntering Topic 3 Lecture Counting ...")

    main()

    print("\nEnd Topic 3 Lecture Counting ...")