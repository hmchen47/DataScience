#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def conditional__probability(rA, wA, rB, wB):
    """compute conditional probability

    P(white | A) = 

    Arguments:
        rA {int} -- number of red balls in A urn
        wA {int} -- number of white balls in A run
        rB {int} -- number of red balls in B urn
        wB {int} -- number of white balls in B urn
    """
    return (wA/(rA+wA)/2)/(wA/(rA+wA)/2 + (wB/(rB+wB)/2))
    # return 1 / (1 + wB * (rA + wA) / wA / (rB + wB))


def main():

    rA1, wA1, rB1, wB1 = 1., 2., 2., 1.
    rA2, wA2, rB2, wB2 = 2., 4., 3., 3.
    rA3, wA3, rB3, wB3 = 1., 3., 5., 2.
    rA4, wA4, rB4, wB4 = 2., 1., 6., 4.

    print("\nP(white | A) w/ (rA, wA, rB, wB) = ({}, {}, {}, {}): {}".format(\
        rA1, wA1, rB1, wB1, conditional__probability(rA1, wA1, rB1, wB1)))
    print("\nP(white | A) w/ (rA, wA, rB, wB) = ({}, {}, {}, {}): {}".format(\
        rA2, wA2, rB2, wB2, conditional__probability(rA2, wA2, rB2, wB2)))
    print("\nP(white | A) w/ (rA, wA, rB, wB) = ({}, {}, {}, {}): {}".format(\
        rA3, wA3, rB3, wB3, conditional__probability(rA3, wA3, rB3, wB3)))
    print("\nP(white | A) w/ (rA, wA, rB, wB) = ({}, {}, {}, {}): {}".format(\
        rA4, wA4, rB4, wB4, conditional__probability(rA4, wA4, rB4, wB4)))

    return None

if __name__ == "__main__":

    print("\nEntering Topic 6 Conditional probability HW ...")

    main()

    print("\nEnd Topic 6 Conditional probability HW ...\n")

