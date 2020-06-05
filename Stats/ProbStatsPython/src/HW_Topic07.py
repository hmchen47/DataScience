#!/usr/bin/env python3
# -*- coding: utf-8 -8-

import numpy as np
import statistics as stat

def median_cal(p):
    """Compute the median of a given list

    Args:
        p (list): a list of probabilities
    """
    cum = 0
    for idx in range(len(p)):
        cum += p[idx]
        if cum == 0.5:
            return idx+1.5
        elif cum > 0.5:
            return idx+1

def sample_median(n, pmf):
    """return median of n generated random numbers w/ given probability

    Args:
        n (int): number of random values
        pmf (list): a list of probability distribution
    """
    # generate the list of random values
    seq = np.random.choice(range(1, len(pmf)+1), size=n, p=pmf)
    # print("\nseq = {}".format(seq))

    return stat.median(seq)

def expected_cal(pmf):
    exp = 0.0
    for i, val in enumerate(pmf):
        exp += val *(i+1)

    return exp


def average_sample_median(n, pmf):
    """return the average M_n of 100 samples of size n sampled from the distribution

    Args:
        n (int): size n sampled from the distributed pmf
        pmf (list): probability density 
    """
    medians = [0.0] * 1000
    for i in range(1000):
        medians[i] = sample_median(n, pmf)

    return stat.mean(medians)


if __name__ == "__main__":

    print("\nStarting Topic 7 HW Python code .....")

    p1 = [0.1, 0.2, 0.1, 0.3, 0.1, 0.2]
    p2 = [0.99, 0.01]
    p3 = [0.12,0.04,0.12,0.12,0.2,0.16,0.16,0.08]
    print("median of {}: {}".format(p1, median_cal(p1)))
    print("median of {}: {}".format(p2, median_cal(p2)))
    print("median of {}: {}".format(p3, median_cal(p3)))

    input("\nPress Enter to continue ........................")

    print(sample_median(10, [0.1, 0.2, 0.1, 0.3, 0.1, 0.2])) 
    print(sample_median(10, [0.1, 0.2, 0.1, 0.3, 0.1, 0.2]))

    print(sample_median(5, [0.3,0.7]))
    print(sample_median(5, [0.3,0.7]))

    print(sample_median(10,[0.1,0.2,0.3,0.2,0.2]))
    print(sample_median(10,[0.1,0.2,0.3,0.2,0.2]))

    print(sample_median(25,[0.2,0.1,0.2,0.3,0.1,0.1]))
    print(sample_median(25,[0.2,0.1,0.2,0.3,0.1,0.1]))

    print(sample_median(10, [0.12,0.04,0.12,0.12,0.2,0.16,0.16,0.08]))
    print(sample_median(10, [0.12,0.04,0.12,0.12,0.2,0.16,0.16,0.08]))


    input("\nPress Enter to continue ........................")

    print(expected_cal([0.25,0.25,0.25,0.25]))
    print(expected_cal([0.3,0.4,0.3]))
    print(expected_cal([0.12, 0.04, 0.12, 0.12, 0.2, 0.16, 0.16, 0.08]))

    input("\nPress Enter to continue ........................")

    print("\naverage_sample_median(10,[0.2,0.1,0.15,0.15,0.2,0.2])= {}".format(\
        average_sample_median(10,[0.2,0.1,0.15,0.15,0.2,0.2])))
    print("\naverage_sample_median(10,[0.2,0.1,0.15,0.15,0.2,0.2])= {}".format(\
        average_sample_median(10,[0.3,0.4,0.3])))
    print("\naverage_sample_median(10,[0.2,0.1,0.15,0.15,0.2,0.2])= {}".format(\
        average_sample_median(10,[0.99,0.01])))
    print("\naverage_sample_median(10,[0.2,0.1,0.15,0.15,0.2,0.2])= {}".format(\
        average_sample_median(20,[0.4,0.6])))
    print("\naverage_sample_median(10,[0.2,0.1,0.15,0.15,0.2,0.2])= {}".format(\
        average_sample_median(200,[0.1,0.2,0.3,0.1,0.1,0.2])))
    print()
    print("\naverage_sample_median(10,[0.2,0.1,0.15,0.15,0.2,0.2])= {}".format(\
        average_sample_median(100,[0.12, 0.04, 0.12, 0.12, 0.2, 0.16, 0.16, 0.08])))


    print("\nEnf Topic 7 HW Python code .....\n")



