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


if __name__ == "__main__":

    print("\nStarting Topic 7 HW Python code .....")

    p1 = [0.1, 0.2, 0.1, 0.3, 0.1, 0.2]
    p2 = [0.99, 0.01]
    p3 = [0.12,0.04,0.12,0.12,0.2,0.16,0.16,0.08]
    print("median of {}: {}".format(p1, median_cal(p1)))
    print("median of {}: {}".format(p2, median_cal(p2)))
    print("median of {}: {}".format(p3, median_cal(p3)))


    print("\nEnf Topic 7 HW Python code .....\n")



