#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import statistics as stat
import matplotlib.pyplot as plt 
import numpy as np 
import math

from scipy.stats import norm, uniform
from math import exp, e, log


def sample_mean(n, r):
    """compute same mean for $\overline{X} = \frac{X_1 + \cdots + X_n}{n}$
    which converges to the distribution mean $\mu$

    Args:
        n (int): sample size
        r (int): number of experiments
    """
    plt.figure(figsize=(15, 7))
    plt.xlim([1, n])
    plt.ylim([-1, 1])
    plt.grid()

    x = range(1, n+1)
    z = 1.0/np.sqrt(x)
    plt.plot(x, z, 'k--')
    plt.plot(x, np.negative(z), 'k--')

    for i in range(r):
        y = np.random.normal(0, 1, n)
        m = np.divide(np.cumsum(y), x)
        plt.plot(x, m, alpha=0.5)

    plt.show()

    return None


def main():

    # implement sample mean
    # sample size: n = (10, 1000), experiments: r = (1, 10)
    n, r = 500, 100
    sample_mean(n, r)

    return None


if __name__ == "__main__":

    print("\nStarting the Lecture Python code for Topic 11 .......")

    main()

    print("\nEnd of the Lecture Python code for Topic 11 .......\n")

