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


def normal_mean(n):
    """computeand plot the sample mean for Normal distribution

    Args:
        n (int): sample size
    """
    plt.figure(figsize=(14, 7))
    plt.title('Histogram of sample means w/ sample size n={:d}'.format(n), fontsize=15)
    plt.xlabel('$\overline{X}$', fontsize=15)
    plt.ylabel('frequency', fontsize=15)
    plt.grid()

    s = 100000

    x = np.linspace(-4, 4, 1000)
    y = [uniform.pdf(i, 0, 1) for i in x]
    plt.plot(x, y)

    X = np.random.uniform(0, 1, [n, s])
    M = np.sum(X, axis=0)/n
    plt.hist(M, bins=40, density=1)

    plt.show()

    return None


def main():

    # implement sample mean
    # sample size: n = (10, 1000), experiments: r = (1, 10)
    n, r = 500, 100
    # sample_mean(n, r)

    # distribution of the sample mean
    # sample size: n = (3, 30)
    n = 3
    normal_mean(n)

    return None


if __name__ == "__main__":

    print("\nStarting the Lecture Python code for Topic 11 .......")

    main()

    print("\nEnd of the Lecture Python code for Topic 11 .......\n")

