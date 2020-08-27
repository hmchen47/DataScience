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


def normal_mean(n, r, fname=None):
    """computeand plot the sample mean for Normal distribution

    Args:
        n (int): sample size
        r (int): number of experiments
    """
    plt.figure(figsize=(14, 7))
    plt.title('Histogram of sample means w/ sample size n={:d} r={:d}'.format(n, r), fontsize=15)
    plt.xlabel('$\overline{X}$', fontsize=15)
    plt.ylabel('frequency', fontsize=15)
    plt.grid()

    s = r

    x = np.linspace(-1, 1, 1000)
    y = [norm.pdf(i,0,1) for i in x]
    # y = [uniform.pdf(i, 0, 1) for i in x]
    plt.plot(x, y)

    X = np.random.uniform(0, 1, [n, s])
    M = np.sum(X, axis=0)/n
    plt.hist(M, bins=40, density=1)

    if fname != None:
        file = '../img/{}'.format(fname)
        plt.savefig(file)

    plt.show()

    return None

def normal_variance(n, df, r):
    """compute and plot normal variance

    Args:
        n (int): sample size
        df (int): 0 - raw variance; 1 - unbiased variance
        r (int): number of experiments
    """
    plt.figure(figsize=(15, 7))
    plt.xlim([0, 3])

    X = np.random.normal(0, 1, [n, r])
    V = np.var(X, axis=0, ddof=df)
    v = np.mean(V)

    plt.plot([v, v], [0, 3], 'r--', linewidth=2.0)
    plt.hist(V, bins=60, density=1)

    plt.plot([1, 1], [0, 3], 'g:', linewidth=2.0)
    plt.ylabel('frequency', fontsize=15)

    return None

def raw_variance(n, r):
    """plot raw variance

    Args:
        n (int): sample size
        r (int): number of experiments
    """
    normal_variance(n, 0, r)
    plt.title('Histogram of "raw" sample variance w/ sample size n={:d} and r= {:d}'\
        .format(n, r), fontsize=15)
    plt.xlabel('$s^2$', fontsize=15)

    # plt.show()

    return None


def unbiased_variance(n, r):
    """compute and plot unbiased variance

    Args:
        n (int): sample size
        r (int): number of experiments
    """
    normal_variance(n, 1, r)
    plt.title('Histogram of unbiased sample variance with sample size n={:d} and r={:d}'\
        .format(n, r), fontsize=20)
    plt.xlabel('$s^2$', fontsize=15)


def normal_sd(n, s):
    """compute and plot standard deviation for normal distr

    Args:
        n (int): sample size
        s (int): number of experiments
    """

    plt.figure(figsize=(15, 7))
    plt.xlim([0, 3])
    plt.title("Histogram of sample standard deviation with sample size n={:d} and r={}"\
        .format(n, s), fontsize=20)
    plt.xlabel('$\widehat{\sigma}$', fontsize=15)

    X = np.random.normal(0, 1, [n, s])
    V = np.sqrt(np.var(X, axis=0, ddof=1))
    v = np.mean(V)

    plt.plot([v, v], [0, 3], 'r--', linewidth=2.0)
    plt.hist(V, bins=60, density=1)

    plt.plot([1, 1], [0, 3], 'g:', linewidth=2.0)
    plt.ylabel('Frequency', fontsize=15)
    plt.grid()

    plt.show()

    return None




def main():

    # implement sample mean
    # sample size: n = (10, 1000), experiments: r = (1, 10)
    n, r = 500, 100
    # sample_mean(n, r)

    # distribution of the sample mean
    # sample size: n = (3, 30)
    n, r = 5, 3000
    normal_mean(n, r, 't11-04a.png')

    n, r = 50, 3000
    normal_mean(n, r, 't11-04b.png')
    
    n, r = 5, 400
    normal_mean(n, r, 't11-04c.png')
    
    n, r = 50, 400
    normal_mean(n, r, 't11-04d.png')

    # raw variance estimator
    # sample size: n = (2, 20),  degree of freedom df = 0
    # n, r = 3, 500
    # raw_variance(n, r)
    # plt.savefig('../img/t11-02a.png')
    # plt.show()

    # n, r = 30, 500
    # raw_variance(n, r)
    # plt.savefig('../img/t11-02b.png')
    # plt.show()

    # n, r = 3, 3000
    # raw_variance(n, r)
    # plt.savefig('../img/t11-02c.png')
    # plt.show()

    # n, r = 30, 3000
    # raw_variance(n, r)
    # plt.savefig('../img/t11-02d.png')
    # plt.show()


    # unbiased variance
    # sample size: n = (2, 20),  degree of freedom df = 1
    # n, r = 3, 500
    # unbiased_variance(n, r)
    # plt.savefig('../img/t11-02a.png')
    # plt.show()

    # n, r = 30, 500
    # unbiased_variance(n, r)
    # plt.savefig('../img/t11-02c.png')
    # plt.show()

    # n, r = 3, 3000
    # unbiased_variance(n, r)
    # plt.savefig('../img/t11-02b.png')
    # plt.show()

    # n, r = 30, 3000
    # unbiased_variance(n, r)
    # plt.savefig('../img/t11-02d.png')
    # plt.show()

    # estimating standard deviation
    # sample size: n = (2, 10)
    # n, s = 10, 1000000
    # normal_sd(n, s)


    return None


if __name__ == "__main__":

    print("\nStarting the Lecture Python code for Topic 11 .......")

    main()

    print("\nEnd of the Lecture Python code for Topic 11 .......\n")

