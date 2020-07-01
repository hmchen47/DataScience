#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

from math import sqrt, exp, factorial


def plot_markov_chebyshev(mu, sig):
    """plot probability inequalities - Markov & Chebyshev

    Args:
        mu (float): mean of the given probability distribution
        sig (float): standar deviation of the probability distribution
    """

    a_min, a_max = mu, min(mu*20, sig*10)

    a = np.linspace(a_min, a_max, 10001)
    plt.plot(a, mu/a, 'b', linewidth=3.0, label='Markov $\mu/a$')

    b_min = mu+sig
    b= np.linspace(b_min, a_max, 10001)
    plt.plot(b, (sig/(b - mu))**2, 'r', linewidth=3.0, \
        label='Chebyshev $\sigma^2/(a - \mu)^2$')
    plt.plot([mu, mu], [0, 1.1], 'b', [mu+sig, mu+sig], [0, 1.1], 'r')

    plt.title('Markov & Chebyshev bounds on $P(X\geq a)$ for $\mu=$ {:0.1f} and $\sigma=${:0.1f}'\
        .format(mu, sig), fontsize=20)
    plt.xlabel('a')
    plt.ylabel('Probability bounds')
    plt.legend()
    plt.grid()
    plt.show()

    return None
    
def generate_uniform_counts(a, b, k, n):
    """generate samples of empirical mean of k uniform 
    distribution distribution on [a, b]

    Args:
        a (int): lower part of unifrom distribution
        b (int): upper part of uniform distribution
        k (int): number of random variables w/ uniform distribution
        n (int): number of samples
    """
    # generate a k by n matrix of uniform random numbers
    X = np.random.uniform(a, b, [k, n])
    S = np.sum(X, axis=0) / k
    return S

def uniform_plot_hist(s, k, h):
    """plot uniform distribution w/ histogram

    Args:
        s (int): width of uniform distribution
        k (int): number of random variables w/ uniform distribution
        h (int): number of samples
    """
    a, b = s
    if h > 0:
        n = h
        counts = generate_uniform_counts(a, b, k, n)
        plt.hist(counts, bins=30, density=True, label=('Histogram of mean values'))
        plt.xlim([0, 10])
        plt.plot([(a+b)/2, (a+b)/2], [0, 1], 'g--', linewidth=2.0, \
            label='Mean of unifrom distribution')
        
    return None


def uniform_mean_pdf(s, n, h, debug=False):
    """plot the pdf of 1/n(\sum_{i=1}^n X_i), X_i \sim U_{a, b}

    Args:
        s (int): width of uniform distribution
        n (int): number of random variables w/ uniform distribution
        h (int): number of samples
    """
    a, b = s
    d = 10.0/1000;
    x = np.linspace(0.01, 10, 1000)
    plt.close()

    if a < b:
        # ideal uniform distribution w/ given values [a, b]
        y = (1.0 * (x >= a)) * (1.0 * (x <= b)) / (b-a)
        z = y
        for j in range(2, n+1):
            t = [item/(j-1) for item in z for i in range(j-1)]
            z = [0, ] + np.convolve(y, t).tolist()
            z = [i*d for i in z]
            if debug:
                print(len(z), "\n",  z)
                print(np.reshape(z, (1000, j)))
            z = np.sum(np.reshape(z, (1000, j)), axis=1)
        plt.plot(x, z, label='Distribution of Mean')
        uniform_plot_hist(s, n, h)
        plt.title('PDF and histogram of $\overline{X}_n$ with n=%d, s=%d'%(n,h), fontsize = 20)
        plt.xlabel('$\overline{X}_n$')
        plt.ylabel('$f_{\overline{X}_n}(x)$')
        plt.ylim([0, 1.1])
        plt.legend()
        plt.show()

    return None


def generate_exponential_counts(lam, k, n):
    """generate the samples of empirical mean of k exponential distributions 
    w/ parameter lambda

    Args:
        lam (float): parameter for exponential distribution
        k (int): number of r.v. for exponential dist
        n (int): number of samples
    """
    X = np.random.exponential(1.0/lam, [k, n])
    S = np.sum(X, axis=0)/k
    
    return S


def exp_plot_hist(lam, k, h):
    """plot the histogram for the exponential distribution

    Args:
        lam (float): parameter of the exponential distribution
        k (int): number of r.v.s of exponential distribution
        h (int): number of samples
    """
    n = h
    if h > 0:
        counts = generate_exponential_counts(lam, k, n)
        plt.hist(counts, bins=int(h/50), density=True, label='Histogram of mean values')

    return max(counts)


def exponential_mean_pdf(lam, n, h):
    """plot the pdf of 1/n(\sum_{i=1}^n X_i), X_i \sim \Exp_{lam}

    Args:
        lam (float): parameter of exponential distribution
        n (int): number of r.v.s
        h (int): number odf samples
    """
    xmax = exp_plot_hist(lam, n, h)

    d = np.ceil(xmax)/200
    x = np.linspace(0.01, np.ceil(xmax), 1000)
    z = [(lam**n)*((i*(n))**(n-1))*exp(-lam*(i*(n)))/(factorial(n-1))*(n)\
        for i in x]
    # plt.close()

    plt.plot(x, z, label='Distribution of mean')
    plt.plot([1.0/lam, 1.0/lam], [0, np.ceil(max(z)*10)/10], 'r--', linewidth=2.0,\
        label='Mean of exponential distribution')
    plt.title('PDF and Histogram of $\overline{X}_n$ w/ $\lambda$=%.2f, n=%d, s=%d'%(lam, n, h))
    plt.xlabel('$x$')
    plt.ylabel('$f_{\overline{X}_n}(x)$')
    plt.legend()
    plt.show()

    return None


def main():

    plt.style.use([{
        "figure.figsize": (12, 9),
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "xx-large",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }, 'seaborn-poster'])

    # Inequalities
    # mu = (0.5, 20)  sig = (0.5, 20)
    mu, sig = 5, 0.5
    # plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 1
    # plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 5
    # plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 10
    # plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 20
    # plot_markov_chebyshev(mu, sig)


    # Uniform distribution: [a, b]
    # k uniform distributions w/ n repetition
    # width = [a, b], min(a, b) = 0.02, max(a, b) = 9.98
    # n = (1, 20),  h = (1, 10000)
    s, n, h = (2, 8), 5, 10000
    # uniform_mean_pdf(s, n, h, True)
    # uniform_mean_pdf(s, n, h)


    # exponential distribution
    # lam = (0.02, 10), n = (1, 20), h = (1, 10000)
    lam, n, h = .5, 12, 10000
    exponential_mean_pdf(lam, n, h)


    # input("\nPress Enter to continue ......")

    return None


if __name__ == "__main__":

    print("\nStarting Topic 10 Lecture NB .......")

    main()

    print("\nEnd Topic 10 Lecture NB .......\n")