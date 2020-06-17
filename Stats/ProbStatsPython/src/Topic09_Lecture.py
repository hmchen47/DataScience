#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pandas
import scipy.stats as stat
import matplotlib.pyplot as plt 

from matplotlib import axes
from math import pi, e

def uniform_pdf_cdf(width):
    """plot uniform PDF, CDF

    Args:
        width (int): width of uniform 
    """

    a, b = 0, width
    xrange = np.linspace(0, 10, 101)
    cdf = [0 for z in xrange if z<=a]
    cdf.extend([(z-a)/(b-a) for z in xrange if z>a and z<=b])
    cdf.extend([1 for z in xrange if z>b])

    # plot the PDF
    if b != a:
        plt.plot([a+(b-a)*(1/10) for i in range(11)], [1/(b-a)]*11,\
            linewidth=3.0, label="PDF")
        plt.plot([a, a], [0, 1/(b-a)], 'g--',
            [b, b], [0, 1/(b-a)], 'g--', 
            [a, b], [1/(b-a), 1/(b-a)], 'g--', linewidth=2.0)
        
    # plot the CDF
    plt.plot(xrange, cdf, 'r', linewidth=3.0, label='CDF')

    plt.title('PDF and CDF of U({}, {})'.format(a, b))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return None


def uniform_pdfcdf(prob):
    """plot CDf & PDF of U(a, b)

    Args:
        prob (float): probability of a uniform distribution, 0.0 <= prob <= 1.0
    """

    # for the PDF
    x_support = np.linspace(0, 1, 101)
    y = [1]*101

    plt.subplot(2, 1, 1)
    plt.plot(x_support, y, 'b', linewidth=3.0, label='PDF')
    plt.plot([0, 0], [0, 1], 'g--',
        [1, 1], [0, 1], 'g--', linewidth=2.0)
    
    plt.gca().fill_between(x_support, y, where= x_support<=prob, facecolor='cyan')
    plt.xlim([-0.2, 1.2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('PDF')

    # for CDF
    x_range = np.linspace(-0.5, 1.5, 201)
    cdf = [0 for z in x_range if z<0]
    cdf.extend([z for z in x_range if z>=0 and z<=1])
    cdf.extend([1 for z in x_range if z>1])

    plt.subplot(2, 1, 2)

    plt.plot(x_range, cdf, 'r', linewidth=3.0, label='CDF')
    plt.plot(prob, prob, 'bo', [0, 0], [0, 1], 'g--', [1, 1], [0, 1], 'g--', 
        [-1, prob], [prob, prob], 'm--', [prob, prob], [0, prob], 'm--', linewidth=2.0)

    plt.xlim([-0.2, 1.2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return None

def plot_expon(lam, x_max, CDF=False, sampleSize=100, hist=True):
    """plot Exponential distribution w/ given lambda

    Args:
        lam (float): parameter lambda, range = (0.01, 10)
        x_max (int): range of plotting exponential dist, range= (10, 100)
        CDF (bool, optional): plot CDF or not. Defaults to False.
        sampleSize (int, optional): sample size to take. Defaults to 100.
        hist (bool, optional): plot histogram or not. Defaults to True.
    """
    x = np.arange(0, x_max, x_max/1000)
    y = lam*np.exp(-lam*x)
    z = 1 - np.exp(-lam*x)

    # plot PDF
    plt.plot(x, y, linewidth=3.0, label='PDF')

    # plot CDF
    if CDF:
        plt.plot(x, z, 'r', linewidth=3.0, label='CDF')
    if hist:
        samples = stat.expon.rvs(1/lam, size=sampleSize)
        plt.hist(samples, bins=30, density=True)
    
    plt.xlim([0, x_max])
    plt.title("Exponential({})".format(lam))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return None

def expon_approx(n, p):
    """Plot geomentic distribution and its exponential distribution approximation   

    Args:
        n (int): number of failed flip w/ last success
        p (float): probability of success flip
    """
    x = np.arange(n+1)
    x2 = np.linspace(0, n, num=10*n)
    y = [((1-p)**(z))*p for z in x]

    lam = p
    y0 = lam*np.exp(-lam*x2)
    plt.plot(x2, y0, label='Exponential({})'.format(lam))
    plt.plot(x, y, 'r', label='Geometric({})'.format(p))

    plt.title("Exponential Approximation of Geometric({:.2f})".format(p), fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return None


def main():

    # set the size and properties o the plots when using matplotlib.pyplot
    plt.style.use([{
        "figure.figsize": (12, 9),
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "xx-large",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # "ytick.major.right": False,
        # "ytick.major.top": False
    }, 'seaborn-poster'])

    # uniform distribution
    # width = [0, 10]
    width = 7
    # uniform_pdf_cdf(width)

    # prob = [0.0, 1.0]
    prob = 0.55
    # uniform_pdfcdf(prob)

    # exponential distribution
    # lam =[0.01, 10.0] x_max=(10, 100), sampleSize=(1, 1000)
    lam, x_max, cdf, size, hist = 1.0, 10, True, 1000, True
    plot_expon(lam, x_max, CDF=cdf, sampleSize=size, hist=hist)

    # n = (2, 100), p=(0.0, 1.0)
    n, p = 10, 0.5
    expon_approx(n, p)

    return None


if __name__ == "__main__":

    print("\nStarting Topic 09 Lecture Python code ....")

    main()

    print("\nEnd Topic 09 Lecture Python code ....\n")


