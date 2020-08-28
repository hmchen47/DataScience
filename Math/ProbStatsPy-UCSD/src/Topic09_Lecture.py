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


def plot_normal(mu, var, CDF=False):
    """plot Normal distribution

    Args:
        mu (float): mean of Gaussian distribution
        var (float): variance of Gaussian distribution
        CFD (bool, optional): plot CDF or not. Defaults to False
    """
    x = np.linspace(-50, 50, 1001)
    sig = var**0.5
    pdf_norm = stat.norm.pdf(x, mu, sig)

    plt.plot(x, pdf_norm, 'b', linewidth=3.0, label="CDF")
    if CDF:
        cdf_norm = stat.norm.cdf(x, mu, sig)
        plt.plot(x, cdf_norm, 'r', linewidth=3.0, label='CDF')
    
    y0 = (1/(sig*np.sqrt(2*pi)))*np.exp(-0.5)
    ym = 1/(sig*np.sqrt(2*pi))

    plt.plot([mu-sig, mu-sig], [0, y0], 'm--', linewidth=2.0)
    plt.plot([mu+sig, mu+sig], [0, y0], 'm--', linewidth=2.0, label='$\mu\pm\sigma$')
    plt.plot([mu, mu], [0, ym], 'g--', linewidth=2.0, label=r'$\mu$')

    plt.title('PDF of N({}, {})'.format(mu, var))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return None


def norm_approx(n, p):
    """Normal approximation of Binomial distribution

    Args:
        n (int): number of binary flips
        p (float): probability of each binary flip w/ success
    """
    x = np.arange(0, n+1)
    x_in = np.linspace(0, n+1, 1001)
    prob_binom = stat.binom.pmf(x, n, p)
    stddev = (n * p * (1 - p))**0.5
    prob_norm = stat.norm.pdf(x_in, n*p, stddev)

    plt.plot(x_in, prob_norm, 'r', linewidth=2.0, label="N({:0.2f}, {:0.2f})"\
        .format(n*p, stddev))
    plt.plot(x, prob_binom, '-', linewidth=2.0, label="Bin({:d}, {:0.2f})"\
        .format(n, p))
    
    plt.title('Normal Approximation of Binomial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    prob_norm = stat.norm.pdf(x, n*p, stddev)
    print("\n|| Prob_Normal - Prob_Binomail ||\u2081 = \n{}".format(abs(prob_norm - prob_binom)))

    return None

# construct_line = lambda x,y,theta: [[x-0.5*np.cos(theta), x+0.5*np.cos(theta)], [y-0.5*np.sin(theta), y+0.5*np.sin(theta)]]
construct_line = lambda x,y,theta: [[x-0.5*np.cos(theta), x+0.5*np.cos(theta)], [y-0.5*np.sin(theta), y+0.5*np.sin(theta)]]
def buffon_pi(size, k, c_update=False):
    """two parallel lines on a plane w/ distance 1 unit. w/o loss of generality, one line as y=0
    and other line as y=1.  throwing a unit-length needle onto the floor in such a way that the
    y-coordinate of its midpoint is uniform over [0, 1] and the angle made by the needle w/ the
    positive x-axis is uniform over [0, pi]. it can be shown that the probability that the needle
    intersects any of the parallel lines is 2/pi.  compute the probability empirically in order
    to estimate the value of pi

    Args:
        size (int): number of throw
        k (int): [description]
        c_update (bool, optional): continuous updated or not. Defaults to False.
    """
    cnt = 0     # to count the number of needle intersecting one or both of the parallel lines
    y_center = (np.random.uniform(0, 1) for _ in range(size+k))
    x_center = (np.random.uniform(0, 1) for _ in range(size+k))
    theta_sample = (np.random.uniform(0, np.pi) for _ in range(size+k))

    for x, y, theta, _ in zip(x_center, y_center, theta_sample, range(size)):
        X, Y = construct_line(x, y, theta)
        if Y[0]<0 or Y[1]>1:
            cnt += 1
    print("\nThe estimate of pi based on {} samples is: {}"\
        .format(size, 2*size/cnt))
    
    plt.plot([-0.3, 1.3], [0, 0], 'k', [-0.3, 1.3], [1, 1], 'k', linewidth=5)
    for x,y,theta in zip(x_center, y_center, theta_sample):
        X, Y = construct_line(x, y, theta)
        plt.plot(X, Y)
        plt.xlim([-0.3, 1.3])
        plt.ylim([-0.3, 1.3])

    plt.show()

    return None

def square_pi(samples):
    """a square w/ aide-length 2a and a inscribed circle w/ radius a.  Pick a point uniformally
    at random from the square, the probability that the selected point also belongs to the 
    inscribed circle is clearly the ratio of the areas of the circle and the square, which
    is given by $\pi a^2 / (2a)^2$. This evaluates to $\pi/4$. estimate the value of pi by 
    determining this probability empirically.  

    Args:
        samples (int): number of samples
    """

    plt.figure(figsize=(12, 12))
    plt.plot([-1, 1], [1, 1], 'k', [-1, 1], [-1, -1], 'k', [-1, -1], [-1, 1], 'k', \
        [1, 1], [-1, 1], 'k', linewidth=5.0)
    x = np.linspace(-1, 1, 101)
    y = (1 - x*x)**0.5
    plt.plot(x, y, 'b', x, -y, 'b', linewidth=5.0)
    samples = zip((np.random.uniform(-1, 1) for _ in range(samples)), \
        (np.random.uniform(-1, 1) for _ in range(samples)))
    cnt = 0
    for x, y in samples:
        if x**2+y**2 <= 1:
            plt.scatter(x, y, c='r', s=12)
            cnt += 1
        else:
            plt.scatter(x, y, c='g', s=12)
    plt.show()
    print("\nEstimated value of pi: {}".format(4*cnt/samples))

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
    # plot_expon(lam, x_max, CDF=cdf, sampleSize=size, hist=hist)

    # n = (2, 100), p=(0.0, 1.0)
    n, p = 10, 0.5
    # expon_approx(n, p)

    # Normal distribution
    # mu = (-25, 25)  var = (0.3, 30)
    mu, var, cdf = 0, 25, False
    # plot_normal(mu, var, CDF=cdf)

    # n = (1, 200)  p = (0.0, 1.0)
    n, p = 50, 0.30
    # norm_approx(n, p)

    # Buffon's needle
    # size = (100, 10000)  k = (10, 100)
    size, k = 10000, 50
    updating = True
    buffon_pi(size, k, c_update=updating)

    # estimate pi w/ circle within a square
    # sample = (10, 1000)
    sample = 500
    square_pi(sample)

    return None


if __name__ == "__main__":

    print("\nStarting Topic 09 Lecture Python code ....")

    main()

    print("\nEnd Topic 09 Lecture Python code ....\n")


