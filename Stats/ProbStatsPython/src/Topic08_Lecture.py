#!/usr/bin/env python3
# -*- coding: utf-8 -8-

from scipy.stats import bernoulli, binom, poisson
import matplotlib.pyplot as plt 
import numpy as np 

def gen_plot_bernoulli():

    # generate bernoulli
    print("\nBernoulli pmf w/ scipy.stats.bernoulli.pmf(0, p=0.3): {:.3f}".format(bernoulli.pmf(0, p=0.3)))
    print("\nBernoulli pmf w/ scipy.stats.bernoulli.pmf(range(3), p=0.3): {}".format(bernoulli.pmf(range(3), p=0.3)))
    print("\nBernoulli cdf w/ scipy.stats.bernoulli.cdf([0, 0.5, 1, 1.5], p=0.3): {}".format(bernoulli.cdf([0, 0.5, 1, 1.5], p=0.3)))
    
    # plot Bernoulli 
    plt.stem([-0.2, 0, 1, 1.2], bernoulli.pmf([-0.2, 0, 1, 1.2], p=.3))
    plt.plot(np.linspace(-0.1, 1.1, 1200), bernoulli.cdf(np.linspace(-0.1, 1.1, 1200), p=0.3), 'g')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.2, 1.1])
    plt.show()

    # generate Bernoulli samples
    print("\nBernoulli samples w/ scipy.stats.bernoulli.rvs(size=10, p=.3): {}".format(bernoulli.rvs(size=10, p=.3)))

    plt.hist(bernoulli.rvs(size=10, p=.3), density=True)
    plt.show()


    return None

def plot_binom_pmf(n, p, samples=200, histogram=False):
    """plot pmf of Binomial distribution, Binom(n, p)

    Args:
        n (int): size of Binomial dist
        p (float): probability of success
        samples (int, optional): number of samples. Defaults to 100.
        histogram (bool, optional): plot historgram. Defaults to False.
    """
    k = np.arange(0, n+1)
    prob_binom = binom.pmf(k, n, p)

    plt.plot(k, prob_binom, '-o', color='b')

    if histogram:
        height, y = np.histogram(binom.rvs(size=samples, n=n, p=p), \
            range=(0, n), bins=n+1, normed=True)
        plt.bar(k, height, color='r')

    plt.title('PMF of Bin({:d}, {:.2f})'.format(n, p))
    plt.xlabel('k')
    plt.ylabel('$B_{20, 0.3}(k)$')
    plt.show()

    return None


def gen_plot_binomial():

    # generate Binomial samples
    print("\Binomial samples w/ scipy.stats.binom.rvs(size=50, n=20, p=.4):\n  {}".format(binom.rvs(size=50, n=20, p=.4)))

    n, p, smp, hist = 100, 0.4, 200, True
    plot_binom_pmf(n, p, samples=smp, histogram=hist)
 
    return None


def f_poisson(n, lamb, samples=100, histogram=False):
    """plot Poisson PMF

    Args:
        n (int): size of Poisson distribution
        lambda (float): parameter of Poisson distribution
        samples (int, optional): sample size. Defaults to 100.
        histogram (bool, optional): plot histogram. Defaults to False.
    """
    k = np.arange(0, n+1)
    prob_poisson = poisson.pmf(k, lamb)
    plt.plot(k, prob_poisson, '-o')

    if histogram:
        height, y = np.histogram(poisson.rs(size=samples, mu=lamb),\
            range=(0, n), bins=n+1, normed=True)
        plt.bar(k, height, color='r')

    plt.title('PMF of Poisson({})'.format(lamb))
    plt.xlabel('Number of Events')
    plt.ylabel('Probability of Number of Events')
    plt.show()

    return None


def poisson_approx(n, p):
    """Poisson approximation of the Binomial distribution

    Args:
        n (int): size of Binomial distribution n >> 1
        p (float): probability of Binomial distribution, p << 1
    """
    k = np.arange(0, n+1)
    x = np.linspace(0, n+1, 1000)
    lam = n*p
    stddev = lam**0.5
    pmf_poisson = poisson.pmf(k, lam)
    pmf_binom = binom.pmf(k, n, p)

    plt.plot(k, pmf_poisson, 'r', label='Poisson({:0.2f})'.format(lam))
    plt.plot(k, pmf_binom, 'b-', label='Bin({:d}, {:0.2f})'.format(n, p))
    plt.title("Poisson Approximation of Binomial")
    plt.xlabel('n')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("|| pmf_poisson - pmf_binom ||\u2081 = {}".format(sum(abs(pmf_poisson - pmf_binom))))

    return None

def gen_plot_poisson():

    # plot Poisson PMF
    # n = (0, 50), samples = (1, 1000) lambda = (0.0, 30.0)
    n, samples, lamb = 40, 5000, 20.0
    f_poisson(n, lamb, samples=samples)

    # Poisson approximation of the Binomial Distribution
    # n = (2, 1000), p = (0.0, 0.2, 0.001)
    n, p = 30, 0.15
    poisson_approx(n, p)

    return None


def main():

    # properties for plots
    plt.style.use([{
        "figure.figsize": (12, 9),         # figure size
        "xtick.labelsize": "large",     # font size of the X-tick
        "ytick.labelsize": "large",     # font size of the Y-tick
        "legend.fontsize": "x-large",   # font size of legend
        "axes.labelsize": "x-large",    # font size of label
        "axes.titlesize": "xx-large",   # font size of title
        "axes.spines.top": False,
        "axes.spines.right": False,
    }, 'seaborn-poster'])

    # Bernoulli
    # gen_plot_bernoulli()

    # Binomial distribution
    # gen_plot_binomial()

    # Poisson distribution
    gen_plot_poisson()


    return None

if __name__ == "__main__":

    print("\nStarting Topic 8 Discrete Distribution Family Lecture Python Code ...")

    main()

    print("\nEnd of Topic 8 Discrete Distribution Family Lecture Python Code ...\n")




