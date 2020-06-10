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
    plt.ylabel('$B_{20, 0.3}(k)')
    plt.show()

    return None


def gen_plot_binomial():

    # generate Binomial samples
    print("\Binomial samples w/ scipy.stats.binom.rvs(size=50, n=20, p=.4):\n  {}".format(binom.rvs(size=50, n=20, p=.4)))

    n, p, smp, hist = 100, 0.4, 200, True
    plot_binom_pmf(n, p, samples=smp, histogram=hist)


    input("\nPress Enter to continue ....................................")

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
    gen_plot_binomial()


    return None

if __name__ == "__main__":

    print("\nStarting Topic 8 Discrete Distribution Family Lecture Python Code ...")

    main()

    print("\nEnd of Topic 8 Discrete Distribution Family Lecture Python Code ...\n")




