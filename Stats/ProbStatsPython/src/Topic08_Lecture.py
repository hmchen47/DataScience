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
    gen_plot_bernoulli()



    return None

if __name__ == "__main__":

    print("\nStarting Topic 8 Discrete Distribution Family Lecture Python Code ...")

    main()

    print("\nEnd of Topic 8 Discrete Distribution Family Lecture Python Code ...\n")




