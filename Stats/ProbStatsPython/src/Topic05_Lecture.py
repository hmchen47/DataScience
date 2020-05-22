#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt


def coin_flip_plot(p, n=1000, debug=False):
    """randomly toss coin for n times, count the number of heads and tails, and visualize them

    Arguments:
        p {float} -- probability of flipping coin w/ tail

    Keyword Arguments:
        n {int} -- the number of coin tosses (default: {1000})
        debug {bool} -- turn on or off the debugging message (default: {False})
    """

    tosses = np.random.choice(["h", "t"], p=[1-p, p], size=n)
    if debug: 
        print("\nlist of tossing result ({}): {}".format(n, ', '.join(tosses)))

    # count hte number of heads and tails
    heads = list(tosses).count("h")
    tails = list(tosses).count("t")

    if debug:
        print("{} heads and {} tails".format(heads, tails))

    # bar plot for the coin flips
    plt.bar([0, 1], [heads, tails], tick_label=['h', 't'], align='center')
    plt.ylim([0, 10])
    plt.show()

    return None


def simulate_coin_tosses(p, n_tosses=1000, n_simulations=10, debug=False):
    """perform n_simulations simulations each simulation emulates n_tosses coin flips and plot the 
    fractions of heads as the tosses number increases.

    Arguments:
        p {float} -- probability of flipping head in a toss

    Keyword Arguments:
        n_tosses {int} -- the number of tosses in a simulation
        n_simulations {int} -- the number of simulations
        debug {bool} -- turn on or off the debugging message (default: {False})
    """

    for _ in range(n_simulations):
        # create three arrays consisting of: the coin flips, their running sums, and partial estimates
        tosses = np.random.choice([0, 1], p=[1-p, p], size=n_tosses)
        partial_sums = np.cumsum(tosses)
        partial_means = partial_sums / np.arange(1, n_tosses+1)

        # plot the partial estimates
        plt.plot(np.arange(1, n_tosses+1), partial_means)

    # plt.plot(range(n_tosses), [p]*n_tosses, 'k', linewidth=5.0, leabel='p')   # plot the value p
    plt.plot(range(n_tosses), [p]* n_tosses, 'k', linewidth=5.0, label = 'p')

    plt.xlabel('Number of coin tosses')
    plt.ylabel('Fraction of heads')
    plt.xlim([1, n_tosses])         # plot limits
    plt.legend()
    plt.show()

    return None



def main():

    # basic configuration for plots
    plt.style.use([{
        "figure.figsize": (12, 9),       # figure size
        "xtick.labelsize": "large",         # font-size pf the X-ticks
        "ytick.labelsize": "large",         # font-size Y-ticks
        "legend.fontsize": "x-large",       # font size of the legend
        "axes.labelsize": "x-large",        # font size of the label
        "axes.titlesize": "xx-large",       # font title size of title
        "axes.spines.top": False,
        "axes.spines.right": False,
    }, 'seaborn-poster'])

    # coin flip
    p, n = 0.5, 10
    # coin_flip_plot(p, n, True)

    # increasing tosses and number of simulation 
    p, n_tosses, n_simulations = 0.5, 1000, 5
    # simulate_coin_tosses(p, n_tosses, n_simulations, False)

    # reproducibility
    np.random.seed(666)
    print("\nnp.random.seed(666) --> np.random.randint(9) x 2: {}, {}".format(np.random.randint(9), np.random.randint(9)))
    np.random.seed(666)
    print("\nrepeat\nnp.random.seed(666) --> np.random.randint(9) x 2: {}, {}".format(np.random.randint(9), np.random.randint(9)))





    # input("Press Enter to continue ...")

    return None

if __name__ == "__main__":

    print("\nStart Topic 5 Intro to Probability Python code ...")

    main()

    print("\nEnd Topic 5 Intro to Probability Python code...\n")