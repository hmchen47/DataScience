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


def tetrahedron_roll_plot(n, debug=False):
    """Simulate rolling tetrahedron n time

    Arguments:
        n {int} -- the number of die rolling

    Keyword Arguments:
        debug {bool} -- turn on/off debuging message (default: {False})
    """

    samples = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4], size=n)
    height, left = np.histogram(samples, bins=4, range=(1, 5))
    heights = height/n

    # plt.figure(figsize=(12, 9))
    colors = 'rgby'
    plt.bar(left[:-1], heights, color=colors, tick_label=[1, 2, 3, 4], align='center')
    plt.xlabel("Outcomes", fontsize=16)
    plt.ylabel("Probabilities", fontsize=16)
    plt.show()

    return None


def tetrahedron_event_plot(n, debug=False):
    """Plot tetrahedron rolling w/ Even and Oldd

    Arguments:
        n {int} -- the number of rolling tetrahedron die

    Keyword Arguments:
        debug {bool} -- turn on/off (default: {False})
    """

    samples = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4], size=n)
    height, left = np.histogram(samples, bins=4, range=(1, 5))
    heights = height/n

    plt.bar([0, 1], [0, 0], tick_label=['Odd', 'Even'], align='center')
    plt.bar([0, 1], [heights[0], 0], color='r', align='center')
    plt.bar([0, 1], [heights[2], 0], color='b', bottom=[heights[0], 0], align='center')
    plt.bar([0, 1], [0, heights[1]], color='g', align='center')
    plt.bar([0, 1], [0, heights[3]], color='y', bottom=[0, heights[1]], align='center')
    plt.xlabel('Events', fontsize=16)
    plt.ylabel('Probabilities', fontsize=16)

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

    # Tetrahedron die events
    n = 10000
    tetrahedron_roll_plot(n, False)

    tetrahedron_event_plot(n, False)



    # input("Press Enter to continue ...")

    return None

if __name__ == "__main__":

    print("\nStart Topic 5 Intro to Probability Python code ...")

    main()

    print("\nEnd Topic 5 Intro to Probability Python code...\n")