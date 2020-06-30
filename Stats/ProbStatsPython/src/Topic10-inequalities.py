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
    plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 1
    plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 5
    plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 10
    plot_markov_chebyshev(mu, sig)
    mu, sig = 5, 20
    plot_markov_chebyshev(mu, sig)

    Input("Press Enter to continue ......")

    return None


if __name__ == "__main__":

    print("\nStarting Topic 10 Lecture NB .......")

    main()

    print("\nEnd Topic 10 Lecture NB .......\n")