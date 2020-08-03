#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def hypothesisTesting(n, option, sig_lvl, debug=False):
    """
    Plot the distribution for the hypothesis test w/ Bernoulli
    distribution as the coin toss
    
    Args:
        n (int): number of trials w/ Bernoulli distribution
        option (str): different type of alternative hypothesis
            + p > 0.5: right-sided tailed Ha
            + p < 0.5: left-sided tailed Ha
            + p ≠ 0.5: two-sided tailed Ha
        sig_lvl (float): significance level for the test
        debug (bool, optional): turn on/off debug msg. Defaults to False.

    Returns:
        None: just disply the figure for hypothesis testing
    """
    # generate list of prob for H0 w/ given number of trials
    pmf = stats.binom.pmf(range(n+1), n=n, p=0.5)

    plt.figure(figsize=(8, 6))
    plt.xlabel("Number of Heads", fontsize=12)
    plt.ylabel("Probability", fontsize=12)

    # plot the computed pmf of Bernoulli dist
    if n >= 30 :
        plt.plot(range(n+1), pmf)
    else:
        for i in range(n+1):
            plt.plot([i, i], [0.0, pmf[i]], 'go-', linewidth=1)

    # compute the ppf for Binomial dist w/ different probability p
    print("\n")
    if option == "p < 0.5":
        k = stats.binom.ppf(1.0-sig_lvl, n=n, p=0.5)
        print("Reject null hypothesis if number if heads is more than {}".format(k))
        plt.plot([np.floor(k)-0.1, np.floor(k)-0.1], [0, pmf[int(np.floor(k))]*(1+0.1)], 'r-', linewidth=2)
        plt.title("The Binomial distribution under the Null Hypothesis \nbelow {}".format(k),\
            fontsize=16)
    elif option == "p > 0.5":
        k = stats.binom.ppf(sig_lvl, n=n, p=0.5) - 1
        print("Reject null hypothesis if number of heads is less than {}".format(k))
        plt.plot([np.ceil(k)+0.1, np.ceil(k)+0.1], [0, pmf[int(np.ceil(k))]*(1+0.1)], 'r-', linewidth=2)
        plt.title("The Binomial distribution under the Null Hypothesis \nbeyond {}".format(k),\
            fontsize=16)
    elif option == "p ≠ 0.5":
        k1 = stats.binom.ppf(1.0-sig_lvl/2.0, n=n, p=0.5) + 1
        k2 = stats.binom.ppf(sig_lvl/2.0, n=n, p=0.5) - 1
        print("Reject null hypothesis if number of heads lies outside {} and {}".format(k2, k1))
        plt.plot([np.ceil(k1)-0.1, np.ceil(k1)-0.1], [0, pmf[int(np.ceil(k1))]*(1+0.1)], 'r-', linewidth=2)
        plt.plot([np.floor(k2)+0.1, np.floor(k2)+0.1], [0, pmf[int(np.floor(k2))]*(1+0.1)], 'r-', linewidth=2)
        plt.title("The Binomial distribution under the Null Hypothesis \nbtw {} and {}".format(k2, k1),\
            fontsize=16)

    plt.show()

    return None


def sample(p=0.5):
    """generate normal distribution approximating the Binomial distribution
    
    Gaussian Mixture Models: X = Y *Z1 + (1 - Y) * Z2

    Y: Bernoulli random variable
    Z1 & Z2: Gaussian random variables

    Sampling X is same as sampling Z1 and Z2 w/ probability p and 1-p, respectively.

    Args:
        p (float, optional): probability of Binomial dist. Defaults to 0.5.
    """
    Y = np.random.rand(1)

    if Y >= p:
        return np.random.normal(10, 2)
    else:
        return np.random.normal(12, 2)


def plot_estimate_samples():
    plt.figure(figsize=(12, 8))
    plt.hist(np.asarray([sample(0.2) for _ in range(10000)]), 50, density=True)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("probability", fontsize=12)
    plt.title('Normal distribution of $X = Y \\ast Z_1 + (1-Y) \\ast Z_2$ w/ $\\mu=10$ & $\\sigma^2=4$')

    plt.show()

    return None


if __name__ == "__main__":

    print("\nStarting Topic 13 Lecture NB Python code .......")

    # hypothesis testing for coin bias
    print("\nHypothesis testing for Bernoulli distribution")
    options = ["p > 0.5", "p < 0.5", "p ≠ 0.5"]
    n, sig_lvl, debug = 25, 0.01, True
    # hypothesisTesting(n, options[0], sig_lvl, debug=debug)
    # hypothesisTesting(n, options[1], sig_lvl, debug=debug)
    # hypothesisTesting(n, options[2], sig_lvl, debug=debug)


    # z-test
    # plot histogram of X = Y*Z_1 + (1-Y)*Z2
    plot_estimate_samples()


    print("\nEnd of Topic 13 Lecture NB Python code .......\n")



