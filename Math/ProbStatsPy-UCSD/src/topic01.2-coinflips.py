#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

# generate the sum of k coin flips, repeat that n times
def generate_counts1(k=1000, n=100):
    X = 2*(np.random.rand(k, n) > 0.5) - 1

    return np.sum(X, axis=0)

# use new random number generator, a faster version but required SciPy 1.4+
def generate_counts2(k=1000, n=100):
    rng = np.random.default_rng()
    X = 2*(rng.integers(2, size=(k, n))) - 1

    return np.sum(X, axis=0)


    

def main():
    # historgram of coin flip w/ generate_count1
    n, k = 1000, 1000
    plt.figure(figsize=[13, 3.5])
    counts = generate_counts1(k, n)
    plt.hist(counts);
    plt.xlim([-k, k])
    plt.xlabel("sum")
    plt.ylabel("count")
    plt.title("Histogram of coin flip sum when flipping a fair coin {0:d} times".format(k))
    plt.grid()
    plt.show();
    # input("Press Enter to continue...")

    # coin flip simulations w/ different trials
    plt.figure(figsize=[13, 3.5])
    for j in range(2, 5):
        k = 10**j
        counts = generate_counts1(k, n=100)
        plt.subplot(130 + j - 1)
        plt.hist(counts, bins=11)
        d = 4*math.sqrt(k)
        plt.plot([-d, -d], [0, 30], 'r')
        plt.plot([+d, +d], [0, 30], 'r')
        plt.grid()
        plt.title("{0:d} flips, bound = +-{1:.1f}".format(k, d))
    plt.show()
    # input("Press Enter to continue...")

    # coin flip simulations w/ different trials and x-axis range
    plt.figure(figsize=[13, 3.5])
    for j in range(2, 5):
        k = 10**j
        counts = generate_counts1(k, n=100)
        plt.subplot(130 + j - 1)
        plt.hist(counts, bins=11)
        d = 4*math.sqrt(k)
        plt.xlim([-k, k])
        plt.plot([-d, -d], [0, 30], 'r')
        plt.plot([+d, +d], [0, 30], 'r')
        plt.grid()
        plt.title("{0:d} flips, bound = +-{1:.1f}".format(k, d))
    plt.show()
    # input("Press Enter to continue...")



if __name__ == "__main__":
    print("\nTopic 1 Lecture Python code execution:\n")

    main()

    print("\n... End of Topic01 Lecture Python code execution ...\n")

