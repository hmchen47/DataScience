#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np

import math
import random
import itertools

import matplotlib.pyplot as plt

# range of number of people
PEOPLE = np.arange(1, 26)

# days in year
DAYS = 365    


def prob_unique_birthdays(num_people):
    """return the probability that all birthdays are unique, among a given
    number of people w/ uniformly-distributed birthdays.

    Arguments:
        num_people {int} -- number of people
    """

    return (np.arange(DAYS, DAYS - num_people, -1) / DAYS).prod()

def sample_unique_birthdays(num_people):
    """select a sample of people w/ uniformly-distributed birthdays, and
    returns True if all birthdays are unique (or False otherwise).

    Arguments:
        num_people {int} -- number of people
    """
    bdays = np.random.randint(0, DAYS, size=num_people)
    unique_bdays = np.unique(bdays)

    return len(bdays) == len(unique_bdays)


def plot_probs(iterations):
    """Plots a comparison of the probability of a group of people all having
    unique birthdays, btw the theoretical and empirical probabilities.

    Arguments:
        iterations {int} -- number of iterations to perform the plot
    """

    sample_prob = []    # empirical prob of unique-birthday sample
    prob = []           # theoretical prob of unique-birthday sample

    # computedata points to plot
    np.random.seed(1)
    for num_people in PEOPLE:
        unique_count = sum(sample_unique_birthdays(num_people) 
            for i in range(iterations))
        sample_prob.append(unique_count / iterations)
        prob.append(prob_unique_birthdays(num_people))

    # plot results
    plt.plot(PEOPLE, prob, 'k-', linewidth=3.0, label="Theoretical probability")
    plt.plot(PEOPLE, sample_prob, 'bo-', linewidth=3.0, label="Empirical probability")
    plt.gcf().set_size_inches(20, 10)
    plt.axhline(0.5, color="red", linewidth=4.0, label="0.5 threshold")
    plt.xlabel("Number of people", fontsize=18)
    plt.ylabel("Probability of unique birthdays", fontsize=18)
    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

    return None


def main():

    # Birthday paradox
    plot_probs(2000)
    

    return None


if __name__ == "__main__":
    print("\nEntering Lecture 6 Conditional Probability Python code ...")

    main()

    print("\nEnd Lecture 6 Conditional Probability Python code ...\n")




