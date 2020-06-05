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


def plot_cond_probs():
    """Plot the conditional probability w/ data set from Kaggle
    https://www.kaggle.com/uciml/student-alcohol-consumption/download
    about the students' performance in Portuguese course

    Attributes in dataset
    + G3 - final grade related with the course subject, Math or Portuguese (numeric: from 0 to 20, output target)
    + studytime - weekly study time (numeric: 1 : < 2 hours, 2 : 2 to 5 hours, 3 : 5 to 10 hours, or 4 : > 10 hours)
    """

    data_por = pd.read_csv("data/student-por.csv")

    attributes = ["G3", "studytime"]
    data_por = data_por[attributes]

    # probability that a student's study-time falls in an interval
    data_temp = data_por["studytime"].value_counts()
    P_studytime = pd.DataFrame((data_temp/data_temp.sum()).sort_index())
    P_studytime.index = ["< 2 hours","2 to 5 hours","5 to 10 hours","> 10 hours"]
    P_studytime.columns = ["probability]"]
    P_studytime.columns.name = "Study Interval"

    # plot study interval probability figure
    P_studytime.plot.bar(figsize=(12, 9), fontsize=18)
    plt.ylabel("Probability", fontsize=16)
    plt.xlabel("Study Interval", fontsize=18)
    plt.show()

    # calculate high score and plot it
    data_temp = (data_por["G3"] >= 15).value_counts()
    P_score15_p = pd.DataFrame(data_temp/data_temp.sum())
    P_score15_p.index = ["Low", "High"]
    P_score15_p.columns = ["probability"]
    P_score15_p.columns.name = "Score"
    print(P_score15_p)
    P_score15_p.plot.bar(figsize=(10, 6), fontsize=16)
    plt.xlabel("Score", fontsize=18)
    plt.ylabel("Probability", fontsize=18)
    plt.show()

    # conditional probability Pr(study inteval | highscore)
    score = 15
    data_temp = data_por.loc[data_por["G3"] >= score, "studytime"]
    P_T_given_score15 = pd.DataFrame((data_temp.value_counts()/data_temp.shape[0]).sort_index())
    P_T_given_score15.index = ["< 2 hours","2 to 5 hours","5 to 10 hours","> 10 hours"]
    P_T_given_score15.columns = ["Probability"]
    print("Probability of study interval given that the student gets a highscore:")
    P_T_given_score15.columns.name = "Study Interval"
    P_T_given_score15.plot.bar(figsize=(12, 9), fontsize=16)
    plt.xlabel("Study interval", fontsize=18)
    plt.ylabel("probability", fontsize=18)
    plt.show()

    # predict probability w/ Bayes rule
    P_score15_given_T_p = P_T_given_score15 * P_score15_p.loc["High"] / P_studytime
    print("probability of high score given study interval:")
    pd.DataFrame(P_score15_given_T_p).plot.bar(figsize=(12,9),fontsize=18).legend(loc="best")
    plt.xlabel("Study interval", fontsize=18)
    plt.ylabel("probability", fontsize=18)
    plt.show()

    return None


def main():

    # Birthday paradox
    # plot_probs(2000)

    # conditional probability
    plot_cond_probs()

    return None


if __name__ == "__main__":
    print("\nEntering Lecture 6 Conditional Probability Python code ...")

    main()

    print("\nEnd Lecture 6 Conditional Probability Python code ...\n")




