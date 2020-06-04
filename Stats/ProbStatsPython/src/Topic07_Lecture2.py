#!/usr/bin/env python3
# -*- coding: utf-8 -8-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def plot_cond_prob(P_G3_given_S_GP, P_G3_given_S_MS):
    """plot the P_G3_given_S_GP and P_G3_given_S_MS w/ bar chart

    Args:
        P_G3_given_S_GP (DataFrame): data w/ GP and the grade 3 of its students
        P_G3_given_S_MS (DataFrame): data w/ MS and the grade 3 of its students
    """
    plt.figure(figsize=(12, 9))
    P_G3_given_S_GP.plot.bar()
    plt.xlabel("Score")
    plt.ylabel("$P(g \mid gp)$")
    plt.title("Distribution of scores for Gabriel Pereira")
    plt.show()

    plt.figure(figsize=(12, 9))
    P_G3_given_S_MS.plot.bar()
    plt.xlabel("Scores")
    plt.ylabel("$P(g \mid ms)$")
    plt.title("Distribution of scores for Mousingo da Silveira")
    plt.show()

    return None


def cond_prob(df_por, debug=False):
    """compute the conditional probability

    Args:
        df_por (DataFrame): dataframe for analsis
    """

    # if debug: print("\ndf_por= \n{}".format(df_por.head(10)))

    # for Gabriel Pereira
    P_G3_given_S_GP = pd.DataFrame(index=range(21)).fillna(0)
    df_tmp = df_por.loc[df_por["school"] == "GP", "G3"].value_counts()
    P_G3_given_S_GP["Probability"] = (df_tmp/df_tmp.sum())
    P_G3_given_S_GP = P_G3_given_S_GP.fillna(0)
    if debug: print("\nP_G3_given_S_GP= \n{}".format(P_G3_given_S_GP.head()))


    # for Mousinho da Silveira
    P_G3_given_S_MS = pd.DataFrame(index=range(21)).fillna(0)
    df_tmp = df_por.loc[df_por["school"] == "MS", "G3"].value_counts()
    P_G3_given_S_MS["Probability"] = df_tmp/df_tmp.sum()
    P_G3_given_S_MS = P_G3_given_S_MS.fillna(0)
    if debug: print("\nP_G3_given_S_MS= \n{}".format(P_G3_given_S_MS.head()))

    plot_cond_prob(P_G3_given_S_GP, P_G3_given_S_MS)


    return None




def main():

    # general setting for plotting style
    plt.style.use([{
        "figure.figsize": (12, 9),      # figure size
        "xtick.labelsize": "large",     # font size of the X-ticks
        "ytick.labelsize": "large",     # font size of the Y-ticks
        "legend.fontsize": "x-large",   # font size of the legend
        "axes.labelsize": "x-large",    # font size of the labels
        "axes.titlesize": "xx-large",   # fontsize of the title
        "axes.spines.top": False,
        "axes.spines.right": False
    }, 'seaborn-poster'])

    # read csv data file student-por.csv
    attributes = ['G3', 'school']
    # df_por = pd.read_csv("./data/student-por.csv")
    df_por = pd.read_csv("./data/student-por.csv", usecols=attributes)


    debug = True
    debug = False
    # if debug: print("\ndf_por= \n{}".format(df_por.head(10)))

    # compute the distribution of the final grades for students in each school
    # P(g|s) = P(G = g|S = s), for all g \in {0, 1, ..., 20}, s \in {'GP', 'MS'}
    cond_prob(df_por, debug)



    return None


if __name__ == "__main__":

    print("\nStarting Topic 7 Lecture NB 2 Python code ...")

    main()

    print("\nEnd Topic 7 Lecture NB 2 Python code ...\n")
