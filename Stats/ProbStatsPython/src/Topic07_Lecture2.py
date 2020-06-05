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


    return P_G3_given_S_GP, P_G3_given_S_MS

def plot_estimating_schools(df_por, P_G3_given_S_GP, P_G3_given_S_MS):
    """plot the probability P(s) that a student belongs to a given school

    Args:
        df_por (DataFrame): data extracted from student_por.csv
    """

    data_tmp = df_por["school"].value_counts()
    P_S = pd.DataFrame(data_tmp/data_tmp.sum())
    P_S.columns = ["Probability"]
    P_S.columns.name = "School"
    P_S.plot.bar()

    P_G3 = P_G3_given_S_GP * P_S.loc["GP"].values + P_G3_given_S_MS * P_S.loc["MS"].values
    plt.figure(figsize=(12, 9))
    P_G3.plot.bar()
    plt.xlabel("Scores")
    plt.ylabel("$P(g)$")
    plt.title("Distribution of scores for both schools")
    plt.show()

    return None

def expectation_schools(df_por, p_G3_given_S_GP, p_G3_given_S_MS):
    """compute the expectations of grades w/ schools

    Args:
        p_G3_given_S_GP (DataFrame): conditional probability of grades on given GP
        p_G3_given_S_MS (DataFrame): conditional probability of grades on given MS
    """
    data_tmp = df_por["school"].value_counts()
    P_S = pd.DataFrame(data_tmp/data_tmp.sum())
    P_G3 = p_G3_given_S_GP * P_S.loc["GP"].values + p_G3_given_S_MS * P_S.loc["MS"].values

    E_G3_given_S_GP = np.sum([index*value for index, value in \
        zip(p_G3_given_S_GP.index, p_G3_given_S_GP.values)])
    print("\nE[G]S = Gabriel Pereira]= {:.3f}".format(E_G3_given_S_GP))

    E_G3_given_S_MP = np.sum([index*value for index, value in \
        zip(p_G3_given_S_MS.index, p_G3_given_S_MS.values)])
    print("\nE[G|S = Mousubho da Silveira]= {:.3f}".format(E_G3_given_S_MP))

    E_G3 = np.sum([index*value for index, value in zip(P_G3.index, P_G3.values)])
    print("\nE[G] = {:.3f}".format(E_G3))

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
    
    p_G3_given_S_GP, p_G3_given_S_MS = cond_prob(df_por, debug)

    # plot_cond_prob(p_G3_given_S_GP, p_G3_given_S_MS)

    # estimating the probability P(s) that a student belongs to a given school
    plot_estimating_schools(df_por, p_G3_given_S_GP, p_G3_given_S_MS)

    # compute expectation
    expectation_schools(df_por, p_G3_given_S_GP, p_G3_given_S_MS)


    return None


if __name__ == "__main__":

    print("\nStarting Topic 7 Lecture NB 2 Python code ...")

    main()

    print("\nEnd Topic 7 Lecture NB 2 Python code ...\n")
