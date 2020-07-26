#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt

def f(x, reg):
    """1st degree polynomial: lineare regession line

    Args:
        x (ndarray): vectore of inputs
        reg (ndarray): parameter vector for regression line
    """
    return reg[0]+x*reg[1]


def df_averages(df, x_name, y_name, debug=False):
    """calculate mean of x and y axis"""
    # calculate the mean weight for each 1-inch interval of height
    df['round_'+x_name] = df[x_name].round()

    per_yname_means = df.groupby('round_'+x_name).mean()[[y_name]]

    if debug:
        print("\nOn rounded {} list of average {}: \n{}"\
            .format(x_name, y_name, per_yname_means))

    return per_yname_means


def get_averages(data):
    # input: the HW's dataset
    # output: a pandas dataframe yielding the mean grade for each rounded number of study hours

    return df_averages(data, 'study_hours', 'grades')


def do_regression(data):
    # input: the HW's dataset
    # output: a numpy array yielding w=(w0,w1) from linear regression

    A = np.array(data['study_hours'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(data['grades'])

    w = np.linalg.lstsq(A.T, y)[0]

    return w



def main(debug=False):
    """main function to execute all

    Args:
        debug (bool, optional): turn on/off debug message. Defaults to False.
    """

    # Regression
    data_df = pd.read_csv('./data/hw_regression_data.csv')

    if debug:
        print("\nLoading data...")
        print("  \ndataframe info: {}".format(data_df.info()))
        print("  \ndataset: \n{}".format(data_df.head()))

    # exercise 1: get averages
    per_hours_means = get_averages(data_df)

    if debug:
        print("\nTable of index= {} and columns= {}: \n{}".format(per_hours_means.index.name, \
            per_hours_means.columns, per_hours_means))

    # Exercise 2: Simple Linear Regression
    w = do_regression(data_df)

    if debug:
        print("\nparameter vector: {}".format(w))
        print("  typ2= {}  shape={}".format(type(w), w.shape))



    return None


if __name__ == "__main__":

    print("\nStarting Topic 12 HW Python code ......")

    main(debug=True)

    print("\nEnd of Topic 12 HW Python code ......\n")


