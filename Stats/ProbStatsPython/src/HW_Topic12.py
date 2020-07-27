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


def do_regression_old(data):
    # input: the HW's dataset
    # output: a numpy array yielding w=(w0,w1) from linear regression

    A = np.array(data['study_hours'])
    A = np.array([np.ones(len(A)), A])
    y = np.array(data['grades'])

    w = np.linalg.lstsq(A.T, y, rcond=None)[0]

    return w

def do_regression(data, debug=False):
    """get regression line parameters

    Args:
        data (dataframe): dataset for analysis
    """
    x = data['study_hours'].values
    y = data['grades'].values
    A = np.vstack([np.ones(len(x)), x]).T

    w = np.linalg.lstsq(A, y, rcond=None)[0]

    if debug:
        print("\nRegression analysis Aw = y")
        print("A= \n{}...... \ny = {}...... \n-> w= {}"\
            .format(A[:2, :], y[:2], w))

    return w


def reverse_regression(data):
    # input: the HW's dataset
    # output: a numpy array yielding w=(w0,w1) for the reversed linear regression
    
    x = data['grades'].values
    y = data['study_hours'].values
    A = np.vstack([np.ones(len(x)), x]).T

    w = np.linalg.lstsq(A, y, rcond=None)[0]

    return w

def y_minus_x(df):
    # input: the HW's dataset
    # output: there is NO OUTPUT

    df['y-x'] = df['y'] - df['x']

    return df
    

def do_regression2(df):
    # input: the HW's dataset
    # output: a numpy array yielding w=(w0,w1) from linear regression
    
    x = df['x'].values
    y = df['y-x'].values
    A = np.vstack([np.ones(len(x)), x]).T

    w = np.linalg.lstsq(A, y, rcond=None)[0]

    return w


def main(debug=False):
    """main function to execute all

    Args:
        debug (bool, optional): turn on/off debug message. Defaults to False.
    """

    # Regression
    data_df = pd.read_csv('./data/hw_regression_data.csv')

    if debug:
        print("\nLoading data for study hours and grades...")
        print("  \ndataframe info: {}".format(data_df.info()))
        print("  \ndataset: \n{}".format(data_df.head()))

    # exercise 1: get averages
    per_hours_means = get_averages(data_df)

    if debug:
        print("\nTable of index= {} and columns= {}: \n{}\n......"\
            .format(per_hours_means.index.name, per_hours_means.columns, \
            per_hours_means.head()))

    # Exercise 2: Simple Linear Regression
    w = do_regression(data_df, True)

    if debug:
        print("\nparameter vector: {}".format(w))
        print("  typ2= {}  shape={}".format(type(w), w.shape))

    # Exercise 3: reversed regression
    w2 = reverse_regression(data_df)

    if debug:
        print("\nreverse parameter vector: {}".format(w2))
        print("  typ2= {}  shape={}".format(type(w2), w2.shape))

    
    # Regression to the Mean
    hw_df = pd.read_csv('./data/gauss_R2.csv')

    if debug:
        print("\nLoading data for gauss...")
        print("  \ndataframe info: {}".format(hw_df.info()))
        print("  \ndataset: \n{}".format(hw_df.head()))

    # Exercise 4: put y - x in DataFrame
    hw_df = y_minus_x(hw_df)

    if debug:
        print("\nLoading data for gauss data...")
        print("  \ndataframe info: {}".format(hw_df.info()))
        print("  \ndataset: \n{}".format(hw_df.head()))


    # Exercise 5: simple linear regression
    w3 = do_regression2(hw_df)

    if debug:
        print("\nreverse parameter vector: {}".format(w3))
        print("  typ2= {}  shape={}".format(type(w3), w3.shape))

    return None


if __name__ == "__main__":

    print("\nStarting Topic 12 HW Python code ......")

    main(debug=True)

    print("\nEnd of Topic 12 HW Python code ......\n")


