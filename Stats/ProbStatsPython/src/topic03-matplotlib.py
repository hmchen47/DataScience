#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def line_plot():
    """Plotting y = cos(x) function
    """
    # generate data
    x = np.arange(0, 4 * np.pi, 0.1)    # x in [0, 4* pi)
    y_cos = np.cos(x)

    plt.figure()
    plt.plot(x, y_cos)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Cosine function in $[0, 4\pi)$ with line plot')
    plt.show()

    return None

def llcm_plot():
    """Display Legends, Linestyle, Colors, and Markers w/ y=2^x and y = x^2
    """
    x = np.arange(0, 10, 0.5)  # x in [0, 10)
    y_1 = 2**x
    y_2 = x**2

    plt.figure()
    # specify color, linestyle, and marker w/ keyword argument
    # plt.plot(x, y_1, label='$2^x$', color='g', linestyle='--', marker='o')
    plt.plot(x, y_1, 'g--o', label='$2^x$')
    # plt.plot(x, y_2, label='$x^2$', color='r', linestyle='-', marker='*')
    plt.plot(x, y_2, 'r-*', label='$x^2$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='upper left')
    plt.title('Line plots w/ Legend, linestyle, color, and marker')
    plt.show()


    return None


def tgfs_plot():
    """ Display title and changing ontsize
    """
    plt.rc('font', size=10)         # control the default font size
    plt.rc('axes', titlesize=11)    # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the axes x and y labels
    plt.rc('xtick', labelsize=10)   # fontsize of the tick label
    plt.rc('ytick', labelsize=10)   # fontsize of the tick label
    plt.rc('legend', fontsize=11)   # legend fontsize

    # generate data
    x = np.arange(-10, 10.1, 0.1)
    y = x**3

    # plot figure
    plt.figure()
    plt.plot(x, y, label='$x^3$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$y = x^3 for title, grid, & font size control "plt.rc()"$')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    return None


"""
Disply multiple subplots in a single figure for cosine and sine
"""
def subplots():
    # generate data
    x = np.arange(0, 6*np.pi + 0.2, 0.2)
    y_1 = np.cos(x)
    y_2 = np.sin(2*x)

    # plot cos(x)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, y_1, label='$\cos(x)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='lower left')
    plt.title('Multiple subplots w/ plt.subplot()')

    # plot sin(2x)
    plt.subplot(2, 1, 2)
    plt.plot(x, y_2, label='$\sin(x)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='lower left')

    plt.show()

    return None


def axes_subplots():
    """ display subplots sharing same axes
    """
    # gerenate data
    x = np.arange(0, 6 * np.pi+0.2, 0.2)
    y_1 = np.cos(x)
    y_2 = np.sin(2*x)
    y_3 = y_1 + y_2

    # display multiple
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle('Subplots w/ shared axes')
    axs[0].plot(x, y_1)
    axs[1].plot(x, y_2)
    axs[2].plot(x, y_3)
    axs[0].set_ylabel('$y$')
    axs[1].set_ylabel('$y$')
    axs[2].set_ylabel('$y$')

    plt.show()

    return None

"""
Plot Bar graph
"""
def bar_plot():
    # generate data
    x = np.arange(0, 7, 1)
    y = x

    # display plot
    plt.figure()
    plt.bar(x, y, label='$x$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='upper left')
    plt.title('Bar chart')
    plt.show()

    return None



def main():

    # illustrate line plot w/ cos function
    # line_plot()

    # display legends, linestyle, colors and markers
    # llcm_plot()

    # display title and grid & change font size
    # tgfs_plot()

    # display multiple figures
    # subplots()

    # subplots sharing axes
    # axes_subplots()

    # bar plot
    bar_plot()

    # input("\nPress Enter to continue ...")


    return None


if __name__ == "__main__":

    print("\nStarting Topic 03 Matplotlib Totorial ...")

    main()

    print("\nEnd of Topic 03 Matplotlib Totorial ...")
