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



def main():

    # illustrate line plot w/ cos function
    line_plot()

    # display legends, linestyle, colors and markers
    llcm_plot()

    input("\nPress Enter to continue ...")


    return None


if __name__ == "__main__":

    print("\nStarting Topic 03 Matplotlib Totorial ...")

    main()

    print("\nEnd of Topic 03 Matplotlib Totorial ...")
