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

def main():

    print("\nLine plot ...")
    line_plot()
    input("\nPress Enter to continue ...")


    return None


if __name__ == "__main__":

    print("\nStarting Topic 03 Matplotlib Totorial ...")

    main()

    print("\nEnd of Topic 03 Matplotlib Totorial ...")
