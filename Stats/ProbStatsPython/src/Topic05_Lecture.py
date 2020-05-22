#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt





def main():

    # basic configuration for plots
    plt.style.use([{
        "figure.figsize": (12, 9),       # figure size
        "xtick.labelsize": "large",         # font-size pf the X-ticks
        "ytick.labelsize": "large",         # font-size Y-ticks
        "legend.fontsize": "x-large",       # font size of the legend
        "axes.labelsize": "x-large",        # font size of the label
        "axes.titlesize": "xx-large",       # font title size of title
        "axes.spines.top": False,
        "axes.spines.right": False,
    }, 'seaborn-poster'])


    return None

if __name__ == "__main__":

    print("\nStart Topic 5 Intro to Probability Python code ...")

    main()

    print("\nEnd Topic 5 Intro to Probability Python code...")