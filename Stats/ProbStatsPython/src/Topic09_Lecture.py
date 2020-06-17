#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pandas
import scipy.stats as stat
import matplotlib.pyplot as plt 

from matplotlib import axes
from math import pi, e

def uniform_pdf_cdf(width):
    """plot uniform PDF, CDF

    Args:
        width (int): width of uniform 
    """

    a, b = 0, width
    xrange = np.linspace(0, 10, 101)
    cdf = [0 for z in xrange if z<=a]
    cdf.extend([(z-a)/(b-a) for z in xrange if z>a and z<=b])
    cdf.extend([1 for z in xrange if z>b])

    # plot the PDF
    if b != a:
        plt.plot([a+(b-a)*(1/10) for i in range(11)], [1/(b-a)]*11,\
            linewidth=3.0, label="PDF")
        plt.plot([a, a], [0, 1/(b-a)], 'g--',\
            [b, b], [0, 1/(b-a)], 'g--', \
            [a, b], [1/(b-a), 1/(b-a)], 'g--', linewidth=2.0)
        
    # plot the CDF
    plt.plot(xrange, cdf, 'r', linewidth=3.0, label='CDF')

    plt.title('PDF and CDF of U({}, {})'.format(a, b))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    return None


def main():

    # set the size and properties o the plots when using matplotlib.pyplot
    plt.style.use([{
        "figure.figsize": (12, 9),
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "xx-large",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # "ytick.major.right": False,
        # "ytick.major.top": False
    }, 'seaborn-poster'])

    # uniform distribution
    width = 7
    uniform_pdf_cdf(width)

    return None


if __name__ == "__main__":

    print("\nStarting Topic 09 Lecture Python code ....")

    main()

    print("\nEnd Topic 09 Lecture Python code ....\n")


