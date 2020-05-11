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


"""
Plotting a figure w/ multiple y-axis
"""

def maxes_plot():
    # Make-up data
    calories = [380.70, 420.98, 454.91, 406.45, 446.16, 498.08, 504.54, 459.05, 459.55, 484.79]
    countries = ['India', 'Japan', 'Korea', 'China', 'Thai', 'Italy', 'France', 'Greece', 'Mexico', 'US']
    obesity_rates = [3.9, 4.3, 4.7, 6.2, 10, 19.9, 21.6, 24.9, 28.9, 36.2]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle('Multiple axis w/ bar chart')
    ax1.bar(countries, obesity_rates, color='C8')
    ax1.set_ylabel('Obesity rate(%)', color='C8')
    ax1.tick_params(axis='y', labelcolor='C8')

    ax2 = ax1.twinx()
    ax2.plot(countries, calories, color='C0')
    ax2.set_ylabel('Calories', color='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    plt.show()

    return None


"""
Scatter plot w y = 2x+3 + epsilon, epsilon ~ N(0, 1)
"""

def scatter_plot():
    # genater date
    x = np.arange(0, 10.2, 0.2) # x in [0, 10]
    noise = np.random.randn(len(x))
    y = 2*x + 3 + noise

    # plot scatter
    plt.figure()
    plt.scatter(x, y, label='$y=2x+3+\epsilob$')
    plt.plot(x, 2*x+3, color='r', label='regression')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Scatter plot w/ $y=2x+3+\epsilon, \epsilon \sim N(0,1)$')
    # plt.legend(loc='upper left')
    plt.show()

    return None


"""
Contour plot J(w) = (w - w_0)^T A (w - w_0) 

w = (-2, 2)^T, A = ((2, 0), (0, 1))
"""

def contour_plot():
    # generate data
    xmin, xmax, xstep = -4, 0, .1
    ymin, ymax, ystep = 0, 4, 0.1

    A = np.array([[2, 0], [0, 1]])
    w0 = np.array([-2., 2.]).reshape(2, 1)

    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), 
                       np.arange(ymin, ymax + ystep, ystep))

    z = J(x, y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contour(x, y, z, levels=np.logspace(0, 5, 35), cmap='jet')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title('Contour plot w/ $\mathbf{J}(\mathbf{W}) = (\mathbf{w} - \mathbf{w}_0)^T \mathbf{A} (\mathbf{w} - \mathbf{w}_0)$')
    plt.show()

    return None


"""
Quiver plot for \nabla J(W) - 2Aw - 2Aw_0
"""

def quiver_plot():
    # generate data
    xmin, xmax, xstep = -4, 0, .1
    ymin, ymax, ystep = 0, 4, 0.1

    A = np.array([[2, 0], [0, 1]])
    w0 = np.array([-2., 2.]).reshape(2, 1)
    
    x1, y1 = np.meshgrid(np.arange(xmin, xmax, 0.2), np.arange(ymin, ymax, 0.2))

    J = lambda x, y: A[0, 0]*(x - w0[0])**2 + (A[0, 1] + A[1, 0])*(x - w0[0])*(y - w0[1]) + A[1, 1]*(y - w0[1])**2
    gradient_u = lambda x, y: (A[0, 0]*(x - w0[0]) + A[0, 1]*(y - w0[1])) + (A[0, 0] * (x - w0[0]) + A[1, 0]*(y - w0[1]))
    gradient_v = lambda x, y: (A[1, 0]*(x - w0[0]) + A[1, 1]*(y - w0[1])) + (A[0, 1] * (x - w0[0]) + A[1, 1]*(y - w0[1]))

    u1 = gradient_u(x1, y1)
    v1 = gradient_v(x1, y1)

    # quiver plotting
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(x1, y1, u1, v1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    ax.set_title(r'Quiver plot for $\nabla \mathbf{J}(\mathbf{w}) = 2\mathbf{Aw} - 2 \mathbf{Aw}_0$')
    plt.show()

    return None


"""
Quiver Contour plot for for \nabla J(W) - 2Aw - 2Aw_0
"""

def quiverContour_plot():
    # generate data
    xmin, xmax, xstep = -4, 0, .1
    ymin, ymax, ystep = 0, 4, 0.1

    A = np.array([[2, 0], [0, 1]])
    w0 = np.array([-2., 2.]).reshape(2, 1)
    
    x1, y1 = np.meshgrid(np.arange(xmin, xmax, 0.2), np.arange(ymin, ymax, 0.2))

    J = lambda x, y: A[0, 0]*(x - w0[0])**2 + (A[0, 1] + A[1, 0])*(x - w0[0])*(y - w0[1]) + A[1, 1]*(y - w0[1])**2
    gradient_u = lambda x, y: (A[0, 0]*(x - w0[0]) + A[0, 1]*(y - w0[1])) + (A[0, 0] * (x - w0[0]) + A[1, 0]*(y - w0[1]))
    gradient_v = lambda x, y: (A[1, 0]*(x - w0[0]) + A[1, 1]*(y - w0[1])) + (A[0, 1] * (x - w0[0]) + A[1, 1]*(y - w0[1]))

    u1 = gradient_u(x1, y1)
    v1 = gradient_v(x1, y1)
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                    np.arange(ymin, ymax + ystep, ystep))
    z = J(x, y)

    # display quiver & contour figure
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contour(x, y, z, levels=np.logspace(0, 5, 35), cmap='jet')
    ax.quiver(x1, y1, u1, v1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_title(r'Quiver-Contour plot w/ $\nabla \mathbf{J}(\mathbf{w}) = 2\mathbf{Aw} - 2 \mathbf{Aw}_0$',  fontsize=14, )
    plt.show()

    return None


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def normal3D_plot():
    """Gaussian 3D plot w/
    \mu = [0, 0]^T, \Sigma = [[1,0], [0, 1]]"""

    # # generate data 
    Mu = np.array([0, 0])
    Cov = np.array([[1, 0], [0, 1]])
    rv = multivariate_normal(Mu, Cov)
    
    xmin, xmax, xstep = -5, 5, 0.2
    ymin, ymax, ystep = -5, 5, 0.2

    # plot 3D Gaussian 
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(xmin, xmax, xstep)
    Y = np.arange(ymin, ymax, ystep)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    Z = rv.pdf(pos)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
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
    # bar_plot()

    # multiple axes
    # maxes_plot()

    # scatter plot
    # scatter_plot()

    # contour plot
    # contour_plot()

    # Quiver plot
    # quiver_plot()

    # Quiver & Contour plot
    # quiverContour_plot()

    # Gaussian 3D plot
    normal3D_plot()

    # input("\nPress Enter to continue ...")


    return None


if __name__ == "__main__":

    print("\nStarting Topic 03 Matplotlib Totorial ...")

    main()

    print("\nEnd of Topic 03 Matplotlib Totorial ...")
