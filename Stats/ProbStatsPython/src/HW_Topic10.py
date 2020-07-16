#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 

from math import exp, sqrt, factorial, pi

def Markov(n, p, c):
    """compute Markov inequality

    \[ P(X \ge \alpha \mu) \le \frac{1}{\alpha} \quad \forall\, \alpha \ge 1 \]

    compute $P(X \ge c \cdot np)$ for $c > 1$

    $\alpha = c \implies P(X \ge c \cdot np) \le \frac 1 c$

    Args:
        n (int): number of trials
        p (float): probability of success trial
        c (float): bound of X
    """
    if c < 1.0:
        return "c must greater than 1: c= {}".format(c)
    else:
        return 1/c


def Chebyshev(n, p, c):
    """Compute Chebyshev inequality

    \[]begin{align*}
      P(|X - \mu | \ge \alpha \sigma) \le \frac{1}{\alpha^2} \quad \forall \alpha \ge 1 \\
      P( \ge \mu + \alpha \sigma) \le P(|X - \mu| \ge \alpha \sigma) \le \frac{1}{\alpha^2}
    \end{align*}\]

    $X \ge c \cdot np \to X - np \ge (c-1)np = \alpha \sigma \to \alpha = (c-1) * (\sqrt(np/(1-p)))$
    
    Args:
        n (int): number of trials
        p (float): probability of success trial
        c (float): bound of X
    """
    if c < 1.0:
        return "c must greater than 1: c= {}".format(c)
    else:
        return (1.0-p) / ((c-1)**2 * n*p)



def Chernoff(n, p, c):
    """compute Chernoff inequality

    \[ P(X \ge (1+\delta) \mu) \le e^{-\frac{\delta^2}{2+\delta} \mu} \]

    $(1+\delta) \mu = c \cdot np \to \delta = (c-1)

    Args:
        n (int): number of trials
        p (float): probability of success trial
        c (float): bound of X
    """
    if c < 1.0:
        return "c must greater than 1: c= {}".format(c)
    else:
        return exp(-(c - 1.0)**2.0 * n*p /(2.0 + c - 1.0))  



def main():

    # parameters for Binomial distribution
    # n: number of 
    n, p, c = 100, 0.2, 1.5

    print("\nBinomial distribution: n ={}, p= {}, c = {}".format(n, p, c))

    # Markov inequality
    print("  Markov inequality: {}".format(Markov(n, p, c)))

    # Chenbyshev inequality
    print("  Chenbyshev inequality: {}".format(Chebyshev(n, p, c)))


    # Chernoff bound
    print("  Chernoff inequality: {}".format(Chernoff(n, p, c)))

    n, p, c = 200, 0.25, 1.25
    print("\nMarkov inequality (n, p, c) = ({}, {}, {}): {}".format(n, p, c, Markov(n, p, c)))

    n, p, c = 100, 0.25, 1.25
    print("\nChebyshev inequality (n, p, c) = ({}, {}, {}): {}".format(n, p, c, Chebyshev(n, p, c)))

    n, p, c = 100, 0.25, 1.25
    print("\nChrenoff inequality (n, p, c) = ({}, {}, {}): {}".format(n, p, c, Chernoff(n, p, c)))

    n, p, c = 200, 0.3, 1.1
    print("\nMarkov inequality (n, p, c) = ({}, {}, {}): {}".format(n, p, c, Markov(n, p, c)))

    return None


if __name__ == "__main__":

  print("\nStarting Topic 10 HW .......")

  main()

  print("\nEnd Topic 10 HW ...........")

