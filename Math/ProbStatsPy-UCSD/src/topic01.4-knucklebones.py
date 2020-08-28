#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# Enter code here

def rand2sym(vec, prob, sym):
    """Transform an array of random numbers, distributed uniformly in [0, 1] into a 
    sequence of symbols, chosen according to the probabilities defined by c (cum of p)

    Arguments:
        vec {[float]} -- [a list of floating numbers]
        prob {[float]} -- [a list of probabilities for the uniform dist]
    """
    ans = []
    cum = np.cumsum(prob)        # return cumulative sum of the elements
    counts = {i:0 for i in range(4)}
    for x in vec:
        for i in range(len(cum)-1):
            if x >= cum[i] and x < cum[i+1]:
                ans.append(sym[i])
                counts[i] += 1
                break
    return ans, counts

def knuckle_trials(trials, sym, prob):

    # 1000 trials of knuckle bone
    n = trials
    R = np.random.rand(n)
    _syms, counts = rand2sym(R, prob, sym)

    # print the symbols w/ the given random trials
    print(''.join(_syms))

    # print counts
    f = [float(y)/n for x, y in counts.items()]
    print("\n")

    # print 'number of trials (n) =', n
    for i in range(4):
        print("{} probability={:3.2f} frequency= {:d}/{:d} = {:3.2f}".format(sym[i], prob[i+1], counts[i], n, f[i]))

    return None

def main():

    # create list of color symbol
    red_bck = "\x1b[41m%s\x1b[0m"
    green_bck = "\x1b[42m%s\x1b[0m"
    tan_bck = "\x1b[43m%s\x1b[0m"
    blue_bck = "\x1b[44m%s\x1b[0m"

    sym = [red_bck%'6', green_bck%'1', tan_bck%'3', blue_bck%'4']
    # print("Symbols= {} {} {} {}".format(sym[0], sym[1], sym[2], sym[3]))

    # probability settings
    p = [0.0, 0.1, 0.2, 0.3, 0.4]
    for i in range(4):
        print('symbol={}, probability = {:5.3f}'.format(sym[i], p[i]))
    input("\nPress Enter to continue ...\n")


    runs = [1000, 100, 10]

    for run in runs:
        knuckle_trials(run, sym, p)
        input("\nPress Enter to continue ...\n")


    return None


if __name__ == "__main__":
    print("\nStarting Topic 01.4 Python code execution ...\n")

    main()

    print("\nEnd Topic 01.4 Python code execution ...")


