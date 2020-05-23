#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def largest_face(dice, x_max, debug=False):
    """calculate the probability of x_max

    assume n dices
    P(X <= x) = (1/dice[0])(1/dice[1]) ... (1/dice[n-1]) - P(X <= x-1)

    Arguments:
        dice {list} -- list of faces of he dice
        x_max {int} -- the maximum number shown on each experiment

    Keyword Arguments:
        debug {bool} -- turn on/off debuging msg (default: {False})
    """

    mface = max(dice)
    n = len(dice)

    prob = [[1, 0.0] for _ in range(mface+1)]     # 1st/2nd as the cumulative and density prob
    prob[0][0] = 0

    base = 1
    for face in dice:
        base *= face

    if debug: print("base= {}".format(base))

    for idx in range(1, mface+1):
        for faces in dice:
            prob[idx][0] *= min(idx, faces)

        prob[idx][1] = (prob[idx][0] - prob[idx-1][0]) / base
        if debug: print("iter({}): {}".format(idx, prob[idx]))

    if debug:
        print("Prob: {}".format(prob))

    return prob[x_max][1]


def main():

    # Problem 1
    dice1, x_max1 = [2, 5, 8], 8
    dice2, x_max2 = [2], 1
    dice3, x_max3 = [3, 4], 2
    dice4, x_max4 = [2, 5, 7, 3], 3
    dice5, x_max5 = [5], 3
    dice6, x_max6 = [11, 5, 4], 5
    dice7, x_max7 = [1, 10], 3
    dice8, x_max8 = [7, 4], 2

    print("dice= {}, x_max= {}: {}".format(dice1, x_max1, largest_face(dice1, x_max1, False)))
    print("dice= {}, x_max= {}: {}".format(dice2, x_max2, largest_face(dice2, x_max2)))
    print("dice= {}, x_max= {}: {}".format(dice3, x_max3, largest_face(dice3, x_max3)))
    print("dice= {}, x_max= {}: {}".format(dice4, x_max4, largest_face(dice4, x_max4)))
    print("dice= {}, x_max= {}: {}".format(dice5, x_max5, largest_face(dice5, x_max5)))
    print("dice= {}, x_max= {}: {}".format(dice6, x_max6, largest_face(dice6, x_max6)))
    print("dice= {}, x_max= {}: {}".format(dice7, x_max7, largest_face(dice7, x_max7)))
    print("dice= {}, x_max= {}: {}".format(dice8, x_max8, largest_face(dice8, x_max8)))


    # problem 2

    return None

if __name__ == "__main__":

    print("\nEntering Topic 5 Intro to Probability HW ...")

    main()

    print("\nEnd Topic 5 Intro to Probability HW ...\n")

