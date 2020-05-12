#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

def main():

    # set seed for the random number generator
    np.random.seed(1)
    # np.random.seed(time.time())

    # generate uniform distribution over [0, 1]
    print('\nrandom numbers (2x3) w/ Uniform distribution: np.random.rand(2, 3)\n{}'.format(np.random.rand(2, 3)))
    print('\nrandom numbers (2x3) w/ Uniform distribution: np.random.rand(2, 3)\n{}'.format(np.random.rand(2, 3)))

    input('\nPress Enter to continue ...')

    # generate integers ranging from low (inclusive) to high (exclusive)
    print('\nrandom numbers (2x3) w/ integer range: np.random.randint(low=0, high=4, size=(2, 3))\n{}'.format(np.random.randint(low=0, high=4, size=(2, 3))))

    input('\nPress Enter to continue ...')

    # geneate random numbers following a given pmf
    num = np.arange(4)
    pmf = [0.1, 0.2, 0.3, 0.4]
    print('\nrandom numbers (3x4) w/ a given pmf: np.random.choice(a=num, size=(3, 4), p=pmf)\n{}'.format(np.random.choice(a=num, size=(3, 4), p=pmf)))

    num = ['Spade', 'Heart', 'Diamond', 'Club']
    pmf = [0.25, 0.25, 0.25, 0.25]
    print('\nrandom numbers (3x4) w/ a given pmf: np.random.choice(a=num, size=(3, 4), p=pmf)\n{}'.format(np.random.choice(a=num, size=(3, 4), p=pmf)))

    input('\nPress Enter to continue ...')

    # exponential distribution
    print('\nrandom numbers (2x3) w/ exponentail dist: np.random.exponential(scale = 2, size = (2, 3))\n{}'.format(np.random.exponential(scale = 2, size = (2, 3))))



    input('\nPress Enter to continue ...')

    return None


if __name__ == "__main__":

    print('\nStarting Topic 3 Numpy Random library ...')

    main()

    print('\nEnd Topic 3 Numpy Random library ...')

    

