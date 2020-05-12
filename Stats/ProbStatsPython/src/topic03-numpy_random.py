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


    return None


if __name__ == "__main__":

    print('\nStarting Topic 3 Numpy Random library ...')

    main()

    print('\nEnd Topic 3 Numpy Random library ...')

    

