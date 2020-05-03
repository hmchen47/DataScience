#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as numpy
import matplotlib.pyplot as plt


# Enter code here


def main():

    # create list of color symbol
    red_bck = "\x1b[41m%s\x1b[0m"
    green_bck = "\x1b[42m%s\x1b[0m"
    tan_bck = "\x1b[43m%s\x1b[0m"
    blue_bck = "\x1b[44m%s\x1b[0m"

    sym = [red_bck%'6', green_bck%'1', tan_bck%'3', blue_bck%'4']
    # print("Symbols= {} {} {} {}".format(sym[0], sym[1], sym[2], sym[3]))

    # input("Press Enter to continue ...")

    return None


if __name__ == "__main__":
    print("\nStarting Topic 01.4 Python code execution ...")

    main()

    print("\nEnd Topic 01.4 Python code execution ...")


