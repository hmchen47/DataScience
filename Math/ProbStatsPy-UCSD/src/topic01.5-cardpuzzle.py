#!/usr/bin/env python3
# -*- codingL utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 

def main():
    # setting card color symbols
    red_bck = "\x1b[41m%s\x1b[0m"
    blue_bck = "\x1b[44m%s\x1b[0m"
    red = red_bck%'R'
    blue = blue_bck%'B'

    # setup cards
    Cards = [(red, blue), (red, red), (blue, blue)]

    counts = {'same': 0, 'different': 0}
    
    for j in range(100):
        card_id = int(np.random.random() * 3)  # select a card
        side_id = int(np.random.random() * 2)   # select the side to face up

        card = Cards[card_id]
        if (side_id == 1):
            card = (card[1], card[0])
        same = 'same' if card[0] == card[1] else 'different'
            # count the no of times the two sides are the same or different
        counts[same] += 1

        print("{} {:10s}".format(''.join(card), same), end='')
        if (j+1)%5 == 0:
            print()
    print("\n{}".format(counts))

    return None

if __name__ == "__main__":
    print("\nStarting Topic 1.5 Three card puzzle ...\n")

    main()

    print("\nEnf of Topic 1.5 Three Cards Puzzle ...\n")

