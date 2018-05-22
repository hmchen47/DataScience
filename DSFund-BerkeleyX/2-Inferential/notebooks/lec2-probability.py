
# coding: utf-8

from datascience import *
import numpy as np


# ## The Monty Hall Problem

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

goats = make_array('first goat', 'second goat')
hidden_behind_doors = np.append(goats, 'car')
hidden_behind_doors

def other_goat(goat):
    if goat == 'first goat':
        return 'second goat'
    if goat == 'second goat':
        return 'first goat'

other_goat("first goat")
other_goat("second goat")
other_goat("sheep")

contestant_goat = np.random.choice(hidden_behind_doors)
contestant_goat

def monty_hall_game():
    """[contestant's guess, revealed, remains]"""
    contestant_guess = np.random.choice(hidden_behind_doors)
    
    if contestant_guess == 'first goat':
        return ['first goat', 'second goat', 'car']
    elif contestant_guess == 'second goat':
        return ['second goat', 'first goat', 'car']
    elif contestant_guess == 'car':
        revealed = np.random.choice(goats)
        return ['car', revealed, other_goat(revealed)]

monty_hall_game()

trials = Table(['guess', 'revealed', 'remains'])
trials

trials.append(monty_hall_game())

trials.append(monty_hall_game())
trials.append(monty_hall_game())
trials.append(monty_hall_game())
trials.append(monty_hall_game())
trials


for i in np.arange(9995):
    trials.append(monty_hall_game())

trials.pivot('remains', 'guess')
trials.group('guess')
trials.group('remains')

