
# coding: utf-8




from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Sampling

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

top = Table.read_table('top_movies_2017.csv')
top = top.with_column('Row Index', np.arange(top.num_rows)).move_to_start('Row Index')
top.set_format(['Gross', 'Gross (Adjusted)'], NumberFormatter)

top.take([3, 5, 7])

top.where('Title', are.containing('and the'))

start = np.random.choice(np.arange(10))
top.take(np.arange(start, start + 5))

top.sample(5)

top.sample(50).group("Title")

top.sample(500).group('Title')

top.sample(5, with_replacement=False)

# ## Dice
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

die = Table().with_column('face', np.arange(6)+1)
die

def face_hist(t):
    t.hist('face', bins=np.arange(0.5, 7, 1), unit='face')
    plots.xlabel('Face')

face_hist(die)


# Try changing the sample size of 1000 to larger and smaller numbers
face_hist(die.sample(1000))

# ## Large Random Samples
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

united = Table.read_table('united.csv')
united

def delay_hist(t):
    t.hist('Delay', unit='minute', bins=np.arange(-30, 301, 10))
    
delay_hist(united)

delay_hist(united.sample(1000))

# ## Simulation
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

k = 4
1 - (5/6) ** 4

dice = np.arange(6) + 1
rolls = np.random.choice(dice, k)
rolls

sum(rolls == 6)

trials = 10000
successes = 0

for _ in np.arange(trials):
    rolls = np.random.choice(dice, k)
    if sum(rolls == 6) > 0:
        successes = successes + 1
        
successes / trials

# ## Statistics

# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

# This cell will take a long time to run

def estimate_by_simulation(trials):
    successes = 0

    for _ in np.arange(trials):
        rolls = np.random.choice(dice, k)
        if sum(rolls == 6) > 0:
            successes = successes + 1

    return successes / trials

estimates = []
for _ in np.arange(1000):
    estimates.append(estimate_by_simulation(10000))


Table().with_column('Estimate', estimates).hist(bins=50, normed=False)

