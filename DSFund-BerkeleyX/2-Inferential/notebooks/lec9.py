
# coding: utf-8

from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# ## Causality

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

bta = Table.read_table('bta.csv')
bta.show()

bta.group('Group')

bta.group('Group', sum)

bta.group('Group', np.average)

observed_outcomes = Table.read_table('observed_outcomes.csv')
observed_outcomes.show()

bta 

obs_proportions = bta.group('Group', np.average).column(1)
obs_proportions

observed_distance = abs(obs_proportions.item(0) - obs_proportions.item(1))
observed_distance

bta

labels = bta.select('Group')
results = bta.select('Result')

shuffled_results = results.sample(with_replacement=False).column(0)
shuffled_tbl = labels.with_column('Shuffled Result', shuffled_results)
proportions = shuffled_tbl.group('Group', np.average).column(1)
new_distance = abs(proportions.item(0) - proportions.item(1))
new_distance

distances = make_array()

for i in np.arange(20000):
    shuffled_results = results.sample(with_replacement=False).column(0)
    shuffled_tbl = labels.with_column('Shuffled Result', shuffled_results)
    proportions = shuffled_tbl.group('Group', np.average).column(1)
    new_distance = abs(proportions.item(0) - proportions.item(1))
    distances = np.append(distances, new_distance)

Table().with_column('Distance', distances).hist(bins=np.arange(0, 1, 0.1), ec='w')
plots.scatter(observed_distance, 0, color='red', s=40);

np.count_nonzero(distances >= observed_distance) / 20000

