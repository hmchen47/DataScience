
# coding: utf-8

from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# ## Decisions and Uncertainty

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

scores = Table.read_table('scores_by_section.csv')
scores

scores.group('Section')

scores.group('Section', np.average).show()

# Null: The Section 3 average is like the average of 27 random scores from the class.

# Alternative: No, it's too low.

# observed statistic

observerd_average = 13.6667

np.average(scores.sample(27, with_replacement=False).column('Midterm'))

averages = make_array()

repetitions = 50000
for i in np.arange(repetitions):
    new_average = np.average(scores.sample(27, with_replacement=False).column('Midterm'))
    averages = np.append(averages, new_average)

Table().with_column('Random Sample Average', averages).hist(bins = 25, ec='w')
plots.scatter(observerd_average, 0, color='red', s=30);

np.count_nonzero(averages <= observerd_average) / repetitions

np.count_nonzero(averages <= 13.6) / repetitions

Table().with_column('Random Sample Average', averages).hist(bins = 25, ec='w')
plots.scatter(observerd_average, 0, color='red', s=30)
plots.plot([13.6, 13.6], [0, 0.35], color='gold', lw=2);

