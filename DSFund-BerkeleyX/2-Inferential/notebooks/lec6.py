
# coding: utf-8

from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

np.set_printoptions(legacy='1.13')

# ## Comparing Distributions

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

jury = Table().with_columns(
    'Ethnicity', make_array('Asian', 'Black', 'Latino', 'White', 'Other'),
    'Eligible', make_array(0.15, 0.18, 0.12, 0.54, 0.01),
    'Panels', make_array(0.26, 0.08, 0.08, 0.54, 0.04)
)
jury

jury.barh('Ethnicity')

jury_with_diffs = jury.with_column('Difference', jury.column('Panels') - jury.column('Eligible'))

jury_with_diffs

jury_with_diffs = jury_with_diffs.with_column('Absolute Difference', np.abs(jury_with_diffs.column('Difference')))

jury_with_diffs

sum(jury_with_diffs.column('Absolute Difference'))

sum(jury_with_diffs.column('Absolute Difference')) / 2

def total_variation_distance(distribution_1, distribution_2):
    return sum(np.abs(distribution_1 - distribution_2)) / 2

total_variation_distance(jury.column('Panels'), jury.column('Eligible'))

eligible = jury.column('Eligible')

panels_and_sample = jury.with_column('Random Sample', sample_proportions(1453, eligible))

panels_and_sample

panels_and_sample.barh('Ethnicity')

total_variation_distance(panels_and_sample.column('Random Sample'), eligible)

total_variation_distance(jury.column('Panels'), eligible)

tvds = make_array()

repetitions = 10000
for i in np.arange(repetitions):
    sample_distribution = sample_proportions(1453, eligible)
    new_tvd = total_variation_distance(sample_distribution, eligible)
    tvds = np.append(tvds, new_tvd)

Table().with_column('Total Variation Distance', tvds).hist(bins = np.arange(0, 0.2, 0.005), ec='w')

total_variation_distance(jury.column('Panels'), eligible)

