
# coding: utf-8

from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# ## Percentiles

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

v = [1, 7, 3, 9, 5]
v

percentile(25, v)

percentile(50, v)

percentile(99, v)

# ## Estimation

# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

sf = Table.read_table('san_francisco_2015.csv').select(3, 11, 21)
sf.set_format('Total Compensation', NumberFormatter(0))
sf = sf.where('Total Compensation', are.above(10000))
sf.show(3)

sf.sort('Total Compensation')

sf.sort('Total Compensation', descending=True)

comp_bins = np.arange(0, 700000, 25000)
sf.hist('Total Compensation', bins=comp_bins, unit="dollar")

percentile(50, sf.column('Total Compensation'))

sample_from_population = sf.sample(200, with_replacement=False)
sample_from_population.show(3)

percentile(50, sample_from_population.column('Total Compensation'))

np.median(sf.column('Total Compensation'))

np.median(sample_from_population.column('Total Compensation'))

medians = []
repetitions = np.arange(100)
for i in repetitions:
    sample = sf.sample(200, with_replacement=False)
    median = np.median(sample.column('Total Compensation'))
    medians.append(median)
    
Table().with_columns('trial', repetitions, 'median', medians).scatter('trial')

Table().with_column('medians', medians).hist(0)

# ## The Bootstrap

# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

sample_from_population = sf.sample(200, with_replacement=False)
sample_from_population.show(3)

np.median(sample_from_population.column('Total Compensation'))

resample = sample_from_population.sample()

np.median(resample.column('Total Compensation'))

medians = []

for i in np.arange(1000):
    resample = sample_from_population.sample()
    median = np.median(resample.column('Total Compensation'))
    medians.append(median)
    
Table().with_column('Reampled median', medians).hist()

percentile(2.5, medians)

percentile(97.5, medians)

percentile(0.5, medians)

percentile(99.5, medians)

intervals = Table(['Lower', 'Upper'])

for j in np.arange(100):
    sample_from_population = sf.sample(200, with_replacement=False)
    medians = []
    for i in np.arange(1000):
        resample = sample_from_population.sample()
        median = np.median(resample.column('Total Compensation'))
        medians.append(median)
        
    interval_95 = [percentile(2.5, medians),
                   percentile(97.5, medians)]
    
    intervals.append(interval_95)

truth = np.median(sf.column('Total Compensation'))
correct = intervals.where('Lower', are.not_above(truth)).where('Upper', are.not_below(truth))
correct.num_rows

intervals.where('Lower', are.above(truth))

intervals.where('Upper', are.below(truth))

