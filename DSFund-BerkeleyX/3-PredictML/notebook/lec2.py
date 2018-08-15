
# coding: utf-8

from datascience import *
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')


# ## Standard Units ##

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

def standard_units(x):
    """Convert the array x to standard units"""
    return (x - np.average(x)) / np.std(x)


births = Table.read_table('baby.csv')

births.labels

ages = births.column('Maternal Age')

ages_in_standard_units = standard_units(ages)

np.average(ages_in_standard_units), np.std(ages_in_standard_units)

both = Table().with_column(
    'Age in Years', ages,
    'Age in Standard Units', ages_in_standard_units

)
both

np.mean(ages), np.std(ages)

both.hist('Age in Years', bins = np.arange(15, 46, 2))

both.hist('Age in Standard Units', bins = np.arange(-2.2, 3.4, 0.35))
plots.xlim(-2, 3.1);

# ## The SD and Bell Shaped Curves ##
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

births.hist('Maternal Height', bins = np.arange(56.5, 72.6, 1), ec = 'w')

heights = births.column('Maternal Height')
np.average(heights), np.std(heights)

births.hist('Birth Weight', ec = 'w')

bw = births.column('Birth Weight')
np.average(bw), np.std(bw)

# ## Central Limit Theorem ##
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

united = Table.read_table('united_summer2015.csv')
united

united.hist('Delay', bins = np.arange(-20, 300, 10), ec='w')

sample_size = 500

averages = make_array()

for i in np.arange(10000):
    sampled_flights = united.sample(sample_size)
    sample_average = np.average(sampled_flights.column('Delay'))
    averages = np.append(averages, sample_average)

Table().with_column('Sample Average', averages).hist(bins = 25, ec='w')
plots.title('Sample Averages: Sample Size ' + str(sample_size))
plots.xlabel('Random Sample Average');

# Population average
pop_ave = np.average(united.column('Delay'))
pop_ave

