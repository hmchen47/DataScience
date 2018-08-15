
# coding: utf-8

from datascience import *
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')


# ## Introduction: Average (Mean) ##

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

values = make_array(2, 3, 3, 9)


sum(values) / len(values), np.average(values), np.mean(values)

(2 + 3 + 3 + 9) / 4

2 * (1/4) + 3 * (2/4) + 9 * (1/4)

2 * 0.25 + 3 * 0.5 + 9 * 0.25

values_table = Table().with_columns('Value', values)
values_table

bins_for_display = np.arange(0.5, 10.6, 1)

values_table.hist(bins = bins_for_display, ec = 'w')

2 * np.ones(10)

twos = 2 * np.ones(10)
threes = 3 * np.ones(20)
nines = 9 * np.ones(10)

new_values = np.append(np.append(twos, threes), nines)

len(new_values)

new_values_table = Table().with_column('Value', new_values)
new_values_table.hist(bins = bins_for_display)

np.average(new_values), np.average(values)

# ## The Average and the Median ##
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

nba = Table.read_table('nba2013.csv')

nba

nba.hist('Height', bins=np.arange(65.5, 90.5), ec='w')

heights = nba.column('Height')
percentile(50, heights), np.average(heights)

# ## Standard Deviation ##

# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

sd_table = Table().with_columns('Value', values)
sd_table

average = np.average(values)
average

deviations = values - average
sd_table = sd_table.with_column('Deviation', deviations)
sd_table

sum(deviations)

sd_table = sd_table.with_column('Squared Deviation', deviations ** 2)
sd_table

# Variance of the data is the average of the squared deviations

variance = np.average(sd_table.column('Squared Deviation'))
variance

# Standard Deviation (SD) is the square root of the variance

sd = variance ** 0.5
sd

np.std(values)

# ## Chebyshev's Bounds ##
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

births = Table.read_table('baby.csv')
births

births.hist('Maternal Pregnancy Weight')

mpw = births.column('Maternal Pregnancy Weight')
average = np.average(mpw)
sd = np.std(mpw)
average, sd

within_3_SDs = births.where('Maternal Pregnancy Weight', are.between(average - 3*sd, average + 3*sd))

within_3_SDs.num_rows / births.num_rows

# Chebyshev's bound for the proportion in the range "average plus or minus 3 SDs"
# is at least

1 - 1/3**2

births.hist(overlay = False)

# See if Chebyshev's bounds work
# for different shapes of distributions

for k in births.labels:
    values = births.column(k)
    average = np.average(values)
    sd = np.std(values)
    print()
    print(k)
    for z in np.arange(2, 6):
        chosen = births.where(k, are.between(average - z*sd, average + z*sd))
        proportion = chosen.num_rows / births.num_rows
        percent = round(proportion * 100, 2)
        print('Average plus or minus', z, 'SDs:', percent, '%')

