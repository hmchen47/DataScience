
# coding: utf-8

    from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline') 

# ## Ounces Per Day

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

births = Table.read_table('baby.csv')
births.show(3) 

babies = births.select('Birth Weight', 'Gestational Days')
babies 

babies.scatter(1, 0) 

ratios = babies.with_column(
'Ratio BW/GD', babies.column(0)/babies.column(1)
)
ratios 

ratios.hist('Ratio BW/GD') 

np.median(ratios.column('Ratio BW/GD')) 

resampled_medians = []
for i in np.arange(1000):
    resample = ratios.sample()
    median = np.median(resample.column('Ratio BW/GD'))
    resampled_medians.append(median)
    
interval_99 = [percentile(0.5, resampled_medians),
               percentile(99.5, resampled_medians)]
print(interval_99)

Table().with_column('Resampled median', resampled_medians).hist(0)
plots.plot(interval_99, [0, 0], color='gold', lw=10);

