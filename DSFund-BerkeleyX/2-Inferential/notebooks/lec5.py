
# coding: utf-8

# In[ ]:


from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## A Model about Random Selection: Swain vs. Alabama

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

# In[ ]:


eligible_population = make_array(0.26, 0.74)


# In[ ]:


sample_proportions(100, eligible_population)


# In[ ]:


both_counts = 100 * (sample_proportions(100, eligible_population))
both_counts


# In[ ]:


both_counts.item(0)


# In[ ]:


counts = make_array()

repetitions = 10000
for i in np.arange(repetitions):
    sample_distribution = sample_proportions(100, eligible_population)
    sampled_count = (100 * sample_distribution).item(0)
    counts = np.append(counts, sampled_count)


# In[ ]:


Table().with_column('Random Sample Count', counts).hist(bins = np.arange(5.5, 44.5, 1))


# ## A Genetic Model: Mendel's Pea Flowers

# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

# In[ ]:


model = make_array(0.75, 0.25)


# In[ ]:


sample_proportions(929, model)


# In[ ]:


percent_purple = (100 * sample_proportions(929, model)).item(0)


# In[ ]:


percent_purple


# In[ ]:


abs(percent_purple - 75)


# In[ ]:


distances = make_array()

repetitions = 10000
for i in np.arange(repetitions):
    one_distance = abs((100 * sample_proportions(929, model)).item(0) - 75)
    distances = np.append(distances, one_distance)


# In[ ]:


Table().with_column('Distance from 75', distances).hist()


# In[ ]:


abs(100 * (705 / 929) - 75)

