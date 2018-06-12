
# coding: utf-8

# In[ ]:


from datascience import *
import numpy as np

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## A/B Testing

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

# In[ ]:


baby = Table.read_table('baby.csv')
baby


# In[ ]:


smoking_and_birthweight = baby.select('Maternal Smoker', 'Birth Weight')
smoking_and_birthweight


# In[ ]:


smoking_and_birthweight.group('Maternal Smoker')


# In[ ]:


smoking_and_birthweight.hist('Birth Weight', group='Maternal Smoker')


# In[ ]:


means_tbl = smoking_and_birthweight.group('Maternal Smoker', np.average)
means_tbl


# In[ ]:


means = means_tbl.column(1)
observed_difference = means.item(0) - means.item(1)
observed_difference


# In[ ]:


weights = smoking_and_birthweight.select('Birth Weight')
weights


# In[ ]:


weights.sample(with_replacement=False)


# In[ ]:


shuffled_weights = weights.sample(with_replacement=False).column(0)


# In[ ]:


original_and_shuffled = smoking_and_birthweight.with_column(
    'Shuffled Birth Weight', shuffled_weights
)


# In[ ]:


original_and_shuffled


# In[ ]:


original_and_shuffled.group('Maternal Smoker', np.average)


# In[ ]:


group_labels = baby.select('Maternal Smoker')
group_labels


# In[ ]:


# array of shuffled weights

# table with shuffled weights assigned to group labels

# array of means of the two groups

# difference between means of the two groups


# In[ ]:


shuffled_weights = weights.sample(with_replacement=False).column(0)
shuffled_tbl = group_labels.with_column('Shuffled Weight', shuffled_weights)
means = shuffled_tbl.group('Maternal Smoker', np.average).column(1)
new_difference = means.item(0) - means.item(1)
new_difference


# In[ ]:


differences = make_array()

for i in np.arange(5000):
    shuffled_weights = weights.sample(with_replacement = False).column(0)
    shuffled_tbl = group_labels.with_column('Shuffled Weight', shuffled_weights)
    means = shuffled_tbl.group('Maternal Smoker', np.average).column(1)
    new_difference = means.item(0) - means.item(1)
    differences = np.append(differences, new_difference)
    


# In[ ]:


Table().with_column('Difference Between Means', differences).hist(bins=20, ec='w')


# In[ ]:


observed_difference


# ## Deflategate

# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

# In[ ]:


football = Table.read_table('deflategate.csv')
football.show()


# In[ ]:


football = football.drop(1, 2).with_column(
    'Combined', (football.column(1) + football.column(2)) / 2
)


# In[ ]:


football.show()


# In[ ]:


np.ones(5)


# In[ ]:


start = np.append(12.5 * np.ones(11), 13 * np.ones(4))
start


# In[ ]:


drops = start - football.column(1)
drops


# In[ ]:


football = football.select('Team').with_column(
    'Drop', drops
)


# In[ ]:


football.show()


# In[ ]:


means_tbl = football.group('Team', np.average)


# In[ ]:


means_tbl


# In[ ]:


means = means_tbl.column(1)
observed_difference = means.item(0) - means.item(1)


# In[ ]:


observed_difference


# In[ ]:


group_labels = football.select('Team')
drop_tbl = football.select('Drop')


# In[ ]:


shuffled_drops = drop_tbl.sample(with_replacement=False).column(0)
shuffled_tbl = group_labels.with_column('Shuffled Drop', shuffled_drops)
means = shuffled_tbl.group('Team', np.average).column(1)
new_difference = means.item(0) - means.item(1)
new_difference


# In[ ]:


differences = make_array()

for i in np.arange(20000):
    shuffled_drops = drop_tbl.sample(with_replacement=False).column(0)
    shuffled_tbl = group_labels.with_column('Shuffled Drop', shuffled_drops)
    means = shuffled_tbl.group('Team', np.average).column(1)
    new_difference = means.item(0) - means.item(1)
    differences = np.append(differences, new_difference)
    


# In[ ]:


Table().with_column('Difference Between Means', differences).hist(ec='w')
plots.scatter(observed_difference, 0, color='red', s=40);


# In[ ]:


np.count_nonzero(differences <= observed_difference) / 20000

