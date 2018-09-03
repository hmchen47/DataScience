
# coding: utf-8

# In[ ]:


from datascience import *
import numpy as np

import matplotlib.pyplot as plots
from mpl_toolkits.mplot3d import Axes3D
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Classification Examples: Medicine

# **Please run all cells before this cell, including the import cell at the top of the notebook.**

# In[ ]:


ckd = Table.read_table('ckd.csv').relabeled('Blood Glucose Random', 'Glucose')
ckd.show(3)


# In[ ]:


ckd.group('Class')


# In[ ]:


ckd.scatter('White Blood Cell Count', 'Glucose', colors='Class')


# In[ ]:


ckd.scatter('Hemoglobin', 'Glucose', colors='Class')


# ## Classification Examples: Counterfeit Banknotes

# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**

# In[ ]:


banknotes = Table.read_table('banknote.csv')
banknotes


# In[ ]:


banknotes.scatter('WaveletVar', 'WaveletCurt', colors='Class')


# In[ ]:


banknotes.scatter('WaveletSkew', 'Entropy', colors='Class')


# In[ ]:


fig = plots.figure(figsize=(8,8))
ax = Axes3D(fig)
ax.scatter(banknotes.column('WaveletSkew'), 
           banknotes.column('WaveletVar'), 
           banknotes.column('WaveletCurt'), 
           c=banknotes.column('Class'),
           cmap='viridis',
           s=50);

