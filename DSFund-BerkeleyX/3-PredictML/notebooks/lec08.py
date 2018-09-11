
# coding: utf-8
from datascience import *
import numpy as np

import matplotlib.pyplot as plots
from mpl_toolkits.mplot3d import Axes3D
plots.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# Classification Examples: Medicine
ckd = Table.read_table('ckd.csv').relabeled('Blood Glucose Random', 'Glucose')
ckd.show(3)

ckd.group('Class')

ckd.scatter('White Blood Cell Count', 'Glucose', colors='Class')

ckd.scatter('Hemoglobin', 'Glucose', colors='Class')


# Classification Examples: Counterfeit Banknotes
banknotes = Table.read_table('banknote.csv')
banknotes

banknotes.scatter('WaveletVar', 'WaveletCurt', colors='Class')

banknotes.scatter('WaveletSkew', 'Entropy', colors='Class')

fig = plots.figure(figsize=(8,8))
ax = Axes3D(fig)
ax.scatter(banknotes.column('WaveletSkew'), 
           banknotes.column('WaveletVar'), 
           banknotes.column('WaveletCurt'), 
           c=banknotes.column('Class'),
           cmap='viridis',
           s=50);

