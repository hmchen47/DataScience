
# coding: utf-8

from datascience import *
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

def r_scatter(r):
    plots.figure(figsize=(5,5))
    "Generate a scatter plot with a correlation approximately r"
    x = np.random.normal(0, 1, 1000)
    z = np.random.normal(0, 1, 1000)
    y = r*x + (np.sqrt(1-r**2))*z
    plots.scatter(x, y, color='darkblue', s=20)
    plots.xlim(-4, 4)
    plots.ylim(-4, 4)

# ## Visualization ##
# **Please run all cells before this cell, including the import cell at the top of the notebook.**

galton = Table.read_table('galton.csv')

heights = Table().with_columns(
    'MidParent', galton.column('midparentHeight'),
    'Child', galton.column('childHeight')
    )
heights

heights.scatter('MidParent')

hybrid = Table.read_table('hybrid.csv')
hybrid

hybrid.scatter('mpg', 'msrp')
hybrid.scatter('acceleration', 'msrp')

suv = hybrid.where('class', 'SUV')
suv.num_rows
suv.scatter('mpg', 'msrp')

def standard_units(x):
    "Convert any array of numbers to standard units."
    return (x - np.average(x)) / np.std(x)

Table().with_columns(
    'mpg (standard units)',  standard_units(suv.column('mpg')), 
    'msrp (standard units)', standard_units(suv.column('msrp'))
).scatter(0, 1)
plots.xlim(-3, 3)
plots.ylim(-3, 3);

# ## Calculation ##
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**
# Draws a scatter diagram of variables that have the specified correlation
r_scatter(0.6)

r_scatter(0)

# ### Calculating $r$ ###
x = np.arange(1, 7, 1)
y = make_array(2, 3, 1, 5, 2, 7)
t = Table().with_columns(
        'x', x,
        'y', y
    )
t

t.scatter('x', 'y', s=30, color='red')

t= t.with_columns(
        'x (standard units)', standard_units(x),
        'y (standard units)', standard_units(y)
    )
t

su_product = t.column(2) * t.column(3)
t = t.with_column('product of standard units', su_product)
t

# r is the average of the products of standard units
r = np.mean(t.column(4))
r

def correlation(tbl, x, y):
    """tbl is a table; 
    x and y are column labels"""
    x_in_standard_units = standard_units(tbl.column(x))
    y_in_standard_units = standard_units(tbl.column(y))
    return np.average(x_in_standard_units * y_in_standard_units)  

correlation(t, 'x', 'y')
correlation(suv, 'mpg', 'msrp')
correlation(t, 'x', 'y')
correlation(t, 'y', 'x')

t.scatter('x', 'y', s=30, color='red')
t.scatter('y', 'x', s=30, color='red')

correlation(t, 'y', 'x')

# ## Interpretation ##
# **Please run all cells before this cell, including the previous example cells and the import cell at the top of the notebook.**
# ### Nonlinearity ###
new_x = np.arange(-4, 4.1, 0.5)
nonlinear = Table().with_columns(
        'x', new_x,
        'y', new_x**2
    )
nonlinear.scatter('x', 'y', s=30, color='r')

correlation(nonlinear, 'x', 'y')

# ### Outliers ###
line = Table().with_columns(
        'x', make_array(1, 2, 3, 4),
        'y', make_array(1, 2, 3, 4)
    )
line.scatter('x', 'y', s=30, color='r')

correlation(line, 'x', 'y')

outlier = Table().with_columns(
        'x', make_array(1, 2, 3, 4, 5),
        'y', make_array(1, 2, 3, 4, 0)
    )
outlier.scatter('x', 'y', s=30, color='r')

correlation(outlier, 'x', 'y')

# ### Ecological Correlation ###
sat2014 = Table.read_table('sat2014.csv').sort('State')
sat2014

sat2014.scatter('Critical Reading', 'Math')

correlation(sat2014, 'Critical Reading', 'Math')

