
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # The Series Data Structure
import pandas as pd
get_ipython().magic('pinfo pd.Series')

animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)

numbers = [1, 2, 3]
pd.Series(numbers)

animals = ['Tiger', 'Bear', None]
pd.Series(animals)

numbers = [1, 2, None]
pd.Series(numbers)

import numpy as np
np.nan == None

np.nan == np.nan

np.isnan(np.nan)

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s

s.index

s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s

# # Querying a Series
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s

s.iloc[3]

s.loc['Golf']

s[3]

s['Golf']

sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)

s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead

s = pd.Series([100.00, 120.00, 101.00, 3.00])
s

total = 0
for item in s:
    total+=item
print(total)

import numpy as np

total = np.sum(s)
print(total)

#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()

len(s)

get_ipython().run_cell_magic('timeit', '-n 100', 'summary = 0\nfor item in s:\n    summary+=item')

get_ipython().run_cell_magic('timeit', '-n 100', 'summary = np.sum(s)')

s+=2 #adds two to each item in s using broadcasting
s.head()

for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()

get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')

get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\ns+=2')

s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)

original_sports

cricket_loving_countries

all_countries

all_countries.loc['Cricket']
# # The DataFrame Data Structure
import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()

df.loc['Store 2']

type(df.loc['Store 2'])

df.loc['Store 1']

df.loc['Store 1', 'Cost']

df.T

df.T.loc['Cost']

df['Cost']

df.loc['Store 1']['Cost']

df.loc[:,['Name', 'Cost']]

df.drop('Store 1')

df

copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df

get_ipython().magic('pinfo copy_df.drop')

del copy_df['Name']
copy_df

df['Location'] = None
df
# # Dataframe Indexing and Loading
costs = df['Cost']
costs

costs+=2
costs

df

get_ipython().system('cat olympics.csv')

df = pd.read_csv('olympics.csv')
df.head()

df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.head()

df.columns

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()
# # Querying a DataFrame
df['Gold'] > 0

only_gold = df.where(df['Gold'] > 0)
only_gold.head()

only_gold['Gold'].count()

df['Gold'].count()

only_gold = only_gold.dropna()
only_gold.head()

only_gold = df[df['Gold'] > 0]
only_gold.head()

len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])

df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]
# # Indexing Dataframes
df.head()

df['country'] = df.index
df = df.set_index('Gold')
df.head()

df = df.reset_index()
df.head()

df = pd.read_csv('census.csv')
df.head()

df['SUMLEV'].unique()

df=df[df['SUMLEV'] == 50]
df.head()

columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
df.head()

df = df.set_index(['STNAME', 'CTYNAME'])
df.head()

df.loc['Michigan', 'Washtenaw County']

df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ]
# # Missing values
df = pd.read_csv('log.csv')
df

get_ipython().magic('pinfo df.fillna')

df = df.set_index('time')
df = df.sort_index()
df

df = df.reset_index()
df = df.set_index(['time', 'user'])
df

df = df.fillna(method='ffill')
df.head()

