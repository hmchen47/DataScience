# Basic Data Processing with Pandas

## Introduction

+ Pandas
    + Created in 2008 by Wes McKinney
    + Open source New BSD license
    + 100+ different contributors
+ References
    + Stack Overflow
        + http://stackoverflow.com
        + Massive knowledge forum of python and pandas related content
        + Free to join and participate in
        + Heavily used by pandas developers instead of a mailing list
    + Books
        + Wes McKinney, 'Python for Data Analysis', O'Reilly
        + Matt Harrison, 'Learning the Pandas Library'
    + Planet Python
        + http://planetpython.org/
        + Excellent blog aggregator for python related news
        + Significant number of data science and python tutorials as posted
        + Great blend of applied beginner and higher level python postings


<a href="https://d3c33hcgiwev3.cloudfront.net/ThPPKZeOEeaK1Q4gRyvE8A.processed/full/540p/index.mp4?Expires=1525478400&Signature=NYPazMrxmZWOy9hKKwMPe8J2PYvD8l7oWRsWCpNaCe30bNglWeval5sYOl6KWmHf8~C3PWCsRyAMhTMfPhe0UaHLSax9lUbIRd2rFWOPgL9ryQ0RPM2lgP5cQ9lKOYJrRE9AEfMemDma~wUeENMyrCExe8tb0HxEhf88hAJPhL4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>

## Week 2 Lectures Jupyter Notebook

To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource.

[Web Notebook](https://hub.coursera-notebooks.org/hub/coursera_login?token=YARyOKhmToCEcjioZl6AkA&next=%2Fnotebooks%2FWeek%25202.ipynb)

[Local Notebook](./notebooks/Week02.ipynb)

## The Series Data Structure

![diagram](https://www.kdnuggets.com/wp-content/uploads/pandas-02.png)

+ The Series
    + Name: label of a column
    + Index: the index of a row
    + Value: the element
+ Class `Series`
    + Syntax: `pd.Series(data, index, dtype=None, copy=False)`
    + One-dimensional ndarray with axis labels (including time series)
    + `data`: array-like, dict, or scalar value; Contains data stored in Series
    + `index`: array-like or Index (1d); Values must be hashable and have the same length as `data`. 
    + `dtype`: numpy.dtype or None
    + `copy`: boolean, default False; Copy input data
+ `np.isnan` func:
    + Syntax: `isnan(x, out=None)`
    + Test element-wise for NaN and return result as a boolean array
    + `x`: array_like; input array
    + `out`: ndarray, None, or tuple of ndarray and None; A location into which the result is stored

+ Demo
    ```python
    import pandas as pd
    pd.Series?

    animals = ['Tiger', 'Bear', 'Moose']
    pd.Series(animals)

    numbers = [1, 2, 3]
    pd.Series(numbers)

    animals = ['Tiger', 'Bear', None]
    pd.Series(animals)      # None --> None; dtype: object

    numbers = [1, 2, None]
    pd.Series(numbers)      # None --> NaN; dtype: float64

    import numpy as np
    np.nan == None          # False

    np.nan == np.nan        # False

    np.isnan(np.nan)

    sports = {'Archery': 'Bhutan',
            'Golf': 'Scotland',
            'Sumo': 'Japan',
            'Taekwondo': 'South Korea'}
    s = pd.Series(sports)

    s.index         # Index(['Archery', 'Golf', 'Sumo', 'Taekwondo'], dtype='object')

    s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])

    sports = {'Archery': 'Bhutan',
            'Golf': 'Scotland',
            'Sumo': 'Japan',
            'Taekwondo': 'South Korea'}
    s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey']) # only last three taken
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/deR9JZmEEeaToA55AQb91A.processed/full/540p/index.mp4?Expires=1525478400&Signature=KsI8ZLjCILliqC04ox1NQDk76fV-CCnCsJA3FeP~J55DpQR6dRMcvN6ffqUJ6Prk00ecOnm8TUMudKNsbiR6A2e7pC0XV1wAArn6rNh~rGyoICswJLp2MHSPUKTWuimK2gNzPpXbH06ucv0T~2s9S-QCQwGqyY7QGsAY-EuJO2w_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>

## Querying a Series

+ `df.loc` property:
    + Purely label-location based indexer for selection by label.
    + `df.loc[]` is primarily label based, but may also be used with a boolean array.
    + Allowed inputs are:
        + A single label, e.g. `5` or `'a'`, (note that `5` is interpreted as a *label* of the index, and **never** as an integer position along the index).
        + A list or array of labels, e.g. `['a', 'b', 'c']`.
        + A slice object with labels, e.g. `'a':'f'` (note that contrary to usual python slices, **both** the start and the stop are included!).
        + A boolean array.
        + A `callable` function with one argument (the calling Series, DataFrame or Panel) and that returns valid output for indexing (one of the above)

+ `df.iloc` property:
    + Purely integer-location based indexing for selection by position.
    + `df.iloc[]` is primarily integer position based (from `0` to `length-1` of the axis), but may also be used with a boolean array.
    + Allowed inputs are:
        + An integer, e.g. `5`. 
        + A list or array of integers, e.g. `[4, 3, 0]`.
        + A slice object with ints, e.g. `1:7`.
        + A boolean array.
        + A `callable` function with one argument (the calling Series, DataFrame or Panel) and that returns valid output for indexing (one of the above)

+ Demo
    ```python
    sports = {'Archery': 'Bhutan',
            'Golf': 'Scotland',
            'Sumo': 'Japan',
            'Taekwondo': 'South Korea'}
    s = pd.Series(sports)

    s.iloc[3]       # index location
    s.loc['Golf']   # value location
    s[3]            # index location
    s['Golf']       # value location

    sports = {99: 'Bhutan',
            100: 'Scotland',
            101: 'Japan',
            102: 'South Korea'}
    s = pd.Series(sports)

    s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead

    s = pd.Series([100.00, 120.00, 101.00, 3.00])   # auto index w/ number sequence from 0

    # np.sum and iteration
    total = 0
    for item in s:
        total+=item
    print(total)

    total = np.sum(s)
    print(total)

    #this creates a big series of random numbers
    s = pd.Series(np.random.randint(0,1000,10000))
    s.head()

    len(s)

    # run to times and get the average time to execute the code
    %%timeit -n 100
    summary = 0
    for item in s:
        summary += item
    # 1.07 ms ± 95.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    # get_ipython().run_cell_magic('timeit', '-n 100', 'summary = 0\nfor item in s:\n    summary+=item')

    # generate random series for 10 times
    %%timeit -n 100
    summary = np.sum(s)
    # 144 µs ± 24.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    # get_ipython().run_cell_magic('timeit', '-n 100', 'summary = np.sum(s)')

    s+=2 #adds two to each item in s using broadcasting
    s.head()

    for label, value in s.iteritems():
        s.set_value(label, value+2)
    s.head()

    %%timeit -n 10
    s = pd.Series(np.random.randint(0,1000,10000))
    for label, value in s.iteritems():
        s.loc[label]= value+2

    # get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')

    %%timeit -n 10
    s = pd.Series(np.random.randint(0,1000,10000))
    s+=2

    # get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\ns+=2')

    s = pd.Series([1, 2, 3])
    s.loc['Animal'] = 'Bears'

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
    ```

<a href="https://d3c33hcgiwev3.cloudfront.net/nE8CiJePEea_cQqzDLeQwg.processed/full/540p/index.mp4?Expires=1525564800&Signature=eml1y2DEZSD3DW4FUnrzfUiypGLnp8kIOTr43m7-sobpf4sXW80ltXDAyktfSfoZuuNMaifteEJuEvFBtf52LUwKIoug-CwAYNIy-eJxzSeQzKCpQvFRcj1DRU~q~il6LL4R31z3WtjZSeOZZM6T~q--KijBe-0ksPF7ftnH2Ko_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>

## The DataFrame Data Structure

<img src="./diagrams/dataframe.png" width="450" alt="DataFrame anotomy">

+ `df.copy` method:
    + Syntax: `df.copy(deep=True)`
    + Make a copy of this objects data
    + `deep`: boolean or string; 
        + True: Make a deep copy, including a copy of the data and the indices.
        + False: neither the indices or the data are copied.
+ `df.drop` method:
    + Syntax: `df.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')`
    + Return new object with labels in requested axis removed
    + `labels`: single label or list-like; Index or column labels to drop.
    + `axis`: int or axis name; Whether to drop labels from the index (0/'index') or columns (1/'columns').
    + `index`, `columns`: single label or list-like; Alternative to specifying `axis` (`labels, axis=1` is equivalent to `columns=labels`).
    + `level`: int or level name; For MultiIndex 
    + `inplace`: bool; True: do operation inplace and return None.
    + `errors`: {'ignore', 'raise'}

    <img src="https://cdn-images-1.medium.com/max/1250/1*ZSehcrMtBWN7_qCWq_HiSg.png" width="600" alt="Anatomy of Pandas DataFrame">

+ Demo
    ```python
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

    df.loc['Store 2']           # 'Store 2' row
    type(df.loc['Store 2'])     # pandas.core.series.Series
    df.loc['Store 1']           # 'Store 1' row
    df.loc['Store 1', 'Cost']   # ('Store 1', Cost) element
    df.T.loc['Cost']            # Cost column
    df['Cost']                  # Cost column
    df.loc['Store 1']['Cost']   # ('Store 1', Cost) element
    df.loc[:,['Name', 'Cost']]  # 
    df.drop('Store 1')

    copy_df = df.copy()
    copy_df = copy_df.drop('Store 1')
    copy_df

    help(opy_df.drop)

    del copy_df['Name']
    copy_df

    df['Location'] = None
    ```

+ Quiz
    + For the purchase records from the pet store, how would you get a list of all items which had been purchased (regardless of where they might have been purchased, or by whom)?
        ```python
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

        # Your code here
        ```
    + Answer: 
        ```python
        df['Item Purchased']
        ```
    + For the purchase records from the pet store, how would you update the DataFrame, applying a discount of 20% across all the values in the 'Cost' column?
        ```python
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

        # Your code here
        ```
    + Answer:
        ```python
        df['Cost'] *= 0.8
        ptint(df)
        ```

    <img src="http://pbpython.com/images/pandas-dataframe-shadow.png" width="600" alt="Creating DataFrame">

<a href="https://d3c33hcgiwev3.cloudfront.net/w6PVAZmGEeaagxL7xdFKxA.processed/full/540p/index.mp4?Expires=1525564800&Signature=cI~uPCjTpOVibCfdgKjXSUO2fSV5tMmHRPm578h5Gfms2Dd08CDs8xYtFW~5uDiS9PwP6SUWTp03wT2h3Ks0OeLf4FmmRAcb9OiFU3x-nkBQv2WjJw7iD13EiRJoRQNN04RMpFTmh5xkALYvwUsoTaweFMTBo9zF2WbtKJnQgwQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>

## DataFrame Indexing and Loading

+ `rename` method
    + Syntax: `rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None)`
    + Alter axes labels.
    + `mapper`, `index`, `columns`: dict-like or function; dict-like or functions transformations to apply to that axis' values. Use either `mapper` and `axis` to specify the axis to target with `mapper`, or `index` and `columns`.
    + `axis`: int or str; Axis to target with `mapper`. Can be either the axis name `('index', 'columns')` or number `(0, 1)`. The default is 'index'.
    + `inplace`: boolean
    + `level`: int or level name; In case of a MultiIndex, only rename labels in the specified level.

+ Demo
    ```python

    !cat olympics.csv # shell cmd execution

    # CSV file loading
    df = pd.read_csv('olympics.csv')
    df.head()

    # CSV file loading w/ index and row skipping
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
    ```

+ Quiz
    + Suppose we have a CSV file exercise.csv that looks like this:

        Exercise CSV  
        Week 1 Exercises  
        
        | Activity ID | Activity Type | Activity Duration | Calories | 
        |-------------|---------------|-------------------|----------|
        | 125 | Running | 65 | 450 | 
        | 126 | Biking | 40 | 280 | 
        | 127 | Running | 90 | 850 | 
        | 128 | Walking | 30 | 160 | 

        Which of the following would return a DataFrame with the columns = ['Activity Type', 'Activity Duration', 'Calories'] and index = [125, 126, 127, 128] with the name 'Activity ID'?
        
        a. `pd.read_csv('exercise.csv', skiprows=2, index_col=0)`<br/>
        b. `pd.read_excel('exercise.csv', skiprows=2, index_col=0)`<br/>
        c. `pd.read_excel('exercise.csv', skiprows=2, sep='\t')`<br/>
        d. `pd.read_csv('exercise.csv', skiprows=2, sep=',')`<br/>

    + Answer:  a


<a href="https://d3c33hcgiwev3.cloudfront.net/zFPMm5ePEea2tg7d5YqbXg.processed/full/540p/index.mp4?Expires=1525564800&Signature=BW85AthCkYtHJ9xLHWM4xnWm9yoYWWVy1WEu1J4mVtFugzdS76x49rkoaL0JJP3xhAaTVuACmjbxmbnUQyMsp0B8-7Uf2PqaJ6xNy5PHdK-kJepTL48FKWd8C5N-JHo6lkDp6cGhKgm~pC79NPt~OTu9cBzjHPPmT1PZFZtn2Sk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>

## Querying a DataFrame

+ Boolean Masking
    + conceptually efficient and cornerstone of efficient NumPy
    + an array or dataframe  w/ `True` or `False` on each element
    <a href="url"> <br/>
        <img src="./diagrams/booleanMasking.png" alt="The example illustrated hwo booklean masking works" title= "Boolean Masking Example" height="200">
    </a>

+ `df.where` method:
    + Syntax: `df.where(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False, raise_on_error=None)`
    + Return an object of same shape as self and whose corresponding entries are from self where `cond` is True and otherwise are from `other`.
    + `cond`: boolean NDFrame, array-like, or callable; 
        + Where `cond` is True, keep the original value. Where False, replace with corresponding value from `other`.
        + If `cond` is callable, it is computed on the NDFrame and should return boolean NDFrame or array. The callable must not change input NDFrame (though pandas doesn't check it).
    + `other`: scalar, NDFrame, or callable
        + Entries where `cond` is False are replaced with corresponding value from `other`.
        + If other is callable, it is computed on the NDFrame and should return scalar or NDFrame. The callable must not change input NDFrame (though pandas doesn't check it).
    + `inplace`: boolean; Whether to perform the operation in place on the data
    + `axis`: alignment axis if needed
    + `level`: alignment level if needed
    + `errors`: str, {'raise', 'ignore'}, default 'raise'
        + `raise` : allow exceptions to be raised
        + `ignore` : suppress exceptions. On error return original object
+ `df.dropna` method:
    + Syntax: `df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)`
    + Return object with labels on given axis omitted where alternately any or all of the data are missing
    + `axis`: {0 or 'index', 1 or 'columns'}, or tuple/list thereof; Pass tuple or list to drop on multiple axes
    + `how`: {'any', 'all'}
        + any : if any NA values are present, drop that label
        + all : if all values are NA, drop that label
    + `thresh`: int; int value : require that many non-NA values
    + `subset`: array-like; Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include 
    + `inplace`: boolean
+ `df.count` method:
    + Syntax: `df.count(axis=0, level=None, numeric_only=False)`
    + Return Series with number of non-NA/null observations over requested axis. Works with non-floating point data as well (detects NaN and None)
    + `axis`: {0 or 'index', 1 or 'columns'}
    + `level`: int or level name; If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a DataFrame
    + `numeric_only`: boolean; Include only float, int, boolean data

+ Demo
    ```python
    df['Gold'] > 0      # create boolean mask

    # apply boolean mask to dataframe w/ where function
    only_gold = df.where(df['Gold'] > 0)
    only_gold.head()

    # some countries do not have gold medal
    only_gold['Gold'].count()
    df['Gold'].count()

    # drop rows w/ NaN
    only_gold = only_gold.dropna()
    only_gold.head()

    only_gold = df[df['Gold'] > 0]
    only_gold.head()

    # countries won gold medals in summer or winter
    len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])

    # counties won golf medal only in winter
    df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]
    ```
+ Quiz
    + Write a query to return all of the names of people who bought products worth more than $3.00
        ```python
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

        # Your code here
        ```
    + Answer: `df['Name'][df['Cost']>3]`

<a href="https://d3c33hcgiwev3.cloudfront.net/LHzOC5mHEeaqggpsvkGGZA.processed/full/540p/index.mp4?Expires=1525564800&Signature=BflZUCj82jZiMQn4jQihd5IuGFo8ZCcK6HwJ9CGkISeYIMzYGnAf91Lo44uTmeUPlxfKhRlHV8GNCs2RPu1iO1lw0V7fzk3fTXeybs3qq8QckukBBhaIyoEZt6SZi3OzXK7zARHJOueBkLSYow1m2LY0fBxlROJUKOXl~YtKqdQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>

## Indexing DataFrames

+ Indexing
    + `index`: row level label
    + `set_index` method: 
        + destructive operation
        + copy the column and set its index to preserve the original index
        + index column offset with original dataframe
        + new 1st row w/ empty value
    + `reset_index` method: remove original indices and create a default numerical indices

+ `df.set_index` method:
    + Syntax: `df.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)`
    + Set the DataFrame index (row labels) using one or more existing columns. By default yields a new object.
    + `keys`: column label or list of column labels / arrays
    + `drop`: boolean; Delete columns to be used as the new index
    + `append`: boolean; Whether to append columns to existing index
    + `inplace`: boolean; Modify the DataFrame in place (do not create a new object)
    + `verify_integrity`: boolean; Check the new index for duplicates. Otherwise defer the check until necessary. Setting to False will improve the performance of this method

+ `df.reset_index` method:
    + Syntax: `df.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')`
    + For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the index names, defaulting to 'level_0', 'level_1', etc. if any are None. For a standard index, the index name will be used (if set), otherwise a default 'index' or 'level_0' (if 'index' is already taken) will be used.
    + `level`: int, str, tuple, or list; Only remove the given levels from the index. Removes all levels by default
    + `drop`: boolean; Do not try to insert index into dataframe columns. This resets the index to the default integer index.
    + `inplace`: boolean; Modify the DataFrame in place (do not create a new object)
    + `col_level`: int or str; If the columns have multiple levels, determines which level the labels are inserted into. By default it is inserted into the first level.
    + `col_fill`: object; If the columns have multiple levels, determines how the other levels are named. If None then the index name is repeated.

+ `df.unique` function
    + Syntax: `df.unique(values)`
    + Hash table-based unique. Uniques are returned in order of appearance. This does NOT sort.
    + `values`: 1d array-like

+ Demo
    ```python
    df['country'] = df.index    # preserve index as column 'country'
    df = df.set_index('Gold')   # set 'Gold' column as index and move to front
    df.head()

    df = df.reset_index()
    df.head()

    df = pd.read_csv('census.csv')
    df.head()                       # only display first 5 rows

    # distinct values in 'SUMLEV' column
    df['SUMLEV'].unique()

    df=df[df['SUMLEV'] == 50]   # reserve rows with SUMLEV=50

    # set of columns to keep
    columns_to_keep = ['STNAME', 'CTYNAME', 'BIRTHS2010',
                    'BIRTHS2011', 'BIRTHS2012', 'BIRTHS2013',
                    'BIRTHS2014', 'BIRTHS2015', 'POPESTIMATE2010',
                    'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013',
                    'POPESTIMATE2014', 'POPESTIMATE2015']
    
    df = df[columns_to_keep]

    df = df.set_index(['STNAME', 'CTYNAME'])    # dual indices
    df.loc['Michigan', 'Washtenaw County']      # single set of values
    df.loc[ [('Michigan', 'Washtenaw County'),  # multi-sets of values
            ('Michigan', 'Wayne County')] ]
    ```

+ Quiz
    + Reindex the purchase records DataFrame to be indexed hierarchically, first by store, then by person. Name these indexes 'Location' and 'Name'. Then add a new entry to it with the value of:

        Name: 'Kevyn', Item Purchased: 'Kitty Food', Cost: 3.00 Location: 'Store 2'.
        ```python
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

        # Your answer here  
        ```
    + Answer:
        ```python
        df = df.set_index([df.index, 'Name'])
        df.index.names = ['Location', 'Name']
        df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
        df
        ```

<a href="https://d3c33hcgiwev3.cloudfront.net/60unRpePEeaK1Q4gRyvE8A.processed/full/540p/index.mp4?Expires=1525564800&Signature=D2XAgk~woURtyknKbi4bx-12FPNB~42JHGHlKi54CzwLUrc8dqLdXNiswpwRvoxmkQoK7MsMbTM-o2ASqaNPSgX2Na3yUrv6iLjvvcmM4~lZlbwegcKYGRSm~ZBLHMGw~Tm23r8HPKpIyUquWvQrDKg6FDsYsKQ5LrVacGB4SyQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>

## Missing Values

+ `fillna` method: 
    + Signature: `df.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)`
    + Docstring: Fill NA/NaN values using the specified method
    + Parameters
        + `value`: scalar, dict, Series, or DataFrame; Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of values specifying which value to use for each index (for a Series) or column (for a DataFrame). (values not in the dict/Series/DataFrame will not be filled). This value cannot be a list.
        + `method`: {'backfill', 'bfill', 'pad', 'ffill', None}; Method to use for filling holes in reindexed Series, `pad`/`ffill`: propagate last valid observation forward to next valid, `backfill`/`bfill`: use NEXT valid observation to fill gap
        + `axis`: {0 or 'index', 1 or 'columns'}  
        + `inplace`: boolean; If True, fill in place. Note: this will modify any other views on this object, (e.g. a no-copy slice for a column in a DataFrame).
        + `limit`: int; If method is specified, this is the maximum number of consecutive NaN values to forward/backward fill. In other words, if there is a gap with more than this number of consecutive NaNs, it will only be partially filled. If method is not specified, this is the maximum number of entries along the entire axis where NaNs will be filled. Must be greater than 0 if not None.
        + `downcast`: dict; a dict of item->dtype of what to downcast if possible, or the string 'infer' which will try to downcast to an appropriate equal type (e.g. float64 to int64 if possible)

+ `df.sort_index` method:
    + Syntax: `df.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)`
    + Sort object by labels (along an axis)
    + Parameters:
        + `axis`: index, columns to direct sorting
        + `level`: int or level name or list of ints or list of level names
            if not None, sort on values in specified index level(s) 
        + `ascending`: boolean; Sort ascending vs. descending
        + `inplace`: bool; if True, perform operation in-place
        + `kind`: {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
        + `na_position`: {'first', 'last'}; `first` puts NaNs at the beginning, `last` puts NaNs at the end. Not implemented for MultiIndex.
        + `sort_remaining`: bool; if true and sorting by level and index is multilevel, sort by other levels too (in order) after sorting by specified level

+ Demo
    ```python
    df = pd.read_csv('log.csv')

    help(pd.DataFrame.fillna)

    df = df.set_index('time')
    df = df.sort_index()

    df = df.reset_index()
    df = df.set_index(['time', 'user'])

    df = df.fillna(method='ffill')
    df.head()
    ```


<a href="https://d3c33hcgiwev3.cloudfront.net/99DDF5ePEeaK1Q4gRyvE8A.processed/full/540p/index.mp4?Expires=1525564800&Signature=BvPVy7JSz-QfzQJawmWgz-jCCdOctkjwFMoV5gw0K4l0A4YSsBnrMrdpZN3iN0O11mh~Ai30CIhpT2v42JDuZ1thRA~i73TxBhU4dfVHsm3T6iF21U3qCp2lRXUksOAslNc5Mz2vs~gQiTQ5JQhIcoK7SzXWrvAkJlFydWmcDhE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" style="width:60px;height:60px;border:0;"> 
</a>


## Hacked Data

This course uses a third-party tool, Hacked Data, to enhance your learning experience. The tool will reference basic information like your name, email, and Coursera ID.

[Open Tool](https://nzh13lxjj0.execute-api.us-east-1.amazonaws.com/prod/response/2/new/)


## Opinion:

Nathaniel Poor and Roei Davidson from the Data and Society Research Institute were faced with an ethical dilemma when the data they wanted to use for research but couldn't (because of logistics issues) was released by hackers. The data they were interested in was publicly available, but difficult to get at requiring an expensive manual process. The hacked dataset included this public data as well as private messages which they were not interested in. 

Read the case study [here](http://bdes.datasociety.net/wp-content/uploads/2016/10/Patreon-Case-Study.pdf). 

Do you agree with the author's decision to not use the hacked data, and state any arguments they haven't considered for or against the use of this data?



> Agree

> I agree authors's decision not to use the data set.  Yes, the data is available in public domain now.  However, the way and origin of the data is illegal.  The root of the data is malicious.   
>
> According to the Fruit of the poisonous tree theory, the fruit is poisonous no matter how great  it is.   Similarly, the use of the data might come out fruitful results due to the accuracy and completeness of the data set.  However, The ethic issues, in particular, privacy, makes the data improper.  The results are also improper.  Though someone might said the data used was filtered and scanned before use and published.  To get such results based on the rotten root makes the results arguable.  
> 
> Alternatively,  why not contact the data source again and ask their permission for using the data they provide or crawled by authors themselves.  The authors still can get similar results with the proper way.  The results are more respectful and appreciate by others.

### Peer Feedback : Peer Feedback 

A student from Ethiopia agrees with you.  
> I totally agree with the author's decision not to use the hacked data because it would intrude the privacy of many users even though the data was to be used for a research. Furthermore, the research could be done using other means and medium of data which would have been legal whereas using this data would set an example that might lead to increase hacking of personal data in the name of research. The research might just be very biased rather than generalizable because no one can confirm the validity of hacked data.


My Opinion: convincing

> I agree the opinion and found the author provided a  good point.  The use of the data set will implicitly encourage more hackers doing similar activities.  In this case, the data uses for research.  The result seems not too bad.  However, many other usage might cause big damage on other issues, such as trusty between people and the Internet service provider.  That will eventually tear down the trust and hinder the development  and use of technology. 

This activity is a new one, and we would like your feedback on the value of peer review in peer review in Coursera courses like this one. Please share your thoughts to the following:

A student from India agrees with you.  
> I strongly agree with the author's decision to not use the hacked data. The advent of internet has had far reaching effects on humanity. While it can be argued that a vast majority of them are progressive, there has been considerable wrong doing as well. This has left a great degree of skepticism about the internet in the minds of humans . As a progressive species we must try to avoid and eradicate the evil effects of internet to foster faith in skeptics. With that intention, we must not only avoid the use of data that was gathered illegally but actively discourage the very act of hacking data.

My Opinion: convincing

> The opinion remind me  on of the value in Google, Don't be evil.  This is a great part of Internet.  So many people, even county, such as China, manipulate the publicity and freedom of the Internet.  That will block the progressive of the technology eventually.


A student from United Kingdom agrees with you.
> 1. Researchers have a limited capability to distinguish between public and private information within the hacked data. 
> 2. May see private data when cleaning the data. 
> 3. Perhaps legitimizing criminal activity. 
> 4. Violating users’ expectation of privacy. 
> 5. Using people’s data without consent. 
> 6. We want this data, but we don’t need it.  
> Other data can be ethically collected and used. These arguments against using the data, we feel, are much stronger than the arguments for using the data. Thus, in the end, we did not use the data copied and released by the hackers. Considering other cases and academic guidelines, we felt it would not be appropriate. Altogether, despite our hoping to do some good with the data and despite our hope to only use parts of it that were originally public, we felt the negatives outweighed the positives, especially when we could gather all or most of the same data in a more legal and more accepted manner. Some cases of using data (or not) will be clear, other cases will not be. In the spirit of making lemonade out of lemons, we hope our case highlights some of the difficulties and considerations academics may encounter when contemplating the use of data.

My Opinion: convincing

> I agree the comments.  However, the opinion just states some facts not provide consequence of using the data significantly.   The consequence of using such illegal  data will make the opinion much stronger.


