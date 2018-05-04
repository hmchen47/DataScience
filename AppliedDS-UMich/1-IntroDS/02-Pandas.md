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

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://d3c33hcgiwev3.cloudfront.net/ThPPKZeOEeaK1Q4gRyvE8A.processed/full/540p/index.mp4?Expires=1525478400&Signature=NYPazMrxmZWOy9hKKwMPe8J2PYvD8l7oWRsWCpNaCe30bNglWeval5sYOl6KWmHf8~C3PWCsRyAMhTMfPhe0UaHLSax9lUbIRd2rFWOPgL9ryQ0RPM2lgP5cQ9lKOYJrRE9AEfMemDma~wUeENMyrCExe8tb0HxEhf88hAJPhL4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){:target="_blank"}

## Week 2 Lectures Jupyter Notebook

To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource.

## The Series Data Structure

![diagram](https://www.kdnuggets.com/wp-content/uploads/pandas-02.png)

+ The Series
    + Name: label of a column
    + Index: the index of a row
    + Value: the element

+ Demo
    ```python
    import pandas as pd
    get_ipython().magic('pinfo pd.Series')

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

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://d3c33hcgiwev3.cloudfront.net/deR9JZmEEeaToA55AQb91A.processed/full/540p/index.mp4?Expires=1525478400&Signature=KsI8ZLjCILliqC04ox1NQDk76fV-CCnCsJA3FeP~J55DpQR6dRMcvN6ffqUJ6Prk00ecOnm8TUMudKNsbiR6A2e7pC0XV1wAArn6rNh~rGyoICswJLp2MHSPUKTWuimK2gNzPpXbH06ucv0T~2s9S-QCQwGqyY7QGsAY-EuJO2w_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){:target="_blank"}

## Querying a Series

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

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}

## The DataFrame Data Structure


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}

## DataFrame Indexing and Loading


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}

## Querying a DataFrame


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}

## Indexing Dataframes


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}

## Missing Values


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}

## Hacked Data


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}
