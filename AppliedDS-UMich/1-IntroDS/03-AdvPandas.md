## Advanced Python Pandas

## Week 3 Lectures Jupyter Notebook

To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource.

[Notebook Link](https://hub.coursera-notebooks.org/user/qceqpnyfwlofzjpttttssh/notebooks/Week%203.ipynb)

[Local Notebook](./notebooks/Week03.ipynb)

[Local Python File](./notebooks/Week03.py)


## Merging Dataframes

+ Pandas Data Structure
    + Series Object (1 dimensional, a row)
    + DataFrame Object (2 dimensional, a table)
    + Querying
        + `iloc[]`, for querying based on _position_ (index)
        + `loc[]`, for querying rows based on _label_
        + Querying the DataFrame directly
            + Projecting a subset of columns
            + Using a boolean mask to filter data
+ Setting Data in Pandas
    + Create DatFrame w/ Dictionary
        + List of dictionary w/ same keys: `lst_dict`
        + apply to `pd.DataFrame(list_dict)`
    + To add new data: `df['lbl'] = [elt1, elt2, ...]` 
    + To set default data (or overwrite all data): 
        + default: `df['lbl'] = val | bool`
        + individuals: `df['label'] = pd.Series({0: val0, 1: val1, ...})`, ignore items replaced w/ `NaN`
+ Vann Diagram:
    + Full outer union (union)
    + Inner join (intersection)
+ `merge` method
    + Syntax: `pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False)`
    + Merge DataFrame objects by performing a database-style join operation by columns or indexes.
    + Parameters:
        + `left`, `right`: DataFrame
        + `how`: {'left', 'right', 'outer', 'inner'}
            + 'left': use only keys from left frame, similar to a SQL left outer join; preserve key order
            + 'right': use only keys from right frame, similar to a SQL right outer join; preserve key order
            + 'outer': use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically
            + 'inner': use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys
        + `on`: label or list; Field names to join on. Must be found in both DataFrames. If on is None and not merging on indexes, then it merges on the intersection of the columns by default.
        + `left_on`, `right_on`: label or list, or array-like; Field names to join on in left/right DataFrame. Can be a vector or list of vectors of the length of the DataFrame to use a particular vector as the join key instead of columns
        + `left_index`, `right_index`: boolean, default False; Use the index from the left/right DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels
        + `sort`: boolean, default False; Use the index from the left DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels

+ Demo
    ```python
    df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                    {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                    {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                    index=['Store 1', 'Store 1', 'Store 2'])

    df['Date'] = ['December 1', 'January 1', 'mid-May']     # add a new column 'Date' w/ individual values

    df['Delivered'] = True      # add new column 'Delivered' w/ default value

    df['Feedback'] = ['Positive', None, 'Negative']     # 

    adf = df.reset_index()
    adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})    # modify element value(s0)

    staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                            {'Name': 'Sally', 'Role': 'Course liasion'},
                            {'Name': 'James', 'Role': 'Grader'}])
    staff_df = staff_df.set_index('Name')
    student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                            {'Name': 'Mike', 'School': 'Law'},
                            {'Name': 'Sally', 'School': 'Engineering'}])
    student_df = student_df.set_index('Name')

    # join w/ index
    pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
    pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)
    pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
    pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)

    # join w/ column labels
    staff_df = staff_df.reset_index()
    student_df = student_df.reset_index()
    pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')

    # same column labels, e.g., locations for staffs and students -> location_x (left), location_y (right)
    staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                            {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                            {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
    student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                            {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                            {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])
    pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')

    # merge with multiple columns 
    staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                            {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                            {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
    student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                            {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                            {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
    pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])
    ```
+ Quiz

    Here are two DataFrames, products and invoices. The product DataFrame has an identifier and a sticker price. The invoices DataFrame lists the people, product identifiers, and quantity. Assuming that we want to generate totals, how do we join these two DataFrames together so that we have one which lists all of the information we need?

    products DataFrame:

    | Product ID | Price | Product |
    |------------|-------|---------|
    | 4109 | 5.0 | Sushi Roll |
    | 1412 | 0.5 | Egg |
    | 8931 | 1.5 | Bagel |

    invoices DataFrame:

    | Customer | Product ID | Quantity |
    |----------|------------|----------|
    | 0 | Ali | 4109 | 1 |
    | 1 | Eric | 1412 | 12 |
    | 2 | Ande | 8931 | 6 |
    | 3 | Sam | 4109 | 2 |

    ```python
    # products and invoices are already initalized.

    answer = pd.merge(# Your Code Here)

    answer == correct_answer
    ```
    + Answer: 
        ```python
        print(pd.merge(products, invoices, left_index=True, right_on='Product ID'))
        ````

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/6IgoSIkTEealuRI3K47d-Q.processed/full/540p/index.mp4?Expires=1525651200&Signature=ZIUkterjQQyn2VnsXdBJqvtNKGJUbtRXN75eDDB3MQMeK3Jq1QIgad7iUER~2~9GZs8MGlLY0PaHLxgxEZ4MGsNXmDdZq0m76ceg4Tmj9tfrtVGK2IO7mGfMLamxv6k~mqSuRjyZ859QY~hn-hqU174sSmwmV7D95uKVgjUHWx8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){:target="_blank"}


## Pandas Idioms

+ Chain Indexing:
    + Making Code Pandorable
    + e.g., `df.loc[“Washtenaw”][“Total Population”]`
    + Generally bad, Pandas could return a copy of a view depending upon Numpy
+ Code smell
    + If you see a `][` you should think carefully about what you are doing (Tom Augspurger)

+ `where` method
    + Syntax: `df.where(cond, other=nan, inplace=False, axis=None)`
    + Return an object of same shape as self and whose corresponding entries are from self where `cond` is True and otherwise are from `other`.
    + Parameters: 
        + `cond`: boolean NDFrame, array-like, or callable  
            Where `cond` is `True`, keep the original value. Where `False`, replace with corresponding value from `other`. If `cond` is callable, it is computed on the NDFrame and should return boolean NDFrame or array. The callable must not change input NDFrame (though pandas doesn't check it).
        + `other`: scalar, NDFrame, or callable
            Entries where `cond` is False are replaced with corresponding value from `other`.  If other is callable, it is computed on the NDFrame and should return scalar or NDFrame. The callable must not change input NDFrame (though pandas doesn't check it).
        + `inplace`: boolean, default False
            Whether to perform the operation in place on the data
        + `axis`: alignment axis if needed, default None
+ `applymap` method
    + Syntax: `df.applymap(func)`
    + Apply a function to a DataFrame that is intended to operate elementwise, i.e. like doing map(func, series) for each series in the DataFrame
    + All the function to all elements
    + Rarely used
+ `apply` method
    + Syntax: `df.apply(func, axis=0)`
    + Applies function along input axis of DataFrame.
    + Parameters: 
        + `func`: function  
            Function to apply to each column/row
        + `axis`: {0 or 'index', 1 or 'columns'}, default 0
            + `0` or 'index': apply function to each column
            + `1` or 'columns': apply function to each row

+ Demo
    ```python
    df = pd.read_csv('census.csv')

    # method chaining
    (df.where(df['SUMLEV']==50)
        .dropna()
        .set_index(['STNAME','CTYNAME'])
        .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))

    df = df[df['SUMLEV']==50]
    df.set_index(['STNAME','CTYNAME'], inplace=True)
    df.rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'})

    # return a Series and build a new DF w/ 2 new columns
    def min_max(row):
        data = row[['POPESTIMATE2010',
                    'POPESTIMATE2011',
                    'POPESTIMATE2012',
                    'POPESTIMATE2013',
                    'POPESTIMATE2014',
                    'POPESTIMATE2015']]
        return pd.Series({'min': np.min(data), 'max': np.max(data)})
    df.apply(min_max, axis=1)

    # append two new columns to original DF
    def min_max(row):
        data = row[['POPESTIMATE2010',
                    'POPESTIMATE2011',
                    'POPESTIMATE2012',
                    'POPESTIMATE2013',
                    'POPESTIMATE2014',
                    'POPESTIMATE2015']]
        row['max'] = np.max(data)
        row['min'] = np.min(data)
        return row
    df.apply(min_max, axis=1)

    # generate a DF w/ max value
    rows = ['POPESTIMATE2010',
            'POPESTIMATE2011',
            'POPESTIMATE2012',
            'POPESTIMATE2013',
            'POPESTIMATE2014',
            'POPESTIMATE2015']
    df.apply(lambda x: np.max(x[rows]), axis=1)
    ```
+ Quiz  
    Suppose we are working on a DataFrame that holds information on our equipment for an upcoming backpacking trip.

    Can you use method chaining to modify the DataFrame df in one statement to drop any entries where 'Quantity' is 0 and rename the column 'Weight' to 'Weight (oz.)'?
    ```python
    print(df.head())

    # Your code here
    ```
    + Answer
        ```python
        print(df.drop(df[df['Quantity'] == 0].index).rename(columns={'Weight': 'Weight (oz.)'}))
        ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/8V0N2YkTEeaXYgo_fJsBPw.processed/full/540p/index.mp4?Expires=1525737600&Signature=JM~4lYOt7r4cwxt9bb89koR86qLqVkYU1tlJw8jDTvITLBgrML5XlnkOaRW2dNOc1rT5inglJ5xjWFMlozrNGUApkTyq-i32GQcfqmxLnmUDNoMtHzN94IjDUr5Aopxm0j~kGhuefLyQ3TaFe3IYwQZhajLGuDNbRbx-Yy9qIqk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){:target="_blank"}


## Group by

+ Split and Combine pattern: 
    + splitting data into groups
    + processing the data
    + combining the results
+ `dropna` method
    + Syntax: `df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)`
    + Return object with labels on given axis omitted where alternately any or all of the data are missing
    + Parameters: 
        + `axis`: {0 or 'index', 1 or 'columns'}, or tuple/list thereof; Pass tuple or list to drop on multiple axes
        + `how`: {'any', 'all'}  
            + `any` : if any NA values are present, drop that label
            + `all` : if all values are NA, drop that label
        + `thresh`: int: require that many non-NA values
        + `subset`: array-like; Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include
        + `inplace`: boolean; `True`: do operation inplace and return None.
+ `groupby` method
    + Syntax: `df.groupby(by=None, axis=0, level=None, as_index=True, sort=True)`
    + Group series using mapper (dict or key function, apply given function to group, return result as series) or by a series of columns.
    + Parameters: 
        + `by`: mapping, function, str, or iterable; Used to determine the groups for the `groupby`.
            + `function`: called on each value of the object's index.
            + `dict` or `Series`: the Series or dict VALUES will be used to determine the groups (the Series' values are first aligned; see `.align()` method). 
            + `ndarray`: the values are used as-is determine the groups.
            + `str` or `list of strs`: group by the columns in `self`
        + `axis`: int, default 0
        + `level`: int, level name, or sequence of such; If the axis is a MultiIndex (hierarchical), group by a particular level or levels
        + `as_index`: boolean; For aggregated output, return object with group labels as the index. Only relevant for DataFrame input. as_index=False is effectively "SQL-style" grouped output
        + `sort`: boolean; Sort group keys. Get better performance by turning this off. Note this does not influence the order of observations within each group.  groupby preserves the order of rows within each group.

+ `agg` method
    + Syntax: `df.agg(func, axis=0)`
    + Aggregate using callable, string, dict, or list of string/callables
    + Parameters: 
        + `func`: callable, string, dictionary, or list of string/callables; Function to use for aggregating the data. If a function, must either work when passed a DataFrame or when passed to DataFrame.apply. For a DataFrame, can pass a dict, if the keys are DataFrame column names.  
        Accepted Combinations are:
            + string function name
            + function
            + list of functions
            + dict of column names -> functions (or list of functions)
+ Demo
    ```python
    df = pd.read_csv('census.csv')
    df = df[df['SUMLEV']==50]

    # calculate average w/ loop
    %%timeit -n 10
    for state in df['STNAME'].unique():
        avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])
        print('Counties in state ' + state + ' have an average population of ' + str(avg))

    %%timeit -n 10
    for group, frame in df.groupby('STNAME'):
        avg = np.average(frame['CENSUS2010POP'])
        print('Counties in state ' + group + ' have an average population of ' + str(avg))

    # data segmented by the given function
    df = df.set_index('STNAME')

    def fun(item):
        if item[0]<'M':
            return 0
        if item[0]<'Q':
            return 1
        return 2

    for group, frame in df.groupby(fun):
        print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')
    # There are 1177 records in group 0 for processing.
    # There are 1134 records in group 1 for processing.
    # There are 831 records in group 2 for processing.

    df = pd.read_csv('census.csv')
    df = df[df['SUMLEV']==50]

    # split & combine method: group data w/ States and then aggregate
    df.groupby('STNAME').agg({'CENSUS2010POP': np.average})
    #	        CENSUS2010POP
    # STNAME
    # Alabama   71339.343284

    print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']))
    # <class 'pandas.core.groupby.DataFrameGroupBy'>
    print(type(df.groupby(level=0)['POPESTIMATE2010']))
    # <class 'pandas.core.groupby.SeriesGroupBy'>

    # 2 column DF
    (df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
        .agg({'avg': np.average, 'sum': np.sum}))
    #	        avg             sum
    # STNAME
    # Alabama	71339.343284	4779736

    # 4 col result: hierarchical level result
    (df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
        .agg({'avg': np.average, 'sum': np.sum}))
    #           avg                                 sum
    #           POPESTIMATE2010 POPESTIMATE2011     POPESTIMATE2010 POPESTIMATE2011
    # STNAME
    # Alabama   71420.313433    71658.328358        4785161         4801108

    # odd behavior: once relabeling, POPESTIMATE2010 = avg, POPESTIMATE2011 = sum
    (df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
        .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))
    #               POPESTIMATE2010     POPESTIMATE2011
    # STNAME
    # Alabama       71420.313433        4801108
    ```
+ Quiz:  
    Looking at our backpacking equipment DataFrame, suppose we are interested in finding our total weight for each category. Use `groupby` to group the dataframe, and apply a function to calculate the total weight (Weight x Quantity) by category.
    ```python
    print(df)

    # Your code here
    ```

        | Item                  | Category | Quantity | Weight (oz.) |
        |-----------------------|----------|---------_|--------------|
        | Pack                  |     Pack |        1 |         33.0 |
        | Tent                  |  Shelter |        1 |         80.0 |
        | Sleeping Pad          |    Sleep |        1 |         27.0 |
        | Sleeping Bag          |    Sleep |        1 |         20.0 |
        | Toothbrush/Toothpaste |   Health |        1 |          2.0 |
        | Sunscreen             |   Health |        1 |          5.0 |
        | Medical Kit           |   Health |        1 |          3.7 |
        | Spoon                 |  Kitchen |        1 |          0.7 |
        | Stove                 |  Kitchen |        1 |         20.0 |
        | Water Filter          |  Kitchen |        1 |          1.8 |
        | Water Bottles         |  Kitchen |        2 |         35.0 |
        | Pack Liner            |  Utility |        1 |          1.0 |
        | Stuff Sack            |  Utility |        1 |          1.0 |
        | Trekking Poles        |  Utility |        1 |         16.0 |
        | Rain Poncho           | Clothing |        1 |          6.0 |
        | Shoes                 | Clothing |        1 |         12.0 |
        | Hat                   | Clothing |        1 |          2.5 |

    + Answer:
        ```python
        print(df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity'))

        # Or alternatively without using a lambda:
        # def totalweight(df, w, q):
        #        return sum(df[w] * df[q])
        #        
        # print(df.groupby('Category').apply(totalweight, 'Weight (oz.)', 'Quantity'))
        ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/JOzRuokUEea8gwpyjKjbvQ.processed/full/540p/index.mp4?Expires=1525737600&Signature=f6NoHnZx~bIg5Jr4sEB6DV4FOj8W7RtCjsgPNTGjb8hpKcmSG2zW89i15eY8Taf~RSD0uzJBe4P6KHE8k2FJ~h4RDs8GgHT-KbH7ec37qUDiXBjaiM9W0AH5f-6fhrHgxfGsT-o3iZ1vsf0PPV8PafKP0puLnMO31IZ0MQYSK-s_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){:target="_blank"}


## Scales

+ (a,b) (c,d): Scales
    + Ratio scale:
        + units are equally spaced
        + mathematical operations of `+-/*` are all valid
        + e.g. height and weight
    + Interval scale: 
        + units are equally spaced, but there is no true zero
        + e.g., temperature in C or F, 0 degree is meaningful; direction of campus w/ 0 degree
    + Ordinal scale:
        + the order of the units is important, but not evenly spaced.
        + e.g., Letter grades such as A+
        + common in ML, but difficult to work with
    + Nominal scale: categorical data
        + categories of data, but the categories have no order with respect to one another
        + e.g. Teams of a sport
        + Commonly use a column w/ `True` or `False` to each element whether a category applied

+ Dummy variable: 
    + a variable with Boolean value
    + `getdummy` method: convert a single column into multiple columns w/ 0 & 1 indicating the presence of a dummy variable

+ Convert ratio scales into a category
    + Loss info of the value
    + Useful in cases
        + visualizing the frequencies of categories, e.g., historgram
        + ML classification approaches

+ `astype` method:
    + Syntax: `df.astype(dtype)`
    + Cast a pandas object to a specified dtype `dtype`
    + Parameters:
        + `dtype`: data type, or dict of column name -> data type
            + Use a `numpy.dtype` or Python `type` to cast entire pandas object to the same type. 
            + Alternatively, use `{col: dtype, ...}`, where `col` is a column label and `dtype` is a `numpy.dtype` or Python `type` to cast one or more of the DataFrame's columns to column-specific types.

+ `cut` method:
    + Syntax: `pd.cut(x, bins, right=True, labels=None)`
    + Return indices of half-open bins to which each value of `x` belongs.
    + Parameters: 
        + `x`: array-like; Input array to be binned. It has to be 1-dimensional.
        + `bins`: int, sequence of scalars, or IntervalIndex; If `bins` is an int, it defines the number of equal-width bins in the range of `x`. However, in this case, the range of `x` is extended by .1% on each side to include the min or max values of `x`. If `bins` is a sequence it defines the bin edges allowing for non-uniform bin width. No extension of the range of `x` is done in this case.
        + `right`: bool; Indicates whether the bins include the rightmost edge or not. If `right == True` (the default), then the bins [1,2,3,4] indicate (1,2], (2,3], (3,4].
        + `labels`: array or boolean; Used as labels for the resulting bins. Must be of the same length as the resulting bins. If False, return only integer indicators of the bins.

+ Demo
    ```python
    df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                    index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
    df.rename(columns={0: 'Grades'}, inplace=True)

    # claim 'Grade' as 'category' type
    df['Grades'].astype('category').head()

    # claim the categories are ordered
    grades = df['Grades'].astype('category',
                                categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                                ordered=True)

    grades > 'C'    # boolean comparison for df --> boolean masking

    df = pd.read_csv('census.csv')
    df = df[df['SUMLEV']==50]
    df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})
    pd.cut(df['avg'],10)
    ```
+ Quiz  
    + Try casting this series to categorical with the ordering Low < Medium < High.
        ```python
        s = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])

        # Your code here
        ```
    + Answer
        ```python
        s.astype('category', categories=['Low', 'Medium', 'High'], ordered=True)
        ```
    + Suppose we have a series that holds height data for jacket wearers. Use `pd.cut` to bin this data into 3 bins.
        ```python
        s = pd.Series([168, 180, 174, 190, 170, 185, 179, 181, 175, 169, 182, 177, 180, 171])

        # Your code here
        ```
    + Answer: 
        ```python
        pd.cut(s, 3)

        # You can also add labels for the sizes [Small < Medium < Large].
        pd.cut(s, 3, labels=['Small', 'Medium', 'Large'])
        ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/MtOhLIkUEeaXYgo_fJsBPw.processed/full/540p/index.mp4?Expires=1525737600&Signature=GjcP6zHIVck8-gI0O1FHc2AtKkGvMBRXlFOOfXN4cSZp5fNqJNHbC3UJOSBymobTR3kYFsej2smLT~yaAHFSv71aMXJunjiRqBQtaHLg1o8~szGPiauiLqV0E5-L9Ys3nGqUuEt5J5dGBXigAeUEUni2xL6GFt~OJDHtx1LpYsg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){:target="_blank"}


## Pivot Tables

+ Pivot Tables
    + Summarizing data for particular purpose
    + Heavily using aggregation function
    + a data frame, row with one variable while column w/ another
    + tend to having marginal values, sum of each column and row  -> relationship btw two variables

    ![Diagram](http://pbpython.com/images/pivot-table-datasheet.png)

+ `pivot_table` method
    + Syntax: `df.pivot_table(values=None, index=None, columns=None, aggfunc='mean')`
    + Create a spreadsheet-style pivot table as a DataFrame. The levels in the pivot table will be stored in MultiIndex objects (hierarchical indexes) on the index and columns of the result DataFrame
    + Parameters: 
        + `values`: column to aggregate, optional
        + `index`: column, Grouper, array, or list of the previous; If an array is passed, it must be the same length as the data. The list can contain any of the other types (except list). Keys to group by on the pivot table index.  If an array is passed, it is being used as the same manner as column values.
        + `columns`: column, Grouper, array, or list of the previous; If an array is passed, it must be the same length as the data. The list can contain any of the other types (except list). Keys to group by on the pivot table column.  If an array is passed, it is being used as the same manner as column values.
        + `aggfunc`: function or list of functions, default `numpy.mean`; If list of functions passed, the resulting pivot table will have hierarchical columns whose top level are the function names (inferred from the function objects themselves)

+ Demo
    ```python
    #http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
    df = pd.read_csv('cars.csv')

    df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)

    df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True)
    ```

+ Quiz  
    Suppose we have a DataFrame with price and ratings for different bikes, broken down by manufacturer and type of bicycle.

    Create a pivot table that shows the mean price and mean rating for every 'Manufacturer' / 'Bike Type' combination.
    
    |   | Bike Type | Manufacturer | Price | Rating |
    |---|-----------|--------------|-------|--------|
    | 0 |  Mountain |            A |   400 |      8 |
    | 1 |  Mountain |            A |   600 |      9 |
    | 2 |      Road |            A |   400 |      4 |
    | 3 |      Road |            A |   450 |      4 |
    | 4 |  Mountain |            B |   300 |      6 |
    | 5 |  Mountain |            B |   250 |      5 |
    | 6 |      Road |            B |   400 |      4 |
    | 7 |      Road |            B |   500 |      6 |
    | 8 |  Mountain |            C |   400 |      5 |
    | 9 |  Mountain |            C |   500 |      6 |
    | 10|      Road |            C |   800 |      9 |
    | 11|      Road |            C |   950 |     10 |

    + Answer:
        ```python
        print(pd.pivot_table(Bikes, index=['Manufacturer','Bike Type']))
        ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/OgnKXYkUEeaXYgo_fJsBPw.processed/full/540p/index.mp4?Expires=1525737600&Signature=D98E0proM6pxQIWf150RxGU6vsMSagI0gSu00~QipMNwBVK1qp-uDmAGXipcwgWEgsoCD5k0ADTrCPUR1z1OFtLvtC6eD-22lrHF8u4tVkQh~K0VKMhhnJCzJxm0WMl4BC5JCHmOJMfq1dTJXx0m9ELlR2kllJhMn-i7PW2vVV0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){:target="_blank"}


## Date Functionality

+ Pandas Time Related Classes
    + Timestamp
    + DatetimeIndex
    + Period
    + PeriodIndex

+ `Timestamp` Class
    + Syntax: `pd.Timestamp(ts_input, freq, tz, unit=None)`
    + The pandas equivalent of python's Datetime and is interchangable with it in most cases.
    + Parameters:
        + `ts_input`: datetime-like, str, int, float; Value to be converted to Timestamp
        + `freq`: str, DateOffset; Offset which Timestamp will have
        + `tz`: string, `pytz.timezone`, `dateutil.tz.tzfile` or None; Time zone for time which Timestamp will have.
        + `unit`: string; numpy unit used for conversion, if ts_input is int or float

+ `Period` Class
    + Syntax: `pd.Period(value=None, freq=None, year=None, month=1, quarter=None, day=1, hour=0, minute=0, second=0)`
    + Represents a period of time
    + Parameters:
        + `value`: Period or compat.string_types; The time period represented (e.g., '4Q2005')
        + `freq`: str; One of pandas period strings or corresponding objects
        + `year`: int, default None
        + `month`: int, default 1
        + `quarter`: int, default None
        + `day`: int, default 1
        + `hour`: int, default 0
        + `minute`: int, default 0
        + `second`: int, default 

+ `to_datetime` function
    + Syntax: `pd.to_datetime(arg, utc=None, format=None)`
    + Convert argument to datetime.
    + Parameters:
        + `arg`: integer, float, string, datetime, list, tuple, 1-d array, Series
        + `utc`: boolean; Return UTC DatetimeIndex if True
        + `format`: string; strftime to parse time, eg "%d/%m/%Y", note that "%f" will parse all the way up to nanoseconds.

+ `data_range` function:
    + Syntax: `pd.date_range(start=None, end=None, periods=None)`
    + Return a fixed frequency DatetimeIndex, with day (calendar) as the default frequency
    + Parameters:
        + `start`: string or datetime-like; Left bound for generating dates
        + `end`: string or datetime-like; Right bound for generating dates
        + `periods`: integer; Number of periods to generate

+ `diff` method:
    + Syntax: `df.diff(periods=1, axis=0)`
    + 1st discrete difference of object
    + Parameters:
        + `periods`: int;  Periods to shift for forming difference
        + `axis`: {0 or 'index', 1 or 'columns'}

+ `resample` method
    + Syntax: `df.resample(rule, how=None, axis=0)`
    + Convenience method for frequency conversion and resampling of time series.
    + Parameters:
        + `rule`: string;  the offset string or object representing target conversion; [Offset Aliases](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)
        + `axis`: int, optional, default 0

        | Alias | Description |
        |--------|------------|
        | `B`     | business day frequency |
        | `C`     | custom business day frequency (experimental) |
        | `D`     | calendar day frequency |
        | `W`     | weekly frequency |
        | `M`     | month end frequency |
        | `SM`    | semi-month end frequency (15th and end of month) |
        | `BM`    | business month end frequency |
        | `CBM`   | custom business month end frequency |
        | `MS`    | month start frequency |
        | `SMS`   | semi-month start frequency (1st and 15th) |
        | `BMS`   | business month start frequency |
        | `CBMS`  | custom business month start frequency |
        | `Q`     | quarter end frequency |
        | `BQ`    | business quarter endfrequency |
        | `QS`    | quarter start frequency |
        | `BQS`   | business quarter start frequency |
        | `A`     | year end frequency |
        | `BA`    | business year end frequency |
        | `AS`    | year start frequency |
        | `BAS`   | business year start frequency |
        | `BH`    | business hour frequency |
        | `H`     | hourly frequency |
        | `T`     | minutely frequency |
        | `S`     | secondly frequency |
        | `L`     | milliseonds |
        | `U`     | microseconds |
        | `N`     | nanoseconds |

+ `asfreq` method:
    + Syntax: `df.asfreq(freq, method=None, how=None, normalize=False, fill_value=None)`
    + Convert TimeSeries to specified frequency.
    + Parameters:
        + `freq`: DateOffset object, or string
        + `method`: {'backfill'/'bfill', 'pad'/'ffill'}; Method to use for filling holes in reindexed Series (note this does not fill NaNs that already were present):
            + `'pad'`/`'ffill'`: propagate last valid observation forward to next valid
            + `'backfill'`/`'bfill'`: use NEXT valid observation to fill
        + `how`: {'start', 'end'}, default end; For PeriodIndex only, see PeriodIndex.asfreq
        + `normalize`: bool; Whether to reset output index to midnight
        + `fill_value`: scalar; Value to use for missing values, applied during upsampling (note this does not fill NaNs that already were present).

+ Demo
    ```python
    # Timestamp
    pd.Timestamp('9/1/2016 10:05AM')

    # Period
    pd.Period('1/2016')

    pd.Period('3/5/2016')

    # DatetimeIndex
    t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
    type(t1.index)

    # PeriodIndex
    t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
    type(t2.index)

    # Converting to Datetime
    d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
    ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))

    ts3.index = pd.to_datetime(ts3.index)
    pd.to_datetime('4.7.12', dayfirst=True)

    # Timedeltas
    pd.Timestamp('9/3/2016') - pd.Timestamp('9/1/2016')
    pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')

    # Working with Dates in a Dataframe
    dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')

    df = pd.DataFrame({'Count 1': 100 + np.random.randint(-5, 10, 9).cumsum(),
                    'Count 2': 120 + np.random.randint(-5, 10, 9)}, index=dates)

    df.index.weekday_name
    df.diff()
    df.resample('M').mean()
    df['2017']
    df['2016-12']
    df['2016-12':]
    df.asfreq('W', method='ffill')

    # Plot
    import matplotlib.pyplot as plt
    %matplotlib inline

    df.plot()
    ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/QnzfqokUEeaoHhIiFgcrVw.processed/full/540p/index.mp4?Expires=1525737600&Signature=XK6i1~u5ajmdCtXpOn5ccMP84cjPeYcI1sGDllbp5CJE-9fmkfD0SMUFJkupawrJ2h~fUDVFZ~pu7qiQhu~auk10TuizzaNPyiDJK8SZB1ik9kvK5~4QHClEP8mLll8e20kjQz-2aP3xjB5Tu3WBe8yodAiDzkk-R6N5YT25q1s_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


## Discussion Prompt: Goodhart's Law

Listen to the Data Skeptic podcast on [Goodhart's law](http://dataskeptic.com/epnotes/goodharts-law.php)

What is the implication for you as a data scientist?


