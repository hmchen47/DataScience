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
    + `left`, `right`: DataFrame
    + `how`: {'left', 'right', 'outer', 'inner'}
        + 'left': use only keys from left frame, similar to a SQL left outer join; preserve key order
        + 'right': use only keys from right frame, similar to a SQL right outer join; preserve key order
        + 'outer': use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically
        + 'inner': use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys
    + `on`: label or list  
        Field names to join on. Must be found in both DataFrames. If on is None and not merging on indexes, then it merges on the intersection of the columns by default.
    + `left_on`, `right_on`: label or list, or array-like  
        Field names to join on in left/right DataFrame. Can be a vector or list of vectors of the length of the DataFrame to use a particular vector as the join key instead of columns
    + `left_index`, `right_index`: boolean, default False  
        Use the index from the left/right DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels
    + `sort`: boolean, default False  
        Use the index from the left DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels

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

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://d3c33hcgiwev3.cloudfront.net/6IgoSIkTEealuRI3K47d-Q.processed/full/540p/index.mp4?Expires=1525651200&Signature=ZIUkterjQQyn2VnsXdBJqvtNKGJUbtRXN75eDDB3MQMeK3Jq1QIgad7iUER~2~9GZs8MGlLY0PaHLxgxEZ4MGsNXmDdZq0m76ceg4Tmj9tfrtVGK2IO7mGfMLamxv6k~mqSuRjyZ859QY~hn-hqU174sSmwmV7D95uKVgjUHWx8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A){: target="_blank"}


## Pandas Idioms


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## Group by


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## Scales


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## Pivot Tables


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## Date Functionality


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


## Discussion Prompt: Goodhart's Law


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){: target="_blank"}


