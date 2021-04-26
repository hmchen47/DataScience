# 23 great Pandas codes for Data Scientists

Author: G. Srif

Date: 2018-08-22

[Original](https://bit.ly/3nnpDeo)


+ Basic data information
  + read in CSV data set: `pd.read_csv("csv_file")`
  + read in an Excel file: `pd.read_excel("excel_file")`
  + write dataframe directly to CSV (w/o indices): `df.to_csv("data.csv", sep=",", index=False)`
  + basic dataset feature info: `df.info`
  + basic dataset statistics: `df.describe()`
  + print dataframe in table: `print(tabulate(tbl, headers=headers))`
    + tbl = list of list, 
    + installation: `pip3 install tabulate`
  + list of column names: `df.columns`

+ Basic data handling
  + drop missing data: `df.dropna(axis=0, how='any')`
  + replacing missing data: `to_replace=None, value=None)`
  + check for NaNs: `pd.isnull(object)`
    + numeric array: NaN
    + object array: NaN/None
  + drop a feature: `df.drop('feature_name', axis=1)` w/ `axis=0` for rows, `axis=1` for columns
  + convert object type to float (string to numeric): `pd.to_numeric(df['feature_name'], error="coerce")`
  + convert data frame to Numpy array: `df.as_matrix()`
  + get first $n$ rows of a data frame: `df.head(n)`
  + get data by feature name: `df.loc['feature_name']`

+ Operating on data frame
  + apply a function to a data frame: 
  
    ```python
    df['height'].apply(lambda height: 2 * height)

    def multiply(x):
      return 2 * x

    df.['height'].apply(multiply)
    ```

  + renaming a column

    ```python
    # rename 3rd col as 'size
    df.rename(column={df.columns[2]: 'size'}, inplace=true)
    ```

  + get the unique entries of a column: `df['name'].unique()`
  + accessing sub-data frames:

    ```python
    # grab a selection of the columns, 'name' and 'size'
    new_df = df[['name', 'size']]`
    ```

  + summary info about data

    ```python
    # sum of values in a data frame
    df.sum()

    # lowest value of a data frame
    df.min()

    # highest value of a data frame
    df.max()

    # index of the lowest value
    df.idxmin()

    # index of the highest value
    df.idxmax()

    # statistical summary of the data frame
    df.describe()

    # average value of a data frame
    df.mean()

    # median value of a data frame
    df.median()

    # correlation btw columns
    df.corr()

    # median value of a specific column
    df['size'].median()
    ```

  + sorting data: `df_sort_values(ascending=False)`
  + Boolean indexing

    ```python
    # filter data column 'size' w/ value = 5
    df[df['size'] == 5]
    ```

  + selecting values:

    ```python
    # select nth row of a column `size`
    df.loc([n-1], ['size'])
    ```

