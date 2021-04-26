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




