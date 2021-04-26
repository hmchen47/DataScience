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
  
