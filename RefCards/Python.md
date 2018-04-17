# Python Reference Cards

## Typical Programming Environment

### UC Berkeley Data Science

```python
# Run this cell to import libraries needed to run the examples
from datascience import *
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
```

## Numpy

| Function | Description | Link |
|----------|-------------|------|
| `np.char.count(<data>, 'str')` | Count `str` appearance in `data` | [Lec 1.4 Demo: Little Women](../DSFund-BerkeleyX/1-CompThinkWPython/01-Intro.md#lec-14-demo-little-women) |

## Array

### Methods

| Function | Description | Link |
|----------|-------------|------|
| `np.array(lst)` | convert a list to an array | |
| `make_array(elt1, elt2, ...)` | create array with given elements or list | [Lec 4.4 Arrays](../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-44-arrays) |
| `ary.barh('label')` | Depict horizontal bar chart with `label` column | [Lec 3.7 Bar Charts](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-37-bar-charts) |


## Table

### Attributes

| Attribute | Description | Link |
|-----------|-------------|------|
| `t.num_row` / `t.num_column` | row/column size |  [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.labels` | returns a list of column labels of a table | [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |


### Methods 

| Function | Description | Link |
|----------|-------------|------|
| `Table().with_columns(name, values, ...)` | creates a table with an array of values for each column name |  [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `Table()` | creates an empty table|  [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.read_table(filename)` | Read a table from a spreadsheet | [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.with_columns(name, values, ...)` | appends a column name with an array of values to an existing table |  [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.item(int)` | data in column `int` | [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.sum(ary), tbl.max(ary), tbl.min(ary)` | methods to work with data in columns | [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.column(col_name_or_index)` | returns an array with only the values in the specified column | [Lec 4.5 Columns](../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-45-columns) |
| `t.select('label')` | constructs a new table with just the specified columns | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `t.drop('label')` | constructs a new table without the specified columns | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `t.sort('label', desending)` | constructs a new table, with rows sorted by the specified column | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `t.take(row_number)` | keep the numbered rows, index starting at 0 | [Lec 5.9 Where](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-59-where) |
| `t.where(column, are.condition)` | keep all rows for which a column's value satisfies a consition | [Lec 5.9 Where](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-59-where) |
| `t.where(column, value)` | keep all rows containing a certain value in a column | [Lec 5.9 Where](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-59-where) |
| `t.set_format(column, FORMAT)` | convert display format, FORMAT= NumberFormatter, PercentFormatter | [Lec 6.2 Column Arithmetic](../DSFund-BerkeleyX/1-CompThinkWPython/06-Census.md#lec-62-column-arithmetic) |



+ [Visualization with Table](../DSFund-BerkeleyX/1-CompThinkWPython/01-Intro.md#lec-16-demo-visualizations-2)

    ```python
    Table().with_columns([
        'Chapter Length', [len(c) for c in chapters],
        'Number of Periods', np.char.count(chapters, '.'),
    ]).scatter('Number of Periods')
    ```

## Visualizations 
