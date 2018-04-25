# Python Reference Cards for UC Berkeley DataScience package

## Typical Programming Environment

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
| `make_array(x1, x2, ...)` | create an array with the given values $xi$, where i = 1, 2, ...  | [Lec 4.4 Arrays](../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-44-arrays); [Lec 8.2 Binning](../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-82-binning) |



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
| `t.with_row(lst)` | appends a row w/ given columns to an existing table |  [Lec 5.7 Lists](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-57-lists) |
| `t.with_rows(lst)` | appends multiple rows (rows of columns) w/ given columns to an existing table |  [Lec 5.7 Lists](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-57-lists) |
| `t.item(int)` | data in column `int` | [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.sum(ary), tbl.max(ary), tbl.min(ary)` | methods to work with data in columns | [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `t.column(col_name_or_index)` | returns an array with only the values in the specified column | [Lec 4.5 Columns](../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-45-columns) |
| `t.select('label')` | constructs a new table with just the specified columns | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `t.drop('label')` | constructs a new table without the specified columns | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `t.sort('label', desending)` | constructs a new table, with rows sorted by the specified column | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `t.take(row_number)` | keep the numbered rows, index starting at 0 | [Lec 5.8 Take](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-58-yake) |
| `t.where(column, are.condition)` | keep all rows for which a column's value satisfies a consition | [Lec 5.9 Where](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-59-where) |
| `t.where(column, value)` | keep all rows containing a certain value in a column | [Lec 5.9 Where](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-59-where) |
| `t.set_format(column, FORMAT)` | convert display format, FORMAT= NumberFormatter, PercentFormatter | [Lec 6.2 Column Arithmetic](../DSFund-BerkeleyX/1-CompThinkWPython/06-Census.md#lec-62-column-arithmetic) |
| `t.relabel(col, label)` | rename the label of selected column | [Lec 7.3 Scatter Plots](../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-73-scatter-plots) |
| `t.group(label)` | counting with given label |[Lec 7.7 Distributions](../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-77-distributions) |
| `t.select(func, col)` | returns an array where a function is applied to each item in a column | [Lec 9.5 Apply](../DSFund-BerkeleyX/1-CompThinkWPython/09-Functions.md#lec-95-apply) |
| `t.group(label[, func])` | aggregates all rows with the same value a column into a single row in the resulting table with given function; default=count | [Lec 10.1 One Attribute Group](../DSFund-BerkeleyX/1-CompThinkWPython/10-Groups.md#lec-101-one-attribute-group) |



+ [Visualization with Table](../DSFund-BerkeleyX/1-CompThinkWPython/01-Intro.md#lec-16-demo-visualizations-2)

    ```python
    Table().with_columns([
        'Chapter Length', [len(c) for c in chapters],
        'Number of Periods', np.char.count(chapters, '.'),
    ]).scatter('Number of Periods')
    ```

## Visualizations

### Attributes

| Attribute | Description | Link |
|-----------|-------------|------|




### Methods

| Function | Description | Link |
|----------|-------------|------|
| `t.plot(x[, y])` | line graph, `x` & `y` are column indices in table `t`, no `y` -> plot all columns in `t` except `x` | [Lec 7.1 Line Graphs](../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-71-line-graphs) |
| `t.scatter(x[, y])` | scatter plot | [Lec 7.3 Scatter Plots](../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-73-scatter-plots) |
| `t.bar('label', col)` | Depict bar chart with `label` column; numerical vs categorical or distribution | [Lec 3.7 Bar Charts](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-37-bar-charts) | 
| `t.barh('label', col)` | Depict horizontal bar chart with `label` column; numerical vs categorical or distribution | [Lec 3.7 Bar Charts](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-37-bar-charts); [Lec 7.7 Distributions](../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-77-Distributions) |
| `t.bin(label, bins=<ary>)` | create bins for further use; the last bin is the point, therefore any item located on the point moves to the __last 2nd bin__ | [Lec 8.2 Binning](../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-82-binning) |
| `t.hist(x, bin=<ary>)` | Histogram with given bins | [Lec 8.4 Drawing Histograms](../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-84-drawing-histograms) |


### Common Graph Arguments

| Argument | Description | Link |
|----------|-------------|------|
| `overlay` | True = one graph; False = separated graphs | [Lec 7.1 Line Graphs](../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-71-line-graphs) |
| `unit` | string as the unit of x-axis | [Lec 8.4 Drawing Histograms](../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-84-drawing-histograms) |
| `normed` | True/False, normalized with area principle, deprecated | [Lec 8.5 Density](../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-85-desity) |
| `density` | True/False, normalized with area principle, deprecated | [Lec 8.5 Density](../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-85-desity) |





