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
| `np.char.count(<data>, 'str')` | Count `str` appearance in `data` | [Lec 1.4 Demo: Little Women][002] |
| `np.random.choice(ary, sample_size)` | randomly select an element from `ary` with sample size `size` | [Random Selection][028] |
| `np.count_nonzero(ary)` | Count nonzero values in array `ary` | [Random Selection][028] |


## Array

### Methods

| Function | Description | Link |
|----------|-------------|------|
| `np.array(lst)` | convert a list to an array | |
| `make_array(x1, x2, ...)` | create an array with the given values $xi$, where i = 1, 2, ...  | [Lec 4.4 Arrays][012]; [Lec 8.2 Binning][010] |



## Table

### Attributes

| Attribute | Description | Link |
|-----------|-------------|------|
| `t.num_row` / `t.num_column` | row/column size |  [Lec 5.1 Creating Tables][024] |
| `t.labels` | returns a list of column labels of a table | [Lec 5.1 Creating Tables][024] |


### Methods ([Ref][027])

| Function | Description | Link |
|----------|-------------|------|
| `Table().with_columns(name, values, ...)` | creates a table with an array of values for each column name |  [Lec 5.1 Creating Tables][024] |
| `Table()` | creates an empty table|  [Lec 5.1 Creating Tables][024] |
| `t.read_table(filename)` | Read a table from a spreadsheet | [Lec 5.1 Creating Tables][024] |
| `t.with_columns(name, values, ...)` | appends a column name with an array of values to an existing table |  [Lec 5.1 Creating Tables][024] |
| `t.with_row(lst)` | appends a row w/ given columns to an existing table |  [Lec 5.7 Lists][025]  |
| `t.with_rows(lst)` | appends multiple rows (rows of columns) w/ given columns to an existing table |  [Lec 5.7 Lists][025] |
| `t.item(int)` | data in column `int` | [Lec 5.1 Creating Tables][024] |
| `t.sum(ary), tbl.max(ary), tbl.min(ary)` | methods to work with data in columns | [Lec 5.1 Creating Tables][024] |
| `t.column(col_name_or_index)` | returns an array with only the values in the specified column | [Lec 4.5 Columns][023] |
| `t.select(col[, ...])` | constructs a new table with just the specified columns | [Lec 3.5 Select][022]; [Table Methods][026] |
| `t.drop(col[, ...])` | constructs a new table without the specified columns | [Lec 3.5 Select][022]; [Table Methods][026] |
| `t.sort(col, descending=False, distinct=False)` | constructs a new table, with rows sorted by the specified col | [Lec 3.5 Select][022]; [Table Methods][026] |
| `t.take([row, ...]r)` | keep the numbered rows, index starting at 0 | [Lec 5.8 Take][021]; [Table Methods][026] |
| `t.where(col, are.condition(...))` | keep all rows for which a column's value satisfies a condition | [Lec 5.9 Where][020]; [Table Methods][026] |
| `t.where(col, value)` | keep all rows containing a certain value in a column | [Lec 5.9 Where][020] |
| `t.set_format(col, FORMAT)` | convert display format, FORMAT= NumberFormatter, PercentFormatter | [Lec 6.2 Column Arithmetic][019] |
| `t.relabel(col, label)` | rename the label of selected column | [Lec 7.3 Scatter Plots][008] |
| `t.group(col)` | counting with given label |[Lec 7.7 Distributions][011] |
| `t.apply(func, col, ...)` | returns an array where a function is applied to each item in a col | [Lec 9.5 Apply][018]; [Table Methods][026] |
| `t.group(col[, func])` | aggregates all rows with the same value a column into a single row in the resulting table with given function; default=count | [Lec 10.1 One Attribute Group][017] |
| `t.group([col, ...][, func])` | aggregate all rows that share the combination of values in multiple columns; default=count | [Lec 10.2 Cross Classification][015]
| `t.pivot(col, roe, [, values=vals, collect=func])` | returns a pivot table where each unique value in col1 has its own column and each unique value in col2 has its own row. The cells of the grid contain row counts (two arguments) or the values from a third col, aggregated by the collect function | [Lec 10.4 Pivot Tables][015]; [Table Methods][026] |
| `tblA.join(colA, tblB, colB)` | returns a table with the columns of `tblA` and `tblB`, containing rows for all values of a `colA` and colB that appear in both tables | [Lec 11.1 Joins][014]; [Table Methods][026] |
| `t.exclude([row, ...])` | return a table that excludes listed rows from given table | [Table Methods][026] |





+ [Visualization with Table][013]

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
| `t.plot(x[, y])` | line graph, `x` & `y` are column indices in table `t`, no `y` -> plot all columns in `t` except `x` | [Lec 7.1 Line Graphs][001] |
| `t.scatter(x[, y])` | scatter plot | [Lec 7.3 Scatter Plots][008] |
| `t.bar(colX, colY)` | Depict bar chart with `colX` column; numerical vs categorical or distribution | [Lec 3.7 Bar Charts][009] | 
| `t.barh(colX, colY)` | Depict horizontal bar chart with `colX` column; numerical vs categorical or distribution | [Lec 3.7 Bar Charts][009]; [Lec 7.7 Distributions][011] |
| `t.bin(col, bins=<ary>)` | create bins for further use; the last bin is the point, therefore any item located on the point moves to the __last 2nd bin__ | [Lec 8.2 Binning][010] |
| `t.hist(x, bin=<ary>)` | Histogram with given bins | [Lec 8.4 Drawing Histograms][006]|
| `{Marker\|Circle}..map_table(tbl, ...)` | A table containing columns of latitude and longitude values used to generate a map of markers | [Lec 11.4 Maps][007] |



### Common Graph Arguments

| Argument | Description | Link |
|----------|-------------|------|
| `overlay` | True = one graph; False = separated graphs | [Lec 7.1 Line Graphs][001] |
| `unit` | string as the unit of x-axis | [Lec 8.4 Drawing Histograms][006] |
| `normed` | True/False, normalized with area principle, deprecated | [Lec 8.5 Density][005] |
| `density` | True/False, normalized with area principle, deprecated | [Lec 8.5 Density][005] |



## iPython 

| Function | Description | Link |
|----------|-------------|------|
| `interact(func, arg=val)` | autogenerate UI controls for function arguments, and then calls the function with those arguments when you manipulate the controls interactively; must define `func` | [Using Interact][004] |



## [Conditions][003]

| Predicate | Description |
|-----------|-------------|
| `are.equal_to(Z)` | Equal to `Z` |
| `are.above(x)` | Greater than `x` |
| `are.above_or_equal_to(x)` | Greater than or equal to `x` |
| `are.below(x)` | Less than `x` |
| `are.below_or_equal_to(x)` | Less than or equal to `x` |
| `are.between(x, y)` | Greater than or equal to `x`, and less than `y` |
| `are.strictly_between(x, y)` | Greater than `x` and less than `y` |
| `are.between_or_equal_to(x, y)` | Greater than or equal to `x`, and less than or equal  to `y` |
| `are.containing(S)` | Contains the string `S` |
| `are.not_equal_to(Z)` | Not equal to `Z` |
| `are.not_above(x)` | Not above `x` |



-------------------

[001]: ../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-71-line-graphs
[002]: [005]
[003]: https://www.inferentialthinking.com/chapters/06/2/selecting-rows.html#Some-More-Conditions
[004]: ../DSFund-BerkeleyX/1-CompThinkWPython/labs/lab04/lab04.ipynb
[005]: ../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-85-desity
[006]: ../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-84-drawing-histograms
[007]: ../DSFund-BerkeleyX/1-CompThinkWPython/11-Joins.md#lec-114-maps
[008]: ../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-73-scatter-plots
[009]: ../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-37-bar-charts
[010]: ../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-82-binning
[011]: ../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-77-Distributions
[012]: ../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-44-arrays
[013]: ../DSFund-BerkeleyX/1-CompThinkWPython/01-Intro.md#lec-16-demo-visualizations-2
[014]: ../DSFund-BerkeleyX/1-CompThinkWPython/11-Joins.md#lec-111-joins
[015]: ../DSFund-BerkeleyX/1-CompThinkWPython/10-Groups.md#lec-104-pivot-tables
[016]: ../DSFund-BerkeleyX/1-CompThinkWPython/10-Groups.md#lec-102-cross-classification
[017]: ../DSFund-BerkeleyX/1-CompThinkWPython/10-Groups.md#lec-101-one-attribute-group
[018]: ../DSFund-BerkeleyX/1-CompThinkWPython/09-Functions.md#lec-95-apply
[019]: ../DSFund-BerkeleyX/1-CompThinkWPython/06-Census.md#lec-62-column-arithmetic
[020]: ../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-59-where
[021]: ../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-58-take
[022]: ../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select
[023]: ../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-45-columns
[024]: ../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables
[025]: ../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-57-lists
[026]: ../DSFund-BerkeleyX/1-CompThinkWPython/12-TableExamples.mc#Lec-121-table-method-review
[027]: http://data8.org/datascience/tables.html
[028]: ../DSFund-BerkeleyX/1-CompThinkWPython/13-Iteration.md#Lec-133-random-selection






