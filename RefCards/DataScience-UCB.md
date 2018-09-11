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
| `np.char.count(<data>, 'str')` | Count `str` appearance in `data` | [I-Lec 1.4 Demo: Little Women][002] |
| `np.random.choice(ary, sample_size)` | randomly select an element from `ary` with sample size `size` | [I-Lec 13.3 Random Selection][028] |
| `np.count_nonzero(ary)` | Count nonzero values in array `ary` | [I-Lec 13.3 Random Selection][028] |
| `np.append(ary, val\|aryA)` | append a value or array to `ary` | [I-Lec 13.7 For Statement][029] |




## Array

### Methods

| Function | Description | Link |
|----------|-------------|------|
| `np.array(lst)` | convert a list to an array | |
| `make_array(x1, x2, ...)` | create an array with the given values $xi$, where i = 1, 2, ...  | [I-Lec 4.4 Arrays][012]; [I-Lec 8.2 Binning][010] |



## Table

### `Table` Class methods and Attributes

+ `append(row_or_table)`: Append a row or all rows of a table. An appended table must have all columns of self.
+ `append_column(label, values)`: Appends a column to the table or replaces a column.
+ `apply(fn, *column_or_columns)`: Apply `fn` to each element or elements of `column_or_columns`. If no `column_or_columns` provided, `fn` is applied to each row.
+ `as_html(max_rows=0)`: Format table as HTML.
+ ``as_text(max_rows=0, sep=' | ')`: Format table as text.
+ `bar(column_for_categories=None, select=None, overlay=True, width=6, height=4, **vargs)`: Plot bar charts for the table.
+ `barh(column_for_categories=None, select=None, overlay=True, width=6, **vargs)`: Plot horizontal bar charts for the table.
+ `bin(*columns, **vargs)`: Group values by bin and compute counts per bin by column.
+ `boxplot(**vargs)`: Plots a boxplot for the table.
+ `column(index_or_label)`: Return the values of a column as an array.
+ `column_index(label)`: Return the index of a column by looking up its label.
+ `copy(*, shallow=False)`: Return a copy of a table.
+ `drop(*column_or_columns)`: Return a Table with only columns other than selected label or labels.
+ `exclude()`: Return a new Table without a sequence of rows excluded by number.
+ `group(column_or_label, collect=None)`: Group rows by unique values in a column; count or aggregate others.
+ `group_bar(column_label, **vargs)`: Plot a bar chart for the table.
+ `group_barh(column_label, **vargs)`: Plot a horizontal bar chart for the table.
+ `groups(labels, collect=None)`: Group rows by multiple columns, count or aggregate others.
+ `hist(*columns, overlay=True, bins=None, bin_column=None, unit=None, counts=None, group=None, side_by_side=False, width=6, height=4, **vargs)`: Plots one histogram for each column in columns. If no column is specified, plot all columns.
+ `index_by(column_or_label)`: Return a dict keyed by values in a column that contains lists of rows corresponding to each value.
+ `join(column_label, other, other_label=None)`: Creates a new table with the columns of self and other, containing rows for all values of a column that appear in both tables.
+ `move_to_end(column_label)`: Move a column to the last in order.
+ `move_to_start(column_label)`: Move a column to the first in order.
+ `percentile(p)`: Return a new table with one row containing the pth percentile for each column.
+ `pivot(columns, rows, values=None, collect=None, zero=None)`: Generate a table with a column for each unique value in `columns`, with rows for each unique value in `rows`. Each row counts/aggregates the values that match both row and column based on `collect`.
+ `pivot_bin(pivot_columns, value_column, bins=None, **vargs)`: Form a table with columns formed by the unique tuples in pivot_columns containing counts per bin of the values associated with each tuple in the value_column.
+ `pivot_hist(pivot_column_label, value_column_label, overlay=True, width=6, height=4, **vargs)`: Draw histograms of each category in a column.
+ `plot(column_for_xticks=None, select=None, overlay=True, width=6, height=4, **vargs)`: Plot line charts for the table.
+ `relabel(column_label, new_label)`: Changes the label(s) of column(s) specified by `column_label` to labels in `new_label`.
+ `relabeled(label, new_label)`: Return a new table with ``label`` specifying column label(s) replaced by corresponding `new_label`.
+ `remove(row_or_row_indices)`: Removes a row or multiple rows of a table in place.
+ `row(index)`: Return a row.
+ `sample(k=None, with_replacement=True, weights=None)`: Return a new table where k rows are randomly sampled from the original table.
+ `sample_from_distribution(distribution, k, proportions=False)`: Return a new table with the same number of rows and a new column. The values in the distribution   + `column are define a multinomial. They are replaced by sample counts/proportions in the output.
+ `scatter(column_for_x, select=None, overlay=True, fit_line=False, colors=None, labels=None, sizes=None, width=5, height=5, s=20, **vargs)`: Creates scatterplots, optionally adding a line of best fit.
+ `select(*column_or_columns)`: Return a table with only the columns in `column_or_columns`.
+ `set_format(column_or_columns, formatter)`: Set the format of a column.
+ `show(max_rows=0)`: Display the table.
+ `sort(column_or_label, descending=False, distinct=False)`: Return a Table of rows sorted according to the values in a column.
+ `split(k)`: Return a tuple of two tables where the first table contains `k` rows randomly sampled and the second contains the remaining rows.
+ `stack(key, labels=None)`: Takes k original columns and returns two columns, with col. 1 of all column names and col. 2 of all associated data.
+ `stats(ops=(<built-in function min>, <built-in function max>, <function median at 0x7f5edc2fc048>, <built-in function sum>))`: Compute statistics for each column and place them in a table.
+ `take()`: Return a new Table with selected rows taken by index.
+ `to_array()`: Convert the table to a structured NumPy array.
+ `to_csv(filename)`: Creates a CSV file with the provided filename.
+ `to_df()`: Convert the table to a Pandas DataFrame.
+ `where(column_or_label, value_or_predicate=None, other=None)`: Return a new `Table` containing rows where `value_or_predicate` returns True for values in `column_or_label`.
+ `with_column(label, values, *rest)`: Return a new table with an additional or replaced column.
+ `with_columns(*labels_and_values)`: Return a table with additional or replaced columns.
+ `with_relabeling(*args)`: # Deprecated
+ `with_row(row)`: Return a table with an additional row.
+ `with_rows(rows)`: Return a table with additional rows.

+ __Class Attributes__
    + `column_labels`: Return a tuple of column labels. [Deprecated]
    + `columns`
    + `labels`: Return a tuple of column labels.
    + `num_columns`: Number of columns.
    + `num_rows`: Number of rows.
    + `rows`: Return a view of all rows.
    + `values`: Return data in `self` as a numpy array.

### Attributes

| Attribute | Description | Link |
|-----------|-------------|------|
| `t.num_row` / `t.num_column` | row/column size |  [I-Lec 5.1 Creating Tables][024] |
| `t.labels` | returns a list of column labels of a table | [I-Lec 5.1 Creating Tables][024] |


### Methods ([Ref][027])

| Function | Description | Link |
|----------|-------------|------|
| `Table().with_columns(name, values, ...)` | creates a table with an array of values for each column name |  [I-Lec 5.1 Creating Tables][024] |
| `Table()` | creates an empty table|  [I-Lec 5.1 Creating Tables][024] |
| `t.read_table(filename)` | Read a table from a spreadsheet | [I-Lec 5.1 Creating Tables][024] |
| `t.with_columns(name, values, ...)` | appends a column name with an array of values to an existing table |  [I-Lec 5.1 Creating Tables][024] |
| `t.with_row(lst)` | appends a row w/ given columns to an existing table |  [I-Lec 5.7 Lists][025]  |
| `t.with_rows(lst)` | appends multiple rows (rows of columns) w/ given columns to an existing table |  [I-Lec 5.7 Lists][025] |
| `t.item(int)` | data in column `int` | [I-Lec 5.1 Creating Tables][024] |
| `t.sum(ary), t.max(ary), t.min(ary)` | methods to work with data in columns | [I-Lec 5.1 Creating Tables][024] |
| `t.column(col_name_or_index)` | returns an array with only the values in the specified column | [I-Lec 4.5 Columns][023] |
| `t.select(col[, ...])` | constructs a new table with just the specified columns | [I-Lec 3.5 Select][022]; [I-Lec 12.1 Table Methods][026] |
| `t.drop(col[, ...])` | constructs a new table without the specified columns | [I-Lec 3.5 Select][022]; [I-Lec 12.1 Table Methods][026] |
| `t.sort(col, descending=False, distinct=False)` | constructs a new table, with rows sorted by the specified col | [I-Lec 3.5 Select][022]; [I-Lec 12.1 Table Methods][026] |
| `t.take([row, ...]r)` | keep the numbered rows, index starting at 0 | [I-Lec 5.8 Take][021]; [I-Lec 12.1 Table Methods][026] |
| `t.where(col, are.condition(...))` | keep all rows for which a column's value satisfies a condition | [I-Lec 5.9 Where][020]; [I-Lec 12.1 Table Methods][026] |
| `t.where(col, value)` | keep all rows containing a certain value in a column | [I-Lec 5.9 Where][020] |
| `t.set_format(col, FORMAT)` | convert display format, FORMAT= NumberFormatter, PercentFormatter | [I-Lec 6.2 Column Arithmetic][019] |
| `t.relabel(col, label)` | rename the label of selected column | [I-Lec 7.3 Scatter Plots][008] |
| `t.group(col)` | counting with given label |[I-Lec 7.7 Distributions][011] |
| `t.apply(func, col, ...)` | returns an array where a function is applied to each item in a col | [I-Lec 9.5 Apply][018]; [I-Lec 12.1 Table Methods][026] |
| `t.group(col[, func])` | aggregates all rows with the same value a column into a single row in the resulting table with given function; default=count | [I-Lec 10.1 One Attribute Group][017] |
| `t.groups([col, ...][, func])` | aggregate all rows that share the combination of values in multiple columns; default=count | [I-Lec 10.2 Cross Classification][015] |
| `t.pivot(col, roe, [, values=vals, collect=func])` | returns a pivot table where each unique value in col1 has its own column and each unique value in col2 has its own row. The cells of the grid contain row counts (two arguments) or the values from a third col, aggregated by the collect function | [I-Lec 10.4 Pivot Tables][015]; [I-Lec 12.1 Table Methods][026] |
| `tblA.join(colA, tblB, colB)` | returns a table with the columns of `tblA` and `tblB`, containing rows for all values of a `colA` and colB that appear in both tables | [I-Lec 11.1 Joins][014]; [I-Lec 12.1 Table Methods][026] |
| `t.exclude([row, ...])` | return a table that excludes listed rows from given table | [I-Lec 12.1 Table Methods][026] |
| `t.sample(k=None, with_replacement=True, weights=None)` | Return a new table where k rows are randomly sampled from the original table | [II-Lec 4.2 Sampling][030] |



+ [Visualization with Table][013]

    ```python
    Table().with_columns([
        'Chapter Length', [len(c) for c in chapters],
        'Number of Periods', np.char.count(chapters, '.'),
    ]).scatter('Number of Periods')
    ```

## Visualizations

### Methods

| Function | Description | Link |
|----------|-------------|------|
| `t.plot(column_for_xticks=None, overlay=True)` | Plot line charts for the table | [I-Lec 7.1 Line Graphs][001] |
| `t.scatter(x[, y])` | scatter plot | [I-Lec 7.3 Scatter Plots][008] |
| `t.bar(colX, colY)` | Depict bar chart with `colX` column; numerical vs categorical or distribution | [I-Lec 3.7 Bar Charts][009] | 
| `t.barh(colX, colY)` | Depict horizontal bar chart with `colX` column; numerical vs categorical or distribution | [I-Lec 3.7 Bar Charts][009]; [I-Lec 7.7 Distributions][011] |
| `t.bin(columns, bins=None, range=None, density=None)` | Group values by bin and compute counts per bin by column. If the original table has $n$ columns, the resulting binned table has $n+1$ columns, where column $0$ contains the lower bound of each bin. | [I-Lec 8.2 Binning][010] |
| `t.hist(*columns, overlay=True, bins=None, bin_column=None, unit=None, group=None, side_by_side=False, width=6, height=4, **vargs)` | Plots one histogram for each column in columns. If no column is specified, plot all columns | [I-Lec 8.4 Drawing Histograms][006]|
| `{Marker\|Circle}.map_table(tbl, ...)` | A table containing columns of latitude and longitude values used to generate a map of markers | [I-Lec 11.4 Maps][007] |



### Common Graph Arguments

| Argument | Description | Link |
|----------|-------------|------|
| `overlay` | True = one graph; False = separated graphs | [I-Lec 7.1 Line Graphs][001] |
| `unit` | string as the unit of x-axis | [I-Lec 8.4 Drawing Histograms][006] |
| `normed` | True/False, normalized with area principle, deprecated | [I-Lec 8.5 Density][005] |
| `density` | True/False, normalized with area principle, deprecated | [I-Lec 8.5 Density][005] |



## iPython 

| Function | Description | Link |
|----------|-------------|------|
| `interact(func, arg=val)` | autogenerate UI controls for function arguments, and then calls the function with those arguments when you manipulate the controls interactively; must define `func` | [I-Lab 4 Using Interact][004] |



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
| `are.contained_in(ary|lst)` | contains in array or list |
| `are.not_contained_in(ary|lst)` | not contain in array or list |
| `are.not_equal_to(Z)` | Not equal to `Z` |
| `are.not_above(x)` | Not above `x` |



## [Utility](http://data8.org/datascience/util.html)

| Function | Description | Link |
|----------|-------------|------|
| `sample_proportions(sample_size, probabilities)` | Return the proportion of random draws for each outcome in a distribution | [II-Lec 5.2 Random Selection][031] |
| `percentile(p, arr=None)` | Returns the pth percentile of the input array (the value that is at least as great as p% of the values in the array) | [Percentiles][032] |
| `minimize(f, start=None, smooth=False, log=None, array=False, **vargs)` | Minimize a function f of one or more arguments | [III-Lec 5.4 Least Squares][033] |


-------------------

[001]: ../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-71-line-graphs
[002]: ../DSFund-BerkeleyX/1-CompThinkWPython/01-Intro.md#lec-14-demo-little-women
[003]: https://www.inferentialthinking.com/chapters/06/2/selecting-rows.html#Some-More-Conditions
[004]: ../DSFund-BerkeleyX/1-CompThinkWPython/labs/lab04/lab04.ipynb
[005]: ../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-85-density
[006]: ../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-84-drawing-histograms
[007]: ../DSFund-BerkeleyX/1-CompThinkWPython/11-Joins.md#lec-114-maps
[008]: ../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-73-scatter-plots
[009]: ../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-37-bar-charts
[010]: ../DSFund-BerkeleyX/1-CompThinkWPython/08-Histograms.md#lec-82-binning
[011]: ../DSFund-BerkeleyX/1-CompThinkWPython/07-Charts.md#lec-77-distributions
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
[026]: ../DSFund-BerkeleyX/1-CompThinkWPython/12-TableExamples.md#lec-121-table-method-review
[027]: http://data8.org/datascience/tables.html
[028]: ../DSFund-BerkeleyX/1-CompThinkWPython/13-Iteration.md#lec-133-random-selection
[029]: ../DSFund-BerkeleyX/1-CompThinkWPython/13-Iteration.md#lec-137-for-statements
[030]: ../DSFund-BerkeleyX/2-Inferential/04-SamplingSimulation.md#lec-42-sampling
[031]: ../DSFund-BerkeleyX/2-Inferential/05-Hypothesis.md#lec-52-a-model-about-random-selection
[032]: ../DSFund-BerkeleyX/2-Inferential/10-CI.md#percentiles
[033]: ../DSFund-BerkeleyX/3--PredictML/05-LeastSquare.md#lec-54-least-squares
[034]: ../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-56-minards-map-code






