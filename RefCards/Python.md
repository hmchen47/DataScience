# Python Reference Card

## Typical Programming Environment

### Data Science

```python
from datascience import *
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plots
```

## Numpy

| Function | Description | Link |
|----------|-------------|------|
| `np.char.count(<data>, 'str')` | Count `str` appearance in `data` | [Lec 1.4 Demo: Little Women](../DSFund-BerkeleyX/1-CompThinkWPython/01-Intro.md#lec-14-demo-little-women) |

## Array

| Function | Description | Link |
|----------|-------------|------|
| `make_array(elt1, elt2, ...)` | create array with given elements or list | [Lec 4.4 Arrays](../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-44-arrays) |
| `ary.barh('label')` | Depict horizontal bar chart with `label` column | [Lec 3.7 Bar Charts](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-37-bar-charts) |


## Table (tbl)

| Function | Description | Link |
|----------|-------------|------|
| `tbl.read_table(filename)` | Read a table from a spreadsheet | [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `tbl.column(col_name_or_index)` | returns an array with only the values in the specified column | [Lec 4.5 Columns](../DSFund-BerkeleyX/1-CompThinkWPython/04-Expression.md#lec-45-columns) |
| `tbl.with_column(label, data)` | Create a table with a single column w/ data as an array |  [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `tbl..with_columns(label1, data1, ...)` | Create a table w/ an array of data for each column |  [Lec 5.1 Creating Tables](../DSFund-BerkeleyX/1-CompThinkWPython/05-Strings.md#lec-51-creating-tables) |
| `tbl.select('label')` | constructs a new table with just the specified columns | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `tbl.drop('label')` | constructs a new table without the specified columns | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `tbl.sort('label')` | constructs a new table, with rows sorted by the specified column | [Lec 3.5 Select](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-35-select) |
| `tbl.sort('label')` | constructs a new table with all columns, but with the rows sorted by the values in that column | [Lec 3.6 Sorting](../DSFund-BerkeleyX/1-CompThinkWPython/03-PythonTables.md#lec-36-sorting) |

+ [Visualization with Table](../DSFund-BerkeleyX/1-CompThinkWPython/01-Intro.md#lec-16-demo-visualizations-2)

    ```python
    Table().with_columns([
        'Chapter Length', [len(c) for c in chapters],
        'Number of Periods', np.char.count(chapters, '.'),
    ]).scatter('Number of Periods')
    ```

## Visualizations 
