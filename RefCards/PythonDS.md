# Python for Data Science

## Table of Contents

+ [General](#general)
    + [Open CSV File](open-cvs-file)
    + [Method](methods)
+ [Date and Time](date-and-times)
    + [Import Files](import-files)
    + [Attributes](attributes)
    + [Methods](methods-1)
+ [SciPy](scipy)
    + [Import Files](import-files-1)
    + [Statistical Module](statistical-module)
+ [Numpy](numpy)
    + [Import Files](import-files-2)
    + [General](general-1)
    + [Array Creation](array-creation)
    + [Combining Array](combining-arrays)
    + [Array Operations](array-operations)
    + [Math Functions](math-functions)
    + [Indexing/Slicing](indexingslicing)
    + [Random Number Generator](random-number-generator)
+ [Pandas](pandas)
    + [Import File](import-file)
    + [General](general-2)
    + [Timestamp](timestamp)
    + [Methods](methods-2)
    + [Lecture Methods](lecture-methods)
+ [DataFrame](dataframe)
    + [Class](class)
    + [Local File](load-file)
    + [Attributes](attributes-1)
    + [Indexing & Slicing](indexing--slicing)
    + [Methods](methods-3)
+ [Matplotlib](matplotlib)
    + [Official Pyplot API](official-pyplot-api)
    + [Environment and Modules](environment-and-module)
    + [Classes](classes)
    + [Official Docs](official-docs)
    + [Methods](methods-4)
    + [Line Style or marker](line-style-or-marker)
    + [Color abbreviations](color-abbreviations)
+ [Seaborn](seaborn)
    + [Seaborn API](seaborn-api)



## General

### [Open CVS File][001]

```python
    import csv

    %precision 2    # floating precision for printing

    with open('filename.csv') as csvfile:
        mpg = list(csv.DictReader(csvfile))
        # read data and convert to nested dictionary
```


### Methods

| Method | Description | Link |
|--------|-------------|------|
| `type(obj)` | return the object's type | [Python Types and Sequences][000] |
| `len(obj)` | return size of the given object | [CSV Files][001] |
| `lst.append(elt)` | add `elt` to end of the list `lst` |  [Types and Sequences][000] |
| `dict.keys()` | return dictionary keys; column names of CSV data | [CSV Files][001] |
| `dict.values()` | return values from dictionary | [CSV Files][001] |
| `dict.items()` | return (key, value) pairs of dictionary | [CSV Files][001] |
| `set(obj)` | return the unique values for the class types; set theory in math | [CSV Files][001] |
| `str.split('char')` | separates string at `char` w/o keeping `char` | [Types and Sequence][000] |
| `map(func, iterable, ...)` | return an iterator that applies `func` to every iterable | [Objects & map][003] |
| `func = lambda var1, ...: expr` | anonymous function, usage: `func(var1, ...)` | [Lambda & List Comprehension][004] |
| `enumerate(iterable[, start])` | Return an enumerate object, obtaining an indexed list:  `(0, seq[0]), (1, seq[1]), (2, seq[2]), ...` | [NumPy][005] |
| `zip(iter1 [,iter2 [...]])` | Return a zip object whose `.__next__()` method returns a tuple where the i-th element comes from the i-th iterable argument. | [NumPy][005] [Scatterplots][026] |


[TOC](#table-of-contents)


## [Date and Times][003]

### Import Files

```python
import datetime as dt
import time as tm
```

### Attributes

| Attribute | Description | Link |
|--------|-------------|------|
| `dt.year` | Year of `dt` | [Dates and Times][002] |
| `dt.month` | Month of `dt` | [Dates and Times][002] |
| `dt.day` | Day of `dt` | [Dates and Times][002] |
| `dt.hour` | Hour of `dt` | [Dates and Times][002] |
| `dt.minute` | Minute of `dt` | [Dates and Times][002] |
| `dt.second` | Second of `dt` | [Dates and Times][002] |


### Methods

| Method | Description | Link |
|--------|-------------|------|
| `tm.time()` | returns the current time in seconds since the Epoch. (January 1st, 1970) | [Dates and Times][002] |
| `dt.datetime.fromtimestamp(ts)` | Convert the timestamp `ts` to datetime | [Dates and Times][002] |
| `dt.timedelta(arg=val)` | a duration expressing the difference between `val` `arg`, `arg` = `<days, seconds, microseconds>` and `val` = `<int>` | [Dates and Times][002] |
| `dt.date.today()` | returns the current local date | [Dates and Times][002] |



[TOC](#table-of-contents)


## SciPy 

### Import Files

```python
import scipy.stats as stats
```

### Statistical Module

| Method | Description | Link |
|--------|-------------|------|
| `skew(a, axis=0, bias=True, nan_policy='propagate')` | For normally distributed data, the skewness should be about $0$. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. The function `skewtest` can be used to determine if the skewness value is close enough to 0, statistically speaking;  `a`: ndarray; `axis`: Axis along which the kurtosis is calculated; `bias`: False=statistical bias;  `nan_policy`: {'propagate', 'raise', 'omit'} | [More Distribution][022] |
| `kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate')` | Kurtosis is the fourth central moment divided by the square of the variance. If Fisher's definition is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution; `a`: array data; `axis`: Axis along which the kurtosis is calculated.; `fisher`: True=Fisher's definition (normal ==> 0.0), False=Pearson's definition (normal ==> 3.0); `bias`: False=statistical bias; 
    + `nan_policy`: {'propagate', 'raise', 'omit'} | [More Distribution][022] |
| `ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')` | Calculates the T-test for the means of two independent samples of scores; `a`, `b`: array_like; `axis`: Axis along which to compute test; `equal_var`: True=perform a standard independent 2 sample test that assumes equal population variances, False=perform Welch's t-test, which does not assume equal population variance; `nan_policy`: {'propagate', 'raise', 'omit'}, | [Hypothesis Testing][023] |




## Numpy

### Import Files

```python
import numpy as np
```

### General

| Method | Description | Link |
|--------|-------------|------|
| `np.nan` | Not a number | [Series][006] |
| `np.isnan(ary)` | Return ndarray or tuple of ndarray with bool value | [Series][006] |
| `np.std(a, axis=None, out=None, ddof=0)` | ompute the standard deviation along the specified axis; `a`: array_like; `axis`: Axis or axes along which the standard deviation is computed; `out`: Alternative output array in which to place the result; `dof`: Means Delta Degrees of Freedom | [More Distribution][022] |
| `np.concatenate



### Array Creation

| Method | Description | Link |
|--------|-------------|------|
| `np.array(object, ndmin=0)` | Create an array; `object`: array_like, `ndim`: minimum dimensions | [NumPy][005] |
| `np.arange([start,] stop[, step,])` | Return evenly spaced values within a given interval | [NumPy][005] |
| `np.reshape(ary, newshape, order='C')` | Gives a new shape to an array without changing its data. | [NumPy][005] |
| `np.linspace(start, stop)` | Returns `num` evenly spaced samples, calculated over the interval [`start`, `stop`]. | [NumPy][005] |
| `np.resize(ary, new_shape)` | Return a new array with the specified shape. | [NumPy][005] |
| `np.ones(shape)` | Return a new array of given shape and type, filled with ones. | [NumPy][005] |
| `np.zeros(shape)` | Return a new array of given shape and type, filled with zeros. | [NumPy][005] |
| `np.eye(N)` | Return a 2-D array with ones on the diagonal and zeros elsewhere. `N`=number of rows | [NumPy][005] |
| `np.diag(ary)` | Extract a diagonal or construct a diagonal array. | [NumPy][005] |
| `np.repeats(ary, repeats)` | Repeat elements of an array. `repeats`: int or array of int | [NumPy][005] |
| `np.arrayA.resize(np.arrayB, shape)` | Return a new array with the specified shape. `shape`: int or tuple of int, Shape of resized array. | [NumPy][005] |

### Combining Arrays

| Method | Description | Link |
|--------|-------------|------|
| `np.vstack(tup)` | Stack arrays in sequence vertically (row wise). `tup`: sequence of ndarray | [NumPy][005] |
| `np.hstack(tup)` | Stack arrays in sequence horizontally (column wise). `tup`: sequence of ndarray | [NumPy][005] |


### Array Operations

| Method | Description | Link |
|--------|-------------|------|
| `np.arrayA {+,-,,} np.arrayB` | Elementwise add/subtract/multiply/divide | [NumPy][005] |
| `np.arrayA.dot(np.arrayB)` | Dot product of two arrays. 1-D - inner product, 2-D - matrix multiplication (`matmul` or `aryA @ aryB` preferred), 0-D (sclar) - multiply, $N \times M$-D - `dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])` | [NumPy][005] |
| `np.array.T` | Transpose of `np.array`  | [NumPy][005] |
| `np.array.dtype` | View the data type of the elements in the array | [NumPy][005] |
| `np.array.astype(typ)` | Cast to a specific type `typ`  | [NumPy][005] |
| `np.array.shape` | Show tuple of array dimension | [NumPy][005] |
| `np.array.copy()` | Return an array copy of the given object. | [NumPy][005] |



### Math Functions

| Method | Description | Link |
|--------|-------------|------|
| `np.array.sum()` | Sum of all elements in the array | [NumPy][005] |
| `np.sum(ser)` | Sum of all elements in `ser` | [Querying Series][007] |
| `np.array.min()` | Minimum element of the array | [NumPy][005] |
| `np.array.max()` | Max element of the array | [NumPy][005] |
| `np.array.mean()` | Mean of the elements in the array | [NumPy][005] |
| `np.array.std()` | Standard deviation of the elements in the array | [NumPy][005] |
| `np.array.argmax([a, axis=None])` | Returns the indices of the maximum values along an axis. `axis=None`: all elements| [NumPy][005] |
| `np.array.argmin([a, axis=None])` | Returns the indices of the minimum values along an axis.`axis=None`: all elements | [NumPy][005] |
| `np.arrayA.sum(np.arrayB, axis=None)` | Sum of array elements over a given axis. `axis=None`: all elements | [NumPy][005] |


### Indexing/Slicing

| Method | Description | Link |
|--------|-------------|------|
| `np.array[idx]` | get the value at a specific index w/ 1D array | [NumPy][005] |
| `np.array[start:end]` | get the values between $[start, end)$ w/ 1D array, default to the beginning/end of the array | [NumPy][005] |
| `np.array[row, col]` | get the value at specific indices w/ 2D array | [NumPy][005] |
| `np.array[rstart:rend, cstart:cend]` | get a su-barray w/ given indices, default to the beginning/end of the row/col | [NumPy][005] |
| `np.array[idx {>, >=, <, <=>>} val]` | conditional index  | [NumPy][005] |


### Random Number Generator

| Method | Description | Link |
|--------|-------------|------|
| `np.random.choice(a, size=None)` | Random sample from 1-D array; `a`: array, `size`: output shape, int  or tuple of ints (sizes of dimensions) | [NumPy][005] |
| `np.random.seed()` | seed the generator | [NumPy][005] |
| `np.random.permutation(ary)` | Randomly permute a sequence, or return a permuted range. `ary`: multi-dim array | [NumPy][005] |
| `np.random.shuffle()` | Modify a sequence in-place by shuffling its contents. | [NumPy][005] |
| `np.random.random_sample(size=None)`, `np.random.random(size=None)` | Return random floats in the half-open interval $[0.0, 1.0)$  | [NumPy][005], [Histograms][038] |
| `np.random.rand(d0, d1, ..., dn)` | Random values in a given shape from a uniform distribution over $[0, 1)$ | [NumPy][005] |
| `np.random.randn(d0, d1, ..., dn)` | Return a sample (or samples) from the "standard normal" distribution. | [NumPy][005] |
| `np.random.randint(low, high)` | Return random integers from `low` (inclusive) to `high` (exclusive). | [NumPy][005] |
| `np.random.binomial(n, p, size)` | Draw samples from a binomial distribution; `n`: event occurrence; `p`: probability of each event; `size`: times of the set events | [Distribution][021] |
| `np.random.uniform(low=0.0, high=1.0, size=None)` | Draw samples from a uniform distribution, $[low, high)$; `size`: Output shape. | [More Distribution][022] |
| `np.random.normal(loc=0.0, scale=1.0, size=None)` | Draw random samples from a normal (Gaussian) distribution; `loc`: mean; `scale`: std dev; `size`: Output shape | [More Distribution][022], [Histograms][038] |
| `np.random.chisquare(df, size=None)` | Draw samples from a chi-square distribution; `df`: Number of degrees of freedom, should be $> 0$; `size`: Output shape | [More Distribution][022] |
| `np.random.gamma(shape, scale=1.0, size=None)` | Draw samples from a Gamma distribution. <br/> Samples are drawn from a Gamma distribution with specified parameters, `shape` (sometimes designated "k") and `scale` (sometimes designated "theta"), where both parameters are > 0. | [Histograms][038] |




[TOC](#table-of-contents)



## Pandas

### Import file

```python
import pandas as pd
```

[Pandas Reference](http://pandas.pydata.org/pandas-docs/stable/api.html)

### General

| Method | Description | Link |
|--------|-------------|------|
| `pd.cut(x, bins, right=True, labels=None)` | Return indices of half-open bins to which each value of `x` belongs. Useful for creating bins | [Scales][018] |
| `df.diff(periods=1, axis=0)` | 1st discrete difference of object; `periods`: Periods to shift for forming difference; `axis`: {0 or 'index', 1 or 'columns'} | [Date Functionality][020] |



### Timestamp

| Attribute | unit | Description |
|-----------|-------------|-------------|
| `ts_input` |  datetime-like, str, int, float  | Value to be converted to Timestamp |
| `freq` |  str, DateOffset  |  Offset which Timestamp will have |
| `tz` |  string, `pytz.timezone`, <br/>`dateutil.tz.tzfile` or None  | Time zone for time which Timestamp will have. |
| `unit` |  string  | numpy unit used for conversion, if ts_input is int or float |
| `offset` |  str, DateOffset  | Deprecated, use freq |


### Methods

| Method | Description | Link |
|--------|-------------|------|
|`pd.to_datetime(arg, **kwargs)` | Convert argument to datetime <br/> `kwargs`: errors='raise', dayfirst=False, yearfirst=False, utc=None, box=True, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=False | [Date Functionality][020], [Line Plots][027] |
| `pd.date_range(start=None, end=None, periods=None)` | Return a fixed frequency DatetimeIndex, with day (calendar) as the default frequency; <br/>[`start`, `end`]; `periods`: umber of periods to generate | [Date Functionality][020] |
| `df.asfreq(freq, method=None, fill_value=None)` | Convert TimeSeries to specified frequency; <br/> `freq`: DateOffset object, or string; `method`: {'backfill'/'bfill', 'pad'/'ffill'} | [Date Functionality][020] |
| `pd.plotting.scatter_matrix(frame, *args, **kwds)` | Draw a matrix of scatter plots <br/> `args`: alpha=0.5, figsize=None, ax=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05 | [Plotting w/ Pandas][044] |
| `pd.tools.plotting.parallel_coordinates(data. col)` | Parallel coordinates is a plotting technique for plotting multivariate data ([viki](https://en.wikipedia.org/wiki/Parallel_coordinates)) | [Plotting w/ Pandas][044] |
| `pd.Period(**kwargs)` | <br/> `kwargs`: value=None, freq=None, year=None, month=1, quarter=None, day=1, hour=0, minute=0, second=0 | [Date Functionality][020] |
| `pd.scatter_matrix(frame, alpha=0.5, figsize=None, ax=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds)` | Draw a matrix of scatter plots | [Examine Data][050] |


| Alias | Description | | Alias | Description |
|--------|------------|-|--------|------------|
| `B`     | business day frequency |     | `C`     | custom business day frequency (experimental) | 
| `D`     | calendar day frequency |    | `W`     | weekly frequency |
| `M`     | month end frequency |    | `SM`    | semi-month end frequency (15th and end of month) |
| `BM`    | business month end frequency | | `CBM`   | custom business month end frequency |
| `MS`    | month start frequency | | `SMS`   | semi-month start frequency (1st and 15th) |
| `BMS`   | business month start frequency | | `CBMS`  | custom business month start frequency |
| `Q`     | quarter end frequency | | `BQ`    | business quarter endfrequency |
| `QS`    | quarter start frequency | | `BQS`   | business quarter start frequency |
| `A`     | year end frequency | | `BA`    | business year end frequency |
| `AS`    | year start frequency | | `BAS`   | business year start frequency |
| `BH`    | business hour frequency | | `H`     | hourly frequency |
| `T`     | minutely frequency | | `S`     | secondly frequency |
| `L`     | milliseonds | | `U`     | microseconds |
| `N`     | nanoseconds |



| Attribute | Description |
|-----------|-------------|
| `s.index` | array-like or Index (1d),  Values must be hashable and have the same length as data. Non-unique index values are allowed. Will default to RangeIndex (len(data)) if not provided. If both a dict and index sequence are used, the index will override the keys found in the dict. |
| `s.at ` | Fast label-based scalar accessor |
| `s.axes ` | Return a list of the row axis labels |
| `s.base ` | return the base object if the memory of the underlying data is |
| `s.blocks ` | Internal property, property synonym for as_blocks() |
| `s.data ` | return the data pointer of the underlying data |
| `s.dtype ` | return the dtype object of the underlying data |
| `s.dtypes ` | return the dtype object of the underlying data |
| `s.empty ` |  |
| `s.flags ` |  |
| `s.ftype ` | return if the data is sparse|dense |
| `s.ftypes ` | return if the data is sparse|dense |
| `s.hasnans ` |  |
| `s.iat ` | Fast integer location scalar accessor. |
| `s.iloc ` | Purely integer-location based indexing for selection by position. |
| `s.imag ` |  |
| `s.is_copy ` |  |
| `s.is_monotonic ` | Return boolean if values in the object are |
| `s.is_monotonic_decreasing ` | Return boolean if values in the object are |
| `s.is_monotonic_increasing ` | Return boolean if values in the object are |
| `s.is_unique ` | Return boolean if values in the object are unique |
| `s.itemsize ` | return the size of the dtype of the item of the underlying data |
| `s.ix ` | A primarily label-location based indexer, with integer position fallback. |
| `s.loc ` | Purely label-location based indexer for selection by label. |
| `s.name ` |  |
| `s.nbytes ` | return the number of bytes in the underlying data |
| `s.ndim ` | return the number of dimensions of the underlying data, |
| `s.real ` |  |
| `s.shape ` | return a tuple of the shape of the underlying data |
| `s.size ` | return the number of elements in the underlying data |
| `s.strides ` | return the strides of the underlying data |
| `s.values ` | Return Series as ndarray or ndarray-like |


[TOC](#table-of-contents)

### Lecture Methods

| Method | Description | Link |
|--------|-------------|------|
| `pd.Series(data=None, Index=None)` | One-dimensional ndarray with axis labels (including time series). Labels need not be unique but must be a hashable type. `data`: array-like, dict, or scalar, `Index`: labels | [Series][006] |
| `s.iloc[idx]` | Purely integer-location based indexing for selection by position | [Querying Series][007] |
| `s.loc[label]` | Purely label-location based indexer for selection by label, or adding w/ assign | [Querying Series][007] |
| `s.head(n=5)` | Return the first n rows | [Querying Series][007] |
| `s.set_value(label, value)` | Quickly set single value at passed label.  If label not existed, create and append. | [Querying Series][007] |
| `s.iteritems()` | Lazily iterate over (index, value) tuples | [Querying Series][007] |
| `s.append(ser)` | Concatenate two or more Series; `ser`: Series or list/tuple of Series  | [Querying Series][007] |



### DataFrame

### [Class](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame)

Syntax: `df(data=None, index=None, columns=None, dtype=None, copy=False)`

| [Parameter][011] (v0.22.0) | Type | Description |
|-----------|-------------|-------------|
| `data` | numpy ndarray (structured or homogeneous), dict, or DataFrame | Dict can contain Series, arrays, constants, or list-like objects |
| `index` | Index or array-like | Index to use for resulting frame. Will default to np.arange(n) if no indexing information part of input data and no index provided |
| `columns` | Index or array-like | Column labels to use for resulting frame. Will default to np.arange(n) if no column labels are provided |
| `dtype` | dtype, default None | Data type to force. Only a single dtype is allowed. If None, infer |
| `copy` | boolean, default False | Copy data from inputs. Only affects DataFrame / 2d ndarray input |

```python
# Create from Series
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

# Create from Dictionary
df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
```

 Load File

```python
df = pd.read_csv('<fname>.csv', skiprows=None, index_col=None)

df = pd.read_excel('<fname>.xls', sheet_name=0, header=0, skiprows=None, index_col=None)
df = pd.read_excel('<fname>.xlsx', sheet_name=0, header=0, skiprows=None, index_col=None)
```


### Attributes

| [Attribute][011] (v0.22.0) | Description |
|-----------|-------------|
| `df.T` | Transpose index and columns |
| `df.at` | Fast label-based scalar accessor |
| `df.axes` | Return a list with the row axis labels and column axis labels as the only members. |
| `df.blocks` | Internal property, property synonym for as_blocks() |
| `df.dtypes` | Return the dtypes in this object. |
| `df.empty` | True if NDFrame is entirely empty [no items], meaning any of the axes are of length 0. |
| `df.ftypes` | Return the ftypes (indication of sparse/dense and dtype) in this object. |
| `df.iat` | Fast integer location scalar accessor. |
| `df.iloc` | Purely integer-location based indexing for selection by position. |
| `df.is_copy` |  |
| `df.ix` | A primarily label-location based indexer, with integer position fallback. |
| `df.loc` | Purely label-location based indexer for selection by label. |
| `df.ndim` | Number of axes / array dimensions |
| `df.shape` | Return a tuple representing the dimensionality of the DataFrame. |
| `df.size` | number of elements in the NDFrame |
| `df.style` | Property returning a Styler object containing methods for building a styled HTML representation fo the DataFrame. |
| `df.values` | Numpy representation of NDFrame |

[TOC](#table-of-contents)


### Indexing & Slicing

| Method | Description | Link |
|--------|-------------|------|
| `df[lbl]` | Column of given `lbl` | [DataFrame][008] |
| `df.loc[lbl]` | Purely label-location based indexer for selection by label. Series of row w/ `lbl` | [DataFrame][008] |
| `df.loc[rlbl, clbl]` | Purely label-location based indexer for selection by label. Value at position (`rlbl`, `clbl`) | [DataFrame][008] |
| `df.loc[rlbl][clbl, ...]` | Purely label-location based indexer for selection by label. Value(s) at position (`rlbl`, `clbl`), ... | [DataFrame][008] |
| `df.iloc[idx]` | Purely integer-location based indexing for selection by position, Series of `idx` row | [DataFrame][008] |
| `df.set_index(keys)` | Set the DataFrame index (row labels) using one or more existing columns. By default yields a new object. | [Indexing DF][012] |
| `df.reset_index(level=None)` | For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the index names, defaulting to 'level_0', 'level_1', etc. if any are None. `level`: int, str, tuple, or list. Only remove the given levels from the index. Removes all levels by default| [Indexing DF][012] |


### Methods

| Method | Description | Link |
|--------|-------------|------|
| `df(data, index=None)` | 2-dim size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). <br/>`data`: numpy ndarray (structured or homogeneous), dict, or DataFrame Dict can contain Series, arrays, constants, or list-like objects; <br/>`index`: Index or array-like. Index to use for resulting frame. Will default to np.arange(n); | [DataFrame][008] |
| `df.head(n=5)` | Return the first n rows  | [DataFrame][008] |
| `df.drop(labels=None, axis=0, index=None, columns=None)` | Return new object with labels in requested axis removed. | [DataFrame][008] |
| `pd.read_csv(fPathName, index_col=None, skiprows=None)` | Read CSV (comma-separated) file into DataFrame, <br/>`index_col`: int or sequence or False. Column to use as the row labels of the DataFrame, <br/>`skiprows`: list-like or integer or callable. Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file | [DF Index & Load][009] |
| `df.columns` | Index of column labels | [DF Index & Load][009] |
| `df.rename(columns=None, axis=None, inplace=False)` | Alter axes labels; <br/>`columns`: columns_mapper, e.g., {"A": "a", "C": "c"}, <br/>`axis`: int or str. Axis to target with `mapper`, <br/>`inplace`: boolean. Whether to return a new %(klass)s | [DF Index & Load][009] |
| `df.where(cond)` | Return an object of same shape as self and whose corresponding entries are from self where `cond` is True and otherwise are from `other`; <br/>`cond`: boolean NDFrame, array-like, or callable. Where `cond` is True, keep the original value. Where False, replace with corresponding value from `other` | [DF Query][010]; [Pandas Idioms][016] |
| `df.count(axis=0)` | Return Series with number of non-NA/null observations over requested axis. Works with non-floating point data as well (detects NaN and None); `axis`: {0 or 'index', 1 or 'columns'}, default 0 or 'index' for row-wise, 1 or 'columns' for column-wise | [DF Query][010] |
| `df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)` | Return object with labels on given axis omitted where alternately any or all of the data are missing; <br/>`axis`: {0 or 'index', 1 or 'columns'}, or tuple/list thereof. Pass tuple or list to drop on multiple axes; <br/>`how`: {'any', 'all'}, `any`: if any NA values are present, drop that label; `all` if all values are NA, drop that label; <br/>`thresh`: int, default None; int value require that many non-NA values; <br/>`subset` array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include; <br/>`inplace`: boolean, default False, f True, do operation inplace and return None. | [DF Query][010] |
| `df.fillna(value=None, method=None)` | Fill NA/NaN values using the specified method | [Missing Values][014] |
| `df.merge(right, how='inner', left_on=None, right_on=None, left_index=False, right_index=False)` | Merge DataFrame objects by performing a database-style join operation by columns or indexes. <br/>`how`: {'left', 'right', 'outer', 'inner'}; <br/>`left_on`/`right_on`: label from left/right; <br/>`left_index`/`right_index`: indexes from left/right | [Merge DFs][015] |
| `df.applymap(func)` | Apply a function to a DataFrame that is intended to operate elementwise, all elements | [Pandas Idioms][016] |
| `df.apply(func, axis=0)` | Applies function along input axis of DataFrame; <br/>`axis`: {0 or 'index', 1 or 'columns'} | [Pandas Idioms][016] |
| `df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)` | Return object with labels on given axis omitted where alternately any or all of the data are missing; <br/>`axis`: {0 or 'index', 1 or 'columns'}, or tuple/list; `how`: {'any', 'all'}; `subset`: Labels along other axis to consider; | [Group by][017] |
| `df.groupby(by=None, axis=0, level=None, as_index=True, sort=True)` | Group series using mapper (dict or key function, apply given function to group, return result as series) or by a series of columns; <br/> `by`: mapping, function, str, or iterable; `axis`: 0 (row), 1 (col); `level`: if the axis is a MultiIndex (hierarchical), group by a particular level or levels; `as_index`: return object with group labels as the index; `sort`: Sort group keys | [Group by][017] |
| `df.agg(func, axis=0)` | Aggregate using callable, string, dict, or list of string/callables; <br/>`func`: callable, string, dictionary, or list of string/callables | [Group by][017] |
| `df.astype(dtype)` | Cast a pandas object to a specified dtype `dtype`; <br/>`dtype`: data type, or dict of column name -> data type | [Scales][018] |
| `df.pivot_table(values=None, index=None, columns=None, aggfunc='mean')` | Create a spreadsheet-style pivot table as a DataFrame. The levels in the pivot table will be stored in MultiIndex objects (hierarchical indexes) on the index and columns of the result DataFrame; <br/> values`: column to aggregate; `index`: column, Grouper, array, or list of the previous; `columns`: column, Grouper, array, or list of the previous; `aggfunc`: function or list of functions, default numpy.mean | [Pivot Tables][019] |
| `df.describe(percentiles=None, include=None, exclude=None)` | Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution, excluding `NaN` values. | [Box Plots][039] |
| `df.plot.box(by=None, **kwds)` | Boxplot | [Plotting w/ Pandas][044] |
| `df.plot.hist(by=None, bins=10, **kwds)` | Histogram | [Plotting w/ Pandas][044] |
| `df.plot.kde(**kwds)` | Kernel Density Estimate plot | [Plotting w/ Pandas][044] |


[TOC](#table-of-contents)




## MatPlotLib

### [Official Pyplot API](https://matplotlib.org/api/pyplot_summary.html)

### Environment and Module

```python
%matplotlib notebook                    # provides an interactive environment in Jupyter and IPuthon

import matplotlib as mpl                # load module in CLI

import matplotlib.pyplot as plt         # load pyplot module

import matplotlib.gridspec as gridspec

import mpl_toolkits.axes_grid1.inset_locator as mpl_il

import matplotlib.animation as animation
```

### Classes

| Method | Description | Link |
|--------|-------------|------|
| `mpl.axes.Axes` | contain most of the figure elements: Axis, Tick, Line2D, Text, Polygon, etc., and sets the coordinate system. | [Axes][030] |
| `plt.gca().xaxis` & `plt.gca().yaxis` | xaxis = class XAxis(Axis), yaxis = class YAxis(Axis) | [Line Plots][027] |
| `gridspec.GridSpec` | specifies the geometry of the grid that a subplot will be placed | [Histograms][038] |

### Official Docs

+ [The Matplotlib API][032]
    + [axis and tick API][030]
    + [PyPlot API][031]
    + [Colors in Matplotlib][033]
    + [Figure][034]
    + [Subplot Parameters][035]
    + [text][036]



### Methods

| Method | Description | Link |
|--------|-------------|------|
| `mpl.get_backend()` | Return the name of the current backend | [Basic Plotting][025] |
| `plt.plot(*args, **kwargs)` | Plot lines and/or markers to the Plot lines and/or markers to the `~matplotlib.axes.Axes` class; <br/> __`kwargs`__: agg_filter, alpha, animated, antialiased, axes, clip_box, clip_on, clip_path, color/c, contains, dash_capstyle, dash_joinstyle, dashes, drawstyle, figure, fillstyle, gid, label, linestyle, linewidth, marker, markeredgecolor, markeredgewidth, markerfacecolor, markerfacwidthmarkersize, markevery, path_effects, picker, pickradius, rasterized, sketch_params, snap, solid_capstyle, solid_joinstyle, transform, url, visible, xdata, ydata, zorder | [Basic Plotting][025], [Line Plots][027] |
| `mpl.figure.Figure(*args)` | The Figure instance supports callbacks through a _callbacks_ attribute which is a `matplotlib.cbook.CallbackRegistry` class instance; `args`: figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None  | [Basic Plotting][025] |
| `mpl.backends.backend_agg. FigureCanvasAgg(figure)` | The canvas the figure renders into | [Basic Plotting][025] |
| `fig.add_subplot(*args, **kwargs)` | Add a subplot; <br/> __`kwargs`__: adjustable, agg_filter, alpha, anchor, animated, aspect, autoscale_on, autoscalex_on, autoscaley_on, axes, axes_locator, axisbelow, clip_box, clip_on, clip_path, color_cycle, contains, facecolor, fc, figure, frame_on, gid, label, navigate, navigate_mode, path_effects, picker, position, rasterization_zorder, rasterized, sketch_params, snap, title, transform, url, visible, xbound, xlabel, xlim, xmargin, xscale, xticklabels, xticks, ybound, ylabel, ylim, ymargin, yscale, yticklabels, yticks, zorder | [Basic Plotting][025]; [Examine Data][050] |
| `subplots(nrows=1, ncols=1, *args, **fig_kw)` | Create a figure and a set of subplots <br/> `*args`: `sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None` <br/> Returns: <br/> + `fig` : `matplotlib.figure.Figure` object <br/> + `ax` (Axes object or array of Axes objects): ax can be either a single `matplotlib.axes.Axes` object or an array of Axes objects if more than one subplot was created. | [Subplots][037] |
| `plt.figure(*args, **kwargs)` | Creates a new figure; `args`: figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None | [Basic Plotting][025]; [Subplots][037] |
| `plt.gca(**kwargs)` | Docstring: Get the current `~matplotlib.axes.Axes` instance on the current figure matching the given keyword `args`, or create one.  | [Basic Plotting][025] |
| `plt.gca().axis(*v, **kwargs)` <br/> `plt.gca().axes(*v, **kwargs)` | Get the current `~matplotlib.axes.Axes` instance on the current figure matching the given keyword `args`, or create one. <br/> __`kwargs`__: adjustable, agg_filter, alpha, anchor, animated, aspect, autoscale_on, autoscalex_on, autoscaley_on, axes, axes_locator, axisbelow, clip_box, clip_on, clip_path, color_cycle, contains, facecolor, fc, figure, frame_on, gid, label, navigate, navigate_mode, path_effects, picker, position, rasterization_zorder, rasterized, sketch_params, snap, title, transform, url, visible, xbound, xlabel, xlim, xmargin, xscale, xticklabels, xticks, ybound, ylabel, ylim, ymargin, yscale, yticklabels, yticks, zorder | [Basic Plotting][025] |
| `plt.gca().get_children()` | return a list of child artists | [Basic Plotting][025] |
| `plt.scatter(x, y, *args, **kwargs)` | Make a Scatterplots of `x` vs `y`; `args`: s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None | [Scatterplots][026], [Histograms][038] |
| `plt.xlabel(s, *args, **kwargs)` | Set the `x` axis label of the current axis | [Scatterplots][026] |
| `plt.ylabel(s, *args, **kwargs)` | Set the `y` axis label of the current axis| [Scatterplots][026] |
| `plt.fill_between(x, y1, y2=0, **kwargs)` | Make filled polygons between two curves; <br/> `kwargs`: where=None, interpolate=False, step=None, *, data=None | [Line Plots][027] |
| `plt.bar(left, height, **kwargs)` <br/> `plt.barh(left, height, **kwargs)` | Make a bar plot with rectangles bounded by: `left`, `left` + `width`, `bottom`, `bottom` + `height` (left, right, bottom and top edges) <br/> `kwargs`: width=0.8, bottom=None, hold=None, data=None,  | [Bar Charts][028] |
| `plt.tick_params(axis='both', **kwargs)` | Change the appearance of ticks and tick labels | [Dejunkify][029] |
| `plt.gcf()` | Get a reference to the current figure. | [Subplots][037] |
| `plt.hist(x, *args, **kwargs)` | Plot a histogram <br/> `*args`: `bins=None, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None` | [Histograms][038] |
| `set_title(label, fontdict=None, loc='center', **kwargs)` | Set a title for the axes of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `set_xlim(left=None, right=None, emit=True, auto=False, **kw)` | Set the data limits for the x-axis of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `set_ylim(bottom=None, top=None, emit=True, auto=False, **kw)` | Set the data limits for the y-axis of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `invert_axis()` | Invert the x-axis of `matplotlib.axes._subplots.AxesSubplot` | [Histograms][038] |
| `plt.boxplot(x, *args)` | Make a box and whisker plot <br/> `args`: `notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_xticks=True, autorange=False, zorder=None, hold=None, data=None` | [Box Plots][039] |
| `inset_axes(parent_axes, width, height, *args)`| Create an inset axes with a given width and height of `mpl_toolkits.axes_grid1.inset_locator`.<br/> `args`: loc=1, bbox_to_anchor=None, bbox_transform=None, axes_class=None, axes_kwargs=None, borderpad=0.5 | [Box Plots][039] |
| `margins(*args, **kw)` | Set or retrieve autoscaling margins | [Box Plots][039] |
| `tick_right()`, `tick_left()` | use ticks only on right/left of `matplotlib.axis.YAxis` | [Box Plots][039] |
| `tick_top()`, `tick_bottom()` | use ticks only on top/bottom of `matplotlib.axis.xAxis`  | [Box Plots][039] |
| `plt.hist2d(x, y, *args, **kwargs)` | Make a 2D histogram plot <br/> `*args`: bins=10, range=None, normed=False, weights=None, cmin=None, cmax=None, hold=None, data=None | [Heatmaps][040] |
| `plt.colorbar(mappable=None, cax=None, ax=None, **kw)` | Add a colorbar to a plot | [Heatmaps][040] |
| `plt.cla()` | Clear the current axes | [Animation][041] |
| `annotate(s, xy, *args, **kwargs)` | Annotate the point `xy` with text `s`<br/> `args`: xytext=None, xycoords=None, textcoords =None, arrowprops=None, annotation_clip=None | [Animation][041] |
| `animation.FuncAnimation(fig, func, *args)` | Makes an animation by repeatedly calling a function `func` <br/> `args`: frames=None, init_func=None, fargs=None, save_count=0, interval=200, repeat_delay=None, repeat=True, blit=False | [Animation][041] |
| `mpl.connect(s, func)` | Connect event with string `s` to `func`.  The signature of `func` is `def func(event)` where event is a `matplotlib.backend_bases.Event` instance | [Interactivity][042] |
| `plterrorbar(x, y, *args, **kwargs)` | Plot an errorbar graph. Plot x versus y with error deltas in `yerr` and `xerr`. Vertical errorbars are plotted if `yerr` is not None. Horizontal errorbars are plotted if `xerr` is not None.<br/> `*args`: yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, hold=None, data=None | [Assignment 3][043] |
| `plt.colormaps()` | Matplotlib provides a number of colormaps, and others can be added using `~matplotlib.cm.register_cmap`.  This function documents the built-in colormaps, and will also return a list of all registered colormaps if called. | [Assignment 3][043] |
| `plt.imshow(X, *args, **kwargs)` | Display an image on the axes <br/> `*args`: cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, hold=None, data=None | [Assignment 3][043] |
| `cm.to_rgba(x, alpha=None, bytes=False, norm=True)` | Return a normalized rgba array corresponding to *x* | [Assignment 3][043] |
| `plt.twinx(ax=None)` | Make a second axes that shares the *x*-axis.  The new axes will overlay *ax* (or the current axes if *ax* is *None*).  The ticks for *ax2* will be  placed on the right, and the *ax2* instance is returned. | [Assignment 3][043] |
| `get_legend_handles_labels( legend_handler_map=None)` | Return handles and labels for legend | [Assignment 3][043] |
| `ax.set_zorder(level)` | Set the zorder for the artist.  Artists with lower zorder values are drawn first. | [Assignment 3][043] |
| `ax.set_visible(b)` | Set the artist's visiblity. | [Assignment 3][043] |
| `fig.tight_layout(renderer=None, pad=1.08, h_pad=None, w_pad=None, rect=None)` | Adjust subplot parameters to give specified padding. | [Assignment 3][043] |
| `plt.show(*args, **kw)` | Display a figure. | [Assignment 3][043] |
| `plt,style.user(style)` | Use matplotlib style settings from a style specification. | [Plotting w/ Pandas][044] |
| `ax.set_aspect(aspect, adjustable=None, anchor=None)` | set aspect |   [Plotting w/ Pandas][044] |
| `cm.get_cmap(name=None, lut=None)` | Get a colormap instance, defaulting to rc values if *name* is None. | [Exame Data][050] |
| `plt.savefig(fname, *kwargs*)` | Save the current figure.  | [Exame Data][050] |



### Line style or marker

| character | description | | character | description |
|-----------|-------------|-|-----------|-------------|
| `'-'`  |  solid line style | | `'1'`  |  tri_down marker |
| `'--'` |  dashed line style | | `'2'`  |  tri_up marker |
| `'-.'` |  dash-dot line style | | `'3'`  |  tri_left marker |
| `':'`  |  dotted line style | | `'4'`  |  tri_right marker |
| `'.'`  |  point marker | | `'*'`  |  star marker |
| `','`  |  pixel marker | | `'+'`  |  plus marker |
| `'o'`  |  circle marker | | `'_'`  |  hline marker |
| `'v'`  |  triangle_down marker | | `'^'`  |  triangle_up marker |
| `'<'`  |  triangle_left marker | | `'>'`  |  triangle_right marker |
| `'s'`  |  square marker | | `'p'`  |  pentagon marker |
| `'h'`  |  hexagon1 marker | | `'H'`  |  hexagon2 marker |
| `'x'`  |  x marker | | `'|'`  |  vline marker |
| `'D'`  |  diamond marker | | `'d'`  |  thin_diamond marker |

### Color abbreviations

| character |  color | | character |  color |
|-----------|--------|-|-----------|--------|
| 'b'       |  blue  | | 'm'       |  magenta |
| 'g'       |  green | | 'y'       |  yellow |
| 'r'       |  red   | | 'k'       |  black |
| 'c'       |  cyan  | | 'w'       |  white |

### Examples - Line Plots

```python
plt.plot(x, y)        # plot x and y using default line style and color
plt.plot(x, y, 'bo')  # plot x and y using blue circle markers
plt.plot(y)           # plot y using x as index array 0..N-1
plt.plot(y, 'r+')     # ditto, but with red plusses

plt.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
plt.plot([1,2,3], [1,4,9], 'rs',  label='line 2')
plt.axis([0, 4, 0, 10])
plt.legend()
```

# Seaborn

+ Seaborn is a library for making attractive and informative statistical graphics in Python.
+ [Official Site][046]
+ [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

```python
import seaborn as sns
```

### [seaborn API][045]

| API | Description | Link |
|-----|-------------|------|
| __Axis grids__ | | |
| `FacetGrid(data[, row, col, hue, col_wrap, ...])` | Subplot grid for plotting conditional relationships. | [Link](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid) |
| `factorplot([x, y, hue, data, row, col, ...])` | Draw a categorical plot onto a FacetGrid. | [Link](https://seaborn.pydata.org/generated/seaborn.factorplot.html#seaborn.factorplot) |
| `lmplot(x, y, data[, hue, col, row, palette, ...])` | Plot data and regression model fits across a FacetGrid. | [Link](https://seaborn.pydata.org/generated/seaborn.lmplot.html#seaborn.lmplot) |
| `PairGrid(data[, hue, hue_order, palette, ...])` | Subplot grid for plotting pairwise relationships in a dataset. | [Link](https://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid) |
| `pairplot(data[, hue, hue_order, palette, ...])` | Plot pairwise relationships in a dataset. | [Link](https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot), [Seaborn][048] |
| `JointGrid(x, y[, data, size, ratio, space, ...])` | Grid for drawing a bivariate plot with marginal univariate plots. | [Link](https://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid) |
| `jointplot(x, y[, data, kind, stat_func, ...])` | Draw a plot of two variables with bivariate and univariate graphs. | [Link](https://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot), [Seaborn][048] |
| __Categorical plots__ | | |
| `stripplot([x, y, hue, data, order, ...])` | Draw a scatterplot where one variable is categorical. | [Link](https://seaborn.pydata.org/generated/seaborn.stripplot.html#seaborn.stripplot) |
| `swarmplot([x, y, hue, data, order, ...])` | Draw a categorical scatterplot with non-overlapping points. | [Link](https://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot), [Seaborn][048] |
| `boxplot([x, y, hue, data, order, hue_order, ...])` | Draw a box plot to show distributions with respect to categories. | [Link](https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot) |
| `violinplot([x, y, hue, data, order, ...])` | Draw a combination of boxplot and kernel density estimate. | [Link](https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot), [Seaborn][048] |
| `lvplot([x, y, hue, data, order, hue_order, ...])` | Draw a letter value plot to show distributions of large datasets. | [Link](https://seaborn.pydata.org/generated/seaborn.lvplot.html#seaborn.lvplot) |
| `pointplot([x, y, hue, data, order, ...])` | Show point estimates and confidence intervals using scatter plot glyphs. | [Link](https://seaborn.pydata.org/generated/seaborn.pointplot.html#seaborn.pointplot) |
| `barplot([x, y, hue, data, order, hue_order, ...])` | Show point estimates and confidence intervals as rectangular bars. | [Link](https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot) |
| `countplot([x, y, hue, data, order, ...])` | Show the counts of observations in each categorical bin using bars. | [Link](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot) |
| __Distribution plots__ | | |
| `distplot(a[, bins, hist, kde, rug, fit, ...])` | Flexibly plot a univariate distribution of observations. | [Link](https://seaborn.pydata.org/generated/seaborn.distplot.html#seaborn.distplot), [Seaborn][048] |
| `kdeplot(data[, data2, shade, vertical, ...])` | Fit and plot a univariate or bivariate kernel density estimate. | [Link](https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot), [Seaborn][048] |
| `rugplot(a[, height, axis, ax])` | Plot datapoints in an array as sticks on an axis. | [Link](https://seaborn.pydata.org/generated/seaborn.rugplot.html#seaborn.rugplot) |
| __Regression plots__ | | |
| `regplot(x, y[, data, x_estimator, x_bins, ...])` | Plot data and a linear regression model fit. | [Link](https://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot) |
| `residplot(x, y[, data, lowess, x_partial, ...])` | Plot the residuals of a linear regression. | [Link](https://seaborn.pydata.org/generated/seaborn.residplot.html#seaborn.residplot) |
| __Matrix plots__ | | |
| `heatmap(data[, vmin, vmax, cmap, center, ...])` | Plot rectangular data as a color-encoded matrix. | [Link](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap) |
| `clustermap(data[, pivot_kws, method, ...])` | Plot a matrix dataset as a hierarchically-clustered heatmap. | [Link](https://seaborn.pydata.org/generated/seaborn.clustermap.html#seaborn.clustermap) |
| __Timeseries plots__ | | |
| `tsplot(data[, time, unit, condition, value, ...])` | Plot one or more timeseries with flexible representation of uncertainty. | [Link](https://seaborn.pydata.org/generated/seaborn.tsplot.html#seaborn.tsplot) |
| __Miscellaneous plots__ | | |
| `palplot(pal[, size])` | Plot the values in a color palette as a horizontal array. | [Link](https://seaborn.pydata.org/generated/seaborn.palplot.html#seaborn.palplot) |
| __Style frontend__ | | |
| `set([context, style, palette, font, ...])` | Set aesthetic parameters in one step. | [Link](https://seaborn.pydata.org/generated/seaborn.set.html#seaborn.set) |
| `axes_style([style, rc])` | Return a parameter dict for the aesthetic style of the plots. | [Link](https://seaborn.pydata.org/generated/seaborn.axes_style.html#seaborn.axes_style) |
| `set_style([style, rc])` | Set the aesthetic style of the plots. | [Link](https://seaborn.pydata.org/generated/seaborn.set_style.html#seaborn.set_style) |
| `plotting_context([context, font_scale, rc])` | Return a parameter dict to scale elements of the figure. | [Link](https://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context), [Seaborn][048] |
| `set_context([context, font_scale, rc])` | Set the plotting context parameters. | [Link](https://seaborn.pydata.org/generated/seaborn.set_context.html#seaborn.set_context) |
| `set_color_codes([palette])` | Change how matplotlib color shorthands are interpreted. | [Link](https://seaborn.pydata.org/generated/seaborn.set_color_codes.html#seaborn.set_color_codes) |
| `reset_defaults()` | Restore all RC params to default settings. | [Link](https://seaborn.pydata.org/generated/seaborn.reset_defaults.html#seaborn.reset_defaults) |
| `reset_orig()` | Restore all RC params to original settings (respects custom rc). | [Link](https://seaborn.pydata.org/generated/seaborn.reset_orig.html#seaborn.reset_orig) |
| __Color palettes__ | | |
| `set_palette(palette[, n_colors, desat, ...])` | Set the matplotlib color cycle using a seaborn palette. | [Link](https://seaborn.pydata.org/generated/seaborn.set_palette.html#seaborn.set_palette) |
| `color_palette([palette, n_colors, desat])` | Return a list of colors defining a color palette. | [Link](https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette) |
| `husl_palette([n_colors, h, s, l])` | Get a set of evenly spaced colors in HUSL hue space. | [Link](https://seaborn.pydata.org/generated/seaborn.husl_palette.html#seaborn.husl_palette) |
| `hls_palette([n_colors, h, l, s])` | Get a set of evenly spaced colors in HLS hue space. | [Link](https://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette) |
| `cubehelix_palette([n_colors, start, rot, ...])` | Make a sequential palette from the cubehelix system. | [Link](https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html#seaborn.cubehelix_palette) |
| `dark_palette(color[, n_colors, reverse, ...])` | Make a sequential palette that blends from dark to color. | [Link](https://seaborn.pydata.org/generated/seaborn.dark_palette.html#seaborn.dark_palette) |
| `light_palette(color[, n_colors, reverse, ...])` | Make a sequential palette that blends from light to color. | [Link](https://seaborn.pydata.org/generated/seaborn.light_palette.html#seaborn.light_palette) |
| `diverging_palette(h_neg, h_pos[, s, l, sep, ...])` | Make a diverging palette between two HUSL colors. | [Link](https://seaborn.pydata.org/generated/seaborn.diverging_palette.html#seaborn.diverging_palette) |
| `blend_palette(colors[, n_colors, as_cmap, input])` | Make a palette that blends between a list of colors. | [Link](https://seaborn.pydata.org/generated/seaborn.blend_palette.html#seaborn.blend_palette) |
| `xkcd_palette(colors)` | Make a palette with color names from the xkcd color survey. | [Link](https://seaborn.pydata.org/generated/seaborn.xkcd_palette.html#seaborn.xkcd_palette) |
| `crayon_palette(colors)` | Make a palette with color names from Crayola crayons. | [Link](https://seaborn.pydata.org/generated/seaborn.crayon_palette.html#seaborn.crayon_palette) |
| `mpl_palette(name[, n_colors])` | Return discrete colors from a matplotlib palette. | [Link](https://seaborn.pydata.org/generated/seaborn.mpl_palette.html#seaborn.mpl_palette) |
| __Palette widgets__ | | |
| `choose_colorbrewer_palette(data_type[, as_cmap])` | Select a palette from the ColorBrewer set. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_colorbrewer_palette.html#seaborn.choose_colorbrewer_palette) |
| `choose_cubehelix_palette([as_cmap])` | Launch an interactive widget to create a sequential cubehelix palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_cubehelix_palette.html#seaborn.choose_cubehelix_palette) |
| `choose_light_palette([input, as_cmap])` | Launch an interactive widget to create a light sequential palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_light_palette.html#seaborn.choose_light_palette) |
| `choose_dark_palette([input, as_cmap])` | Launch an interactive widget to create a dark sequential palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_dark_palette.html#seaborn.choose_dark_palette) |
| `choose_diverging_palette([as_cmap])` | Launch an interactive widget to choose a diverging color palette. | [Link](https://seaborn.pydata.org/generated/seaborn.choose_diverging_palette.html#seaborn.choose_diverging_palette) |
| __Utility functions__ | | |
| `despine([fig, ax, top, right, left, bottom, ...])` | Remove the top and right spines from plot(s). | [Link](https://seaborn.pydata.org/generated/seaborn.despine.h) |
| `desaturate(color, prop)` | Decrease the saturation channel of a color by some percent. | [Link](https://seaborn.pydata.org/generated/seaborn.desaturate.html#seaborn.desaturate) |
| `saturate(color)` | Return a fully saturated color with the same hue. | [Link](https://seaborn.pydata.org/generated/seaborn.saturate.html#seaborn.saturate) |
| `set_hls_values(color[, h, l, s])` | Independently manipulate the h, l, or s channels of a color. | [Link](https://seaborn.pydata.org/generated/seaborn.set_hls_values.html#seaborn.set_hls_values) |


# Scikit-learn

+ `scikit-learn`: Python Machine Learning Library
    + [scikit-learn Homepage](http://scikit-learn.org/)
    + [scikit-learn User Guide](http://scikit-learn.org/stable/user_guide.html)
    + [scikit-learn API reference](http://scikit-learn.org/stable/modules/classes.html)


```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
```

## Method

| API | Description | Link |
|-----|-------------|------|
| `train_test_split(*arrays, **options)` | Split arrays or matrices into random train and test subsets | [ML Example][049] |
| `KNeighborsClassifier(`<br/> `n_neighbors=5, weights='uniform',` <br/> `algorithm='auto', leaf_size=30,`<br/> `p=2, metric='minkowski',` <br/> `metric_params=None, n_jobs=1, **kwargs)` | Classifier implementing the k-nearest neighbors vote. | [KNN Calssifier][051] |
| `knn.fit(X, y)` | Fit the model using X as training data and y as target values | [KNN Calssifier][051] |
| `knn.fit(X, y)` | Fit the model using X as training data and y as target values | [KNN Calssifier][051] |
| `knn.score(X, y, sample_weight=None)` | Returns the mean accuracy on the given test data and labels. | [KNN Calssifier][051] |
| `knn.predict(X)` | Predict the class labels for the provided data | [KNN Calssifier][051] |




-------------------------------------

[000]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#python-types-and-sequences
[001]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#python-demonstration-reading-and-writing-csv-files
[002]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#python-dates-and-times
[003]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#advanced-python-objects-map
[004]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#advanced-python-lambda-and-list-comprehensions
[005]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#advanced-python-demonstration-the-numerical-python-library-numPy
[006]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#the-series-data-structure
[007]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#querying-a-series
[008]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#the-dataframe-data-structure
[009]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#dataframe-indexing-and-loading
[010]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#querying-a-dataframe
[011]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[012]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#missing-valuesindexing-dataframes
[013]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html
[014]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#missing-values
[015]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#merging-dataframes
[016]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#pandas-idioms
[017]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#group-by
[018]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#scales
[019]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#pivot_tables
[020]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#date-functionality
[021]: ../AppliedDS-UMich/1-IntroDS/04-Stats.md#distribution
[022]: ./AppliedDS-UMich/1-IntroDS/04-Stats.md#more-distribution
[023]: ./AppliedDS-UMich/1-IntroDS/04-Stats.md#hypothesis-testing-in-python
[024]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
[025]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#basic-plotting-with-matplotlib
[026]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#scatter-plot
[027]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#line-plots
[028]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#bar-charts
[029]: ../AppliedDS-UMich/2-InfoVis/02-BasicChart.md#dejunkifying-a-plot
[030]: https://matplotlib.org/api/axes_api.html
[031]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html
[032]: https://matplotlib.org/api/index.html
[033]: https://matplotlib.org/api/pyplot_summary.html#colors-in-matplotlib
[034]: https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html
[035]: https://matplotlib.org/api/_as_gen/matplotlib.figure.SubplotParams.html
[036]: https://matplotlib.org/api/text_api.html
[037]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#subplots
[038]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#histograms
[039]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#box-plots
[040]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#heatmaps
[041]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#animations
[042]: ../AppliedDS-UMich/2-InfoVis/03-ChartFund.md#interactivity
[043]: ../AppliedDS-UMich/2-InfoVis/asgn03.md#related-methods-used
[044]: ../AppliedDS-UMich/2-InfoVis/04-AppliedVis.md#plotting-with-pandas
[045]: https://seaborn.pydata.org/api.html#api-ref
[046]: https://seaborn.pydata.org/
[047]: https://seaborn.pydata.org/tutorial.html
[048]: ../AppliedDS-UMich/2-InfoVis/04-AppliedVis.md#seaborn
[049]: ../AppliedDS-UMich/3-AML/01-FundML.ms#an-example-machine-learning-problem
[050]: ../AppliedDS-UMich/3-AML/01-FundML.ms#examining-the-data
[051]: ../AppliedDS-UMich/3-AML/01-FundML.ms#k---nearest-neighbors-classification
[052]: 
[053]: 
[054]: 
[055]: 
[056]: 
[057]: 
[058]: 
[059]: 
[060]: 
[061]: 
[062]: 
[063]: 
[064]: 
[065]: 
[066]: 
[067]: 
[068]: 
[069]: 
[070]: 
[071]: 
[072]: 
[073]: 
[074]: 
[075]: 
[076]: 
[077]: 
[078]: 
[079]: 
[080]: 
[081]: 
[082]: 
[083]: 
[084]: 
[085]: 
[086]: 
[087]: 
[088]: 
[089]: 




