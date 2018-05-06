# Python for Data Science

## Table of Contents

+ [General](#general)
    + [Open CVS File](#ioen-csv-file)
    + [Methods](#methods)
+ [Date and Times](#date-and-times)
    + [Import Files](#import-files)
    + [Attributes](#attributes)
    + [Methods](#methods-1)
+ [NumPy](#numpy)
    + [Import File](#import-files-1)
    + [General](#general)
    + [Array Creation](#array-creation)
    + [Combining Array](#combining-arrays)
    + [Array Operations](array-operations)
    + [Math Functions](#math-functions)
    + [Indexing/Slicing](#indexingslicing)
    + [Random Number Generator](#random-number-generator)
+ [Pandas](#pandas)
    + [Import File](#import-file)
    + [Series](#series)
        + [Creation](#creation)
        + [Attributes](#attributes-1)
        + [Methods](#methods-2)
        + [Lecture Methods](#lecture-methods)
    + [DataFrame](#dataframe)
        + [Creation](#creation-1)
        + [Attributes](#attributes-2)
        + [Methods](#methods-3)
        + [Lecture Methods](#lecture-methods-1)





## General

### [Open CVS File][001]

`python
    import csv

    %precision 2    # floating precision for printing

    with open('filename.csv') as csvfile:
        mpg = list(csv.DictReader(csvfile))
        # read data and convert to nested dictionary
`


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
| `zip(iter1 [,iter2 [...]])` | Return a zip object whose `.__next__()` method returns a tuple where the i-th element comes from the i-th iterable argument. | [NumPy][005] |


[TOC](#table-of-contents)


## [Date and Times][003]

### Import Files

`python
import datetime as dt
import time as tm
`

### Attributes

| Attribute | Description | Link |
|--------|-------------|------|
| `dt.year` | Year of `dt` | [Dates and Times][002] |
| `dt.month` | Month of `dt` | [Dates and Times][002] |
| `dt.day` | Day of `dt` | [Dates and Times][002] |
| `dt.hour` | Hour of `dt` | [Dates and Times][002] |
| `dt.minute` | Minute of `dt` | [Dates and Times][002] |
| `t.second` | Second of `dt` | [Dates and Times][002] |


### Methods

| Method | Description | Link |
|--------|-------------|------|
| `tm.time()` | returns the current time in seconds since the Epoch. (January 1st, 1970) | [Dates and Times][002] |
| `dt.datetime.fromtimestamp(ts)` | Convert the timestamp `ts` to datetime | [Dates and Times][002] |
| `dt.timedelta(arg=val)` | a duration expressing the difference between `val` `arg`, `arg` = `<days|seconds|microseconds>` and `val` = <int> | [Dates and Times][002] |
| `dt.date.today()` | returns the current local date | [Dates and Times][002] |



[TOC](#table-of-contents)


## Numpy

### Import Files

`python
import numpy as np
`

### General

| Method | Description | Link |
|--------|-------------|------|
| `np.nan` | Not a number | [Series][006] |
| `np.isnan(ary)` | Return ndarray or tuple of ndarray with bool value | [Series][006] |



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
| `np.random.random_sample(size=None)` | Return random floats in the half-open interval $[0.0, 1.0)$  | [NumPy][005] |
| `np.random.rand(d0, d1, ..., dn)` | Random values in a given shape from a uniform distribution over $[0, 1)$ | [NumPy][005] |
| `np.random.randn(d0, d1, ..., dn)` | Return a sample (or samples) from the "standard normal" distribution. | [NumPy][005] |
| `np.random.randint(low, high)` | Return random integers from `low` (inclusive) to `high` (exclusive). | [NumPy][005] |


[TOC](#table-of-contents)





## Pandas

### Import file

`python
import pandas as pd
`

### Series

#### Creation 

Syntax: `pd.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)`

| [Parameter][013] (v0.22.0) | Type | Description |
|-----------|-------------|-------------|
| `data` | array-like, dict, or scalar value | Contains data stored in Series |
| `index` | array-like or Index (1d) | Values must be hashable and have the same length as data. Non-unique index values are allowed. Will default to RangeIndex (len(data)) if not provided. If both a dict and index sequence are used, the index will override the keys found in the dict. |
| `dtype` | numpy.dtype or None | If None, dtype will be inferred |
| `copy` | boolean, default False | Copy input data |

```python
sports = {'Archery': 'Bhutan',
        'Golf': 'Scotland',
        'Sumo': 'Japan',
        'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
```

#### Attributes

| [Attribute][013] (v0.22.0) | Description |
|-----------|-------------|
| `s.T ` | return the transpose, which is by definition self |
| `s.asobject ` | return object Series which contains boxed values |
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


#### Methods

| [Method][011] (v0.22.0) | Description |
|---------------|-------------|
| `s.abs()` | Return an object with absolute value taken–only applicable to objects that are all numeric. |
| `s.add(other[, level, fill_value, axis])` | Addition of series and other, element-wise (binary operator add). |
| `s.add_prefix(prefix)` | Concatenate prefix string with panel items names. |
| `s.add_suffix(suffix)` | Concatenate suffix string with panel items names. |
| `s.agg(func[, axis])` | Aggregate using callable, string, dict, or list of string/callables |
| `s.aggregate(func[, axis])` | Aggregate using callable, string, dict, or list of string/callables |
| `s.align(other[, join, axis, level, copy, ...])` | Align two objects on their axes with the |
| `s.all([axis, bool_only, skipna, level])` | Return whether all elements are True over requested axis |
| `s.any([axis, bool_only, skipna, level])` | Return whether any element is True over requested axis |
| `s.append(to_append[, ignore_index, ...])` | Concatenate two or more Series. |
| `s.apply(func[, convert_dtype, args])` | Invoke function on values of Series. |
| `s.argmax(*args, **kwargs)` |  |
| `s.argmin(*args, **kwargs)` |  |
| `s.argsort([axis, kind, order])` | Overrides ndarray.argsort. |
| `s.as_blocks([copy])` | Convert the frame to a dict of dtype -> Constructor Types that each has a homogeneous dtype. |
| `s.as_matrix([columns])` | Convert the frame to its Numpy-array representation. |
| `s.asfreq(freq[, method, how, normalize, ...])` | Convert TimeSeries to specified frequency. |
| `s.asof(where[, subset])` | The last row without any NaN is taken (or the last row without |
| `s.astype(dtype[, copy, errors])` | Cast a pandas object to a specified dtype dtype. |
| `s.at_time(time[, asof])` | Select values at particular time of day (e.g. |
| `s.autocorr([lag])` | Lag-N autocorrelation |
| `s.between(left, right[, inclusive])` | Return boolean Series equivalent to left <= series <= right. |
| `s.between_time(start_time, end_time[, ...])` | Select values between particular times of the day (e.g., 9:00-9:30 AM). |
| `s.bfill([axis, inplace, limit, downcast])` | Synonym for DataFrame.fillna(method='bfill') |
| `s.bool()` | Return the bool of a single element PandasObject. |
| `s.cat ` | alias of CategoricalAccessor |
| `s.clip([lower, upper, axis, inplace])` | Trim values at input threshold(s). |
| `s.clip_lower(threshold[, axis, inplace])` | Return copy of the input with values below given value(s) truncated. |
| `s.clip_upper(threshold[, axis, inplace])` | Return copy of input with values above given value(s) truncated. |
| `s.combine(other, func[, fill_value])` | Perform elementwise binary operation on two Series using given function |
| `s.combine_first(other)` | Combine Series values, choosing the calling Series’s values first. |
| `s.compound([axis, skipna, level])` | Return the compound percentage of the values for the requested axis |
| `s.compress(condition, *args, **kwargs)` | Return selected slices of an array along given axis as a Series |
| `s.consolidate([inplace])` | DEPRECATED: consolidate will be an internal implementation only. |
| `s.convert_objects([convert_dates, ...])` | Deprecated. |
| `s.copy([deep])` | Make a copy of this objects data. |
| `s.corr(other[, method, min_periods])` | Compute correlation with other Series, excluding missing values |
| `s.count([level])` | Return number of non-NA/null observations in the Series |
| `s.cov(other[, min_periods])` | Compute covariance with Series, excluding missing values |
| `s.cummax([axis, skipna])` | Return cumulative max over requested axis. |
| `s.cummin([axis, skipna])` | Return cumulative minimum over requested axis. |
| `s.cumprod([axis, skipna])` | Return cumulative product over requested axis. |
| `s.cumsum([axis, skipna])` | Return cumulative sum over requested axis. |
| `s.describe([percentiles, include, exclude])` | Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values. |
| `s.diff([periods])` | 1st discrete difference of object |
| `s.div(other[, level, fill_value, axis])` | Floating division of series and other, element-wise (binary operator truediv). |
| `s.divide(other[, level, fill_value, axis])` | Floating division of series and other, element-wise (binary operator truediv). |
| `s.dot(other)` | Matrix multiplication with DataFrame or inner-product with Series |
| `s.drop([labels, axis, index, columns, level, ...])` | Return new object with labels in requested axis removed. |
| `s.drop_duplicates([keep, inplace])` | Return Series with duplicate values removed |
| `s.dropna([axis, inplace])` | Return Series without null values |
| `s.dt ` | alias of CombinedDatetimelikeProperties |
| `s.duplicated([keep])` | Return boolean Series denoting duplicate values |
| `s.eq(other[, level, fill_value, axis])` | Equal to of series and other, element-wise (binary operator eq). |
| `s.equals(other)` | Determines if two NDFrame objects contain the same elements. |
| `s.ewm([com, span, halflife, alpha, ...])` | Provides exponential weighted functions |
| `s.expanding([min_periods, freq, center, axis])` | Provides expanding transformations. |
| `s.factorize([sort, na_sentinel])` | Encode the object as an enumerated type or categorical variable |
| `s.ffill([axis, inplace, limit, downcast])` | Synonym for DataFrame.fillna(method='ffill') |
| `s.fillna([value, method, axis, inplace, ...])` | Fill NA/NaN values using the specified method |
| `s.filter([items, like, regex, axis])` | Subset rows or columns of dataframe according to labels in the specified index. |
| `s.first(offset)` | Convenience method for subsetting initial periods of time series data based on a date offset. |
| `s.first_valid_index()` | Return index for first non-NA/null value. |
| `s.floordiv(other[, level, fill_value, axis])` | Integer division of series and other, element-wise (binary operator floordiv). |
| `s.from_array(arr[, index, name, dtype, copy, ...])` |  |
| `s.from_csv(path[, sep, parse_dates, header, ...])` | Read CSV file (DEPRECATED, please use pandas.read_csv() instead). |
| `s.ge(other[, level, fill_value, axis])` | Greater than or equal to of series and other, element-wise (binary operator ge). |
| `s.get(key[, default])` | Get item from object for given key (DataFrame column, Panel slice, etc.). |
| `s.get_dtype_counts()` | Return the counts of dtypes in this object. |
| `s.get_ftype_counts()` | Return the counts of ftypes in this object. |
| `s.get_value(label[, takeable])` | Quickly retrieve single value at passed index label |
| `s.get_values()` | same as values (but handles sparseness conversions); is a view |
| `s.groupby([by, axis, level, as_index, sort, ...])` | Group series using mapper (dict or key function, apply given function to group, return result as series) or by a series of columns. |
| `s.gt(other[, level, fill_value, axis])` | Greater than of series and other, element-wise (binary operator gt). |
| `s.head([n])` | Return the first n rows. |
| `s.hist([by, ax, grid, xlabelsize, xrot, ...])` | Draw histogram of the input series using matplotlib |
| `s.idxmax([axis, skipna])` | Index label of the first occurrence of maximum of values. |
| `s.idxmin([axis, skipna])` | Index label of the first occurrence of minimum of values. |
| `s.infer_objects()` | Attempt to infer better dtypes for object columns. |
| `s.interpolate([method, axis, limit, inplace, ...])` | Interpolate values according to different methods. |
| `s.isin(values)` | Return a boolean Series showing whether each element in the Series is exactly contained in the passed sequence of values. |
| `s.isna()` | Return a boolean same-sized object indicating if the values are NA. |
| `s.isnull()` | Return a boolean same-sized object indicating if the values are NA. |
| `s.item()` | return the first element of the underlying data as a python |
| `s.items()` | Lazily iterate over (index, value) tuples |
| `s.iteritems()` | Lazily iterate over (index, value) tuples |
| `s.keys()` | Alias for index |
| `s.kurt([axis, skipna, level, numeric_only])` | Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).
| `s.kurtosis([axis, skipna, level, numeric_only])` | Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis (kurtosis of normal  |== | `0.0). |
| `s.last(offset)` | Convenience method for subsetting final periods of time series data based on a date offset. |
| `s.last_valid_index()` | Return index for last non-NA/null value. |
| `s.le(other[, level, fill_value, axis])` | Less than or equal to of series and other, element-wise (binary operator le). |
| `s.lt(other[, level, fill_value, axis])` | Less than of series and other, element-wise (binary operator lt). |
| `s.mad([axis, skipna, level])` | Return the mean absolute deviation of the values for the requested axis |
| `s.map(arg[, na_action])` | Map values of Series using input correspondence (which can be |
| `s.mask(cond[, other, inplace, axis, level, ...])` | Return an object of same shape as self and whose corresponding entries are from self where cond is  |False | `and otherwise are from other. |
| `s.max([axis, skipna, level, numeric_only])` | This method returns the maximum of the values in the object. |
| `s.mean([axis, skipna, level, numeric_only])` | Return the mean of the values for the requested axis |
| `s.median([axis, skipna, level, numeric_only])` | Return the median of the values for the requested axis |
| `s.memory_usage([index, deep])` | Memory usage of the Series |
| `s.min([axis, skipna, level, numeric_only])` | This method returns the minimum of the values in the object. |
| `s.mod(other[, level, fill_value, axis])` | Modulo of series and other, element-wise (binary operator mod). |
| `s.mode()` | Return the mode(s) of the dataset. |
| `s.mul(other[, level, fill_value, axis])` | Multiplication of series and other, element-wise (binary operator mul). |
| `s.multiply(other[, level, fill_value, axis])` | Multiplication of series and other, element-wise (binary operator mul). |
| `s.ne(other[, level, fill_value, axis])` | Not equal to of series and other, element-wise (binary operator ne). |
| `s.nlargest([n, keep])` | Return the largest n elements. |
| `s.nonzero()` | Return the indices of the elements that are non-zero |
| `s.notna()` | Return a boolean same-sized object indicating if the values are not NA. |
| `s.notnull()` | Return a boolean same-sized object indicating if the values are not NA. |
| `s.nsmallest([n, keep])` | Return the smallest n elements. |
| `s.nunique([dropna])` | Return number of unique elements in the object. |
| `s.pct_change([periods, fill_method, limit, freq])` | Percent change over given number of periods. |
| `s.pipe(func, *args, **kwargs)` | Apply func(self, *args, **kwargs) |
| `s.plot ` | alias of SeriesPlotMethods |
| `s.pop(item)` | Return item and drop from frame. |
| `s.pow(other[, level, fill_value, axis])` | Exponential power of series and other, element-wise (binary operator pow). |
| `s.prod([axis, skipna, level, numeric_only, ...])` | Return the product of the values for the requested axis |
| `s.product([axis, skipna, level, numeric_only, ...])` | Return the product of the values for the requested axis |
| `s.ptp([axis, skipna, level, numeric_only])` | Returns the difference between the maximum value and the minimum value in the object. |
| `s.put(*args, **kwargs)` | Applies the put method to its values attribute if it has one. |
| `s.quantile([q, interpolation])` | Return value at the given quantile, a la numpy.percentile. |
| `s.radd(other[, level, fill_value, axis])` | Addition of series and other, element-wise (binary operator radd). |
| `s.rank([axis, method, numeric_only, ...])` | Compute numerical data ranks (1 through n) along axis. |
| `s.ravel([order])` | Return the flattened underlying data as an ndarray |
| `s.rdiv(other[, level, fill_value, axis])` | Floating division of series and other, element-wise (binary operator rtruediv). |
| `s.reindex([index])` | Conform Series to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. |
| `s.reindex_axis(labels[, axis])` | for compatibility with higher dims |
| `s.reindex_like(other[, method, copy, limit, ...])` | Return an object with matching indices to myself. |
| `s.rename([index])` | Alter Series index labels or name |
| `s.rename_axis(mapper[, axis, copy, inplace])` | Alter the name of the index or columns. |
| `s.reorder_levels(order)` | Rearrange index levels using input order. |
| `s.repeat(repeats, *args, **kwargs)` | Repeat elements of an Series. |
| `s.replace([to_replace, value, inplace, limit, ...])` | Replace values given in ‘to_replace’ with ‘value’. |
| `s.resample(rule[, how, axis, fill_method, ...])` | Convenience method for frequency conversion and resampling of time series. |
| `s.reset_index([level, drop, name, inplace])` | Analogous to the pandas.DataFrame.reset_index() function, see docstring there. |
| `s.reshape(*args, **kwargs)` | Deprecated since version 0.19.0. |
| `s.rfloordiv(other[, level, fill_value, axis])` | Integer division of series and other, element-wise (binary operator rfloordiv). |
| `s.rmod(other[, level, fill_value, axis])` | Modulo of series and other, element-wise (binary operator rmod). |
| `s.rmul(other[, level, fill_value, axis])` | Multiplication of series and other, element-wise (binary operator rmul). |
| `s.rolling(window[, min_periods, freq, center, ...])` | Provides rolling window calculations. |
| `s.round([decimals])` | Round each value in a Series to the given number of decimals. |
| `s.rpow(other[, level, fill_value, axis])` | Exponential power of series and other, element-wise (binary operator rpow). |
| `s.rsub(other[, level, fill_value, axis])` | Subtraction of series and other, element-wise (binary operator rsub). |
| `s.rtruediv(other[, level, fill_value, axis])` | Floating division of series and other, element-wise (binary operator rtruediv). |
| `s.sample([n, frac, replace, weights, ...])` | Returns a random sample of items from an axis of object. |
| `s.searchsorted(value[, side, sorter])` | Find indices where elements should be inserted to maintain order. |
| `s.select(crit[, axis])` | Return data corresponding to axis labels matching criteria |
| `s.sem([axis, skipna, level, ddof, numeric_only])` | Return unbiased standard error of the mean over requested axis. |
| `s.set_axis(labels[, axis, inplace])` | Assign desired index to given axis |
| `s.set_value(label, value[, takeable])` | Quickly set single value at passed label. |
| `s.shift([periods, freq, axis])` | Shift index by desired number of periods with an optional time freq |
| `s.skew([axis, skipna, level, numeric_only])` | Return unbiased skew over requested axis |
| `s.slice_shift([periods, axis])` | Equivalent to shift without copying data. |
| `s.sort_index([axis, level, ascending, ...])` | Sort object by labels (along an axis) |
| `s.sort_values([axis, ascending, inplace, ...])` | Sort by the values along either axis |
| `s.sortlevel([level, ascending, sort_remaining])` | DEPRECATED: use Series.sort_index() |
| `s.squeeze([axis])` | Squeeze length 1 dimensions. |
| `s.std([axis, skipna, level, ddof, numeric_only])` | Return sample standard deviation over requested axis. |
| `s.str ` | alias of StringMethods |
| `s.sub(other[, level, fill_value, axis])` | Subtraction of series and other, element-wise (binary operator sub). |
| `s.subtract(other[, level, fill_value, axis])` | Subtraction of series and other, element-wise (binary operator sub). |
| `s.sum([axis, skipna, level, numeric_only, ...])` | Return the sum of the values for the requested axis |
| `s.swapaxes(axis1, axis2[, copy])` | Interchange axes and swap values axes appropriately |
| `s.swaplevel([i, j, copy])` | Swap levels i and j in a MultiIndex |
| `s.tail([n])` | Return the last n rows. |
| `s.take(indices[, axis, convert, is_copy])` | Return the elements in the given positional indices along an axis. |
| `s.to_clipboard([excel, sep])` | Attempt to write text representation of object to the system clipboard This can be pasted into Excel, for example. |
| `s.to_csv([path, index, sep, na_rep, ...])` | Write Series to a comma-separated values (csv) file |
| `s.to_dense()` | Return dense representation of NDFrame (as opposed to sparse) |
| `s.to_dict([into])` | Convert Series to {label -> value} dict or dict-like object. |
| `s.to_excel(excel_writer[, sheet_name, na_rep, ...])` | Write Series to an excel sheet |
| `s.to_frame([name])` | Convert Series to DataFrame |
| `s.to_hdf(path_or_buf, key, **kwargs)` | Write the contained data to an HDF5 file using HDFStore. |
| `s.to_json([path_or_buf, orient, date_format, ...])` | Convert the object to a JSON string. |
| `s.to_latex([buf, columns, col_space, header, ...])` | Render an object to a tabular environment table. |
| `s.to_msgpack([path_or_buf, encoding])` | msgpack (serialize) object to input file path |
| `s.to_period([freq, copy])` | Convert Series from DatetimeIndex to PeriodIndex with desired |
| `s.to_pickle(path[, compression, protocol])` | Pickle (serialize) object to input file path. |
| `s.to_sparse([kind, fill_value])` | Convert Series to SparseSeries |
| `s.to_sql(name, con[, flavor, schema, ...])` | Write records stored in a DataFrame to a SQL database. |
| `s.to_string([buf, na_rep, float_format, ...])` | Render a string representation of the Series |
| `s.to_timestamp([freq, how, copy])` | Cast to datetimeindex of timestamps, at beginning of period |
| `s.to_xarray()` | Return an xarray object from the pandas object. |
| `s.tolist()` | Return a list of the values. |
| `s.transform(func, *args, **kwargs)` | Call function producing a like-indexed NDFrame |
| `s.transpose(*args, **kwargs)` | return the transpose, which is by definition self |
| `s.truediv(other[, level, fill_value, axis])` | Floating division of series and other, element-wise (binary operator truediv). |
| `s.truncate([before, after, axis, copy])` | Truncates a sorted DataFrame/Series before and/or after some particular index value. |
| `s.tshift([periods, freq, axis])` | Shift the time index, using the index’s frequency if available. |
| `s.tz_convert(tz[, axis, level, copy])` | Convert tz-aware axis to target time zone. |
| `s.tz_localize(tz[, axis, level, copy, ambiguous])` | Localize tz-naive TimeSeries to target time zone. |
| `s.unique()` | Return unique values in the object. |
| `s.unstack([level, fill_value])` | Unstack, a.k.a. |
| `s.update(other)` | Modify Series in place using non-NA values from passed Series. |
| `s.valid([inplace])` |  |
| `s.value_counts([normalize, sort, ascending, ...])` | Returns object containing counts of unique values. |
| `s.var([axis, skipna, level, ddof, numeric_only])` | Return unbiased variance over requested axis. |
| `s.view([dtype])` |  |
| `s.where(cond[, other, inplace, axis, level, ...])` | Return an object of same shape as self and whose corresponding entries are from self where cond is True and otherwise are from other. |
| `s.xs(key[, axis, level, drop_level])` | Returns a cross-section (row(s) or column(s)) from the Series/DataFrame. |

[TOC](#table-of-contents)

#### Lecture Methods

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

#### Creation

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

#### Load CSV File

```python
df = pd.read_csv('<fname>.csv')
```

#### Attributes

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

#### Methods

| [Method][011] (v0.22.0) | Description |
|---------------|-------------|
| `df.abs()` | Return an object with absolute value taken–only applicable to objects that are all numeric. |
| `df.add(other[, axis, level, fill_value])` | Addition of dataframe and other, element-wise (binary operator add). |
| `df.add_prefix(prefix)` | Concatenate prefix string with panel items names. |
| `df.add_suffix(suffix)` | Concatenate suffix string with panel items names. |
| `df.agg(func[, axis])` | Aggregate using callable, string, dict, or list of string/callables |
| `df.aggregate(func[, axis])` | Aggregate using callable, string, dict, or list of string/callables |
| `df.align(other[, join, axis, level, copy, ...])` | Align two objects on their axes with the |
| `df.all([axis, bool_only, skipna, level])` | Return whether all elements are True over requested axis |
| `df.any([axis, bool_only, skipna, level])` | Return whether any element is True over requested axis |
| `df.append(other[, ignore_index, verify_integrity])` | Append rows of other to the end of this frame, returning a new object. |
| `df.apply(func[, axis, broadcast, raw, reduce, args])` | Applies function along input axis of DataFrame. |
| `df.applymap(func)` | Apply a function to a DataFrame that is intended to operate elementwise |
| `df.as_blocks([copy])` | Convert the frame to a dict of dtype -> Constructor Types that each has a homogeneous dtype. |
| `df.as_matrix([columns])` | Convert the frame to its Numpy-array representation. |
| `df.asfreq(freq[, method, how, normalize, ...])` | Convert TimeSeries to specified frequency. |
| `df.asof(where[, subset])` | The last row without any NaN is taken (or the last row without |
| `df.assign(**kwargs)` | Assign new columns to a DataFrame, returning a new object (a copy) with all the original columns in addition to the new ones. |
| `df.astype(dtype[, copy, errors])` | Cast a pandas object to a specified dtype dtype. |
| `df.at_time(time[, asof])` | Select values at particular time of day (e.g. |
| `df.between_time(start_time, end_time[, ...])` | Select values between particular times of the day (e.g., 9:00-9:30 AM). |
| `df.bfill([axis, inplace, limit, downcast])` | Synonym for DataFrame.fillna(method='bfill') |
| `df.bool()` | Return the bool of a single element PandasObject. |
| `df.boxplot([column, by, ax, fontsize, rot, ...])` | Make a box plot from DataFrame column optionally grouped by some columns or |
| `df.clip([lower, upper, axis, inplace])` | Trim values at input threshold(s). |
| `df.clip_lower(threshold[, axis, inplace])` | Return copy of the input with values below given value(s) truncated. |
| `df.clip_upper(threshold[, axis, inplace])` | Return copy of input with values above given value(s) truncated. |
| `df.combine(other, func[, fill_value, overwrite])` | Add two DataFrame objects and do not propagate NaN values, so if for a |
| `df.combine_first(other)` | Combine two DataFrame objects and default to non-null values in frame calling the method. |
| `df.compound([axis, skipna, level])` | Return the compound percentage of the values for the requested axis |
| `df.consolidate([inplace])` | DEPRECATED: consolidate will be an internal implementation only. |
| `df.convert_objects([convert_dates, ...])` | Deprecated. |
| `df.copy([deep])` | Make a copy of this objects data. |
| `df.corr([method, min_periods])` | Compute pairwise correlation of columns, excluding NA/null values |
| `df.corrwith(other[, axis, drop])` | Compute pairwise correlation between rows or columns of two DataFrame objects. |
| `df.count([axis, level, numeric_only])` | Return Series with number of non-NA/null observations over requested axis. |
| `df.cov([min_periods])` | Compute pairwise covariance of columns, excluding NA/null values |
| `df.cummax([axis, skipna])` | Return cumulative max over requested axis. |
| `df.cummin([axis, skipna])` | Return cumulative minimum over requested axis. |
| `df.cumprod([axis, skipna])` | Return cumulative product over requested axis. |
| `df.cumsum([axis, skipna])` | Return cumulative sum over requested axis. |
| `df.describe([percentiles, include, exclude])` | Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values. |
| `df.diff([periods, axis])` | 1st discrete difference of object |
| `df.div(other[, axis, level, fill_value])` | Floating division of dataframe and other, element-wise (binary operator truediv). |
| `df.divide(other[, axis, level, fill_value])` | Floating division of dataframe and other, element-wise (binary operator truediv). |
| `df.dot(other)` | Matrix multiplication with DataFrame or Series objects |
| `df.drop([labels, axis, index, columns, level, ...])` | Return new object with labels in requested axis removed. |
| `df.drop_duplicates([subset, keep, inplace])` | Return DataFrame with duplicate rows removed, optionally only |
| `df.dropna([axis, how, thresh, subset, inplace])` | Return object with labels on given axis omitted where alternately any |
| `df.duplicated([subset, keep])` | Return boolean Series denoting duplicate rows, optionally only |
| `df.eq(other[, axis, level])` | Wrapper for flexible comparison methods eq |
| `df.equals(other)` | Determines if two NDFrame objects contain the same elements. |
| `df.eval(expr[, inplace])` | Evaluate an expression in the context of the calling DataFrame instance. |
| `df.ewm([com, span, halflife, alpha, ...])` | Provides exponential weighted functions |
| `df.expanding([min_periods, freq, center, axis])` | Provides expanding transformations. |
| `df.ffill([axis, inplace, limit, downcast])` | Synonym for DataFrame.fillna(method='ffill') |
| `df.fillna([value, method, axis, inplace, ...])` | Fill NA/NaN values using the specified method |
| `df.filter([items, like, regex, axis])` | Subset rows or columns of dataframe according to labels in the specified index. |
| `df.first(offset)` | Convenience method for subsetting initial periods of time series data based on a date offset. |
| `df.first_valid_index()` | Return index for first non-NA/null value. |
| `df.floordiv(other[, axis, level, fill_value])` | Integer division of dataframe and other, element-wise (binary operator floordiv). |
| `df.from_csv(path[, header, sep, index_col, ...])` | Read CSV file (DEPRECATED, please use pandas.read_csv() instead). |
| `df.from_dict(data[, orient, dtype])` | Construct DataFrame from dict of array-like or dicts |
| `df.from_items(items[, columns, orient])` | Convert (key, value) pairs to DataFrame. |
| `df.from_records(data[, index, exclude, ...])` | Convert structured or record ndarray to DataFrame |
| `df.ge(other[, axis, level])` | Wrapper for flexible comparison methods ge |
| `df.get(key[, default])` | Get item from object for given key (DataFrame column, Panel slice, etc.). |
| `df.get_dtype_counts()` | Return the counts of dtypes in this object. |
| `df.get_ftype_counts()` | Return the counts of ftypes in this object. |
| `df.get_value(index, col[, takeable])` | Quickly retrieve single value at passed column and index |
| `df.get_values()` | same as values (but handles sparseness conversions) |
| `df.groupby([by, axis, level, as_index, sort, ...])` | Group series using mapper (dict or key function, apply given function to group, return result as series) or by a series of columns. |
| `df.gt(other[, axis, level])` | Wrapper for flexible comparison methods gt |
| `df.head([n])` | Return the first n rows. |
| `df.hist(data[, column, by, grid, xlabelsize, ...])` | Draw histogram of the DataFrame’s series using matplotlib / pylab. |
| `df.idxmax([axis, skipna])` | Return index of first occurrence of maximum over requested axis. |
| `df.idxmin([axis, skipna])` | Return index of first occurrence of minimum over requested axis. |
| `df.infer_objects()` | Attempt to infer better dtypes for object columns. |
| `df.info([verbose, buf, max_cols, memory_usage, ...])` | Concise summary of a DataFrame. |
| `df.insert(loc, column, value[, allow_duplicates])` | Insert column into DataFrame at specified location. |
| `df.interpolate([method, axis, limit, inplace, ...])` | Interpolate values according to different methods. |
| `df.isin(values)` | Return boolean DataFrame showing whether each element in the DataFrame is contained in values. |
| `df.isna()` | Return a boolean same-sized object indicating if the values are NA. |
| `df.isnull()` | Return a boolean same-sized object indicating if the values are NA. |
| `df.items()` | Iterator over (column name, Series) pairs. |
| `df.iteritems()` | Iterator over (column name, Series) pairs. |
| `df.iterrows()` | Iterate over DataFrame rows as (index, Series) pairs. |
| `df.itertuples([index, name])` | Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple. |
| `df.join(other[, on, how, lsuffix, rsuffix, sort])` | Join columns with other DataFrame either on index or on a key column. |
| `df.keys()` | Get the ‘info axis’ (see Indexing for more) |
| `df.kurt([axis, skipna, level, numeric_only])` | Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0). |
| `df.kurtosis([axis, skipna, level, numeric_only])` | Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis (kurtosis of normal == | `0.0). |
| `df.last(offset)` | Convenience method for subsetting final periods of time series data based on a date offset. |
| `df.last_valid_index()` | Return index for last non-NA/null value. |
| `df.le(other[, axis, level])` | Wrapper for flexible comparison methods le |
| `df.lookup(row_labels, col_labels)` | Label-based “fancy indexing” function for DataFrame. |
| `df.lt(other[, axis, level])` | Wrapper for flexible comparison methods lt |
| `df.mad([axis, skipna, level])` | Return the mean absolute deviation of the values for the requested axis |
| `df.mask(cond[, other, inplace, axis, level, ...])` | Return an object of same shape as self and whose corresponding entries are from self where cond is False and otherwise are from other. |
| `df.max([axis, skipna, level, numeric_only])` | This method returns the maximum of the values in the object. |
| `df.mean([axis, skipna, level, numeric_only])` | Return the mean of the values for the requested axis |
| `df.median([axis, skipna, level, numeric_only])` | Return the median of the values for the requested axis |
| `df.melt([id_vars, value_vars, var_name, ...])` | “Unpivots” a DataFrame from wide format to long format, optionally |
| `df.memory_usage([index, deep])` | Memory usage of DataFrame columns. |
| `df.merge(right[, how, on, left_on, right_on, ...])` | Merge DataFrame objects by performing a database-style join operation by columns or indexes. |
| `df.min([axis, skipna, level, numeric_only])` | This method returns the minimum of the values in the object. |
| `df.mod(other[, axis, level, fill_value])` | Modulo of dataframe and other, element-wise (binary operator mod). |
| `df.mode([axis, numeric_only])` | Gets the mode(s) of each element along the axis selected. |
| `df.mul(other[, axis, level, fill_value])` | Multiplication of dataframe and other, element-wise (binary operator mul). |
| `df.multiply(other[, axis, level, fill_value])` | Multiplication of dataframe and other, element-wise (binary operator mul). |
| `df.ne(other[, axis, level])` | Wrapper for flexible comparison methods ne |
| `df.nlargest(n, columns[, keep])` | Get the rows of a DataFrame sorted by the n largest values of columns. |
| `df.notna()` | Return a boolean same-sized object indicating if the values are not NA. |
| `df.notnull()` | Return a boolean same-sized object indicating if the values are not NA. |
| `df.nsmallest(n, columns[, keep])` | Get the rows of a DataFrame sorted by the n smallest values of columns. |
| `df.nunique([axis, dropna])` | Return Series with number of distinct observations over requested axis. |
| `df.pct_change([periods, fill_method, limit, freq])` | Percent change over given number of periods. |
| `df.pipe(func, *args, **kwargs)` | Apply func(self, *args, **kwargs) |
| `df.pivot([index, columns, values])` | Reshape data (produce a “pivot” table) based on column values. |
| `df.pivot_table([values, index, columns, ...])` | Create a spreadsheet-style pivot table as a DataFrame. |
| `df.plot` | alias of FramePlotMethods |
| `df.pop(item)` | Return item and drop from frame. |
| `df.pow(other[, axis, level, fill_value])` | Exponential power of dataframe and other, element-wise (binary operator pow). |
| `df.prod([axis, skipna, level, numeric_only, ...])` | Return the product of the values for the requested axis |
| `df.product([axis, skipna, level, numeric_only, ...])` | Return the product of the values for the requested axis |
| `df.quantile([q, axis, numeric_only, interpolation])` | Return values at the given quantile over requested axis, a la numpy.percentile. |
| `df.query(expr[, inplace])` | Query the columns of a frame with a boolean expression. |
| `df.radd(other[, axis, level, fill_value])` | Addition of dataframe and other, element-wise (binary operator radd). |
| `df.rank([axis, method, numeric_only, ...])` | Compute numerical data ranks (1 through n) along axis. |
| `df.rdiv(other[, axis, level, fill_value])` | Floating division of dataframe and other, element-wise (binary operator rtruediv). |
| `df.reindex([labels, index, columns, axis, ...])` | Conform DataFrame to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. |
| `df.reindex_axis(labels[, axis, method, level, ...])` | Conform input object to new index with optional filling logic, placing NA/NaN in locations having no  value in the previous index. |
| `df.reindex_like(other[, method, copy, limit, ...])` | Return an object with matching indices to myself. |
| `df.rename([mapper, index, columns, axis, copy, ...])` | Alter axes labels. |
| `df.rename_axis(mapper[, axis, copy, inplace])` | Alter the name of the index or columns. |
| `df.reorder_levels(order[, axis])` | Rearrange index levels using input order. |
| `df.replace([to_replace, value, inplace, limit, ...])` | Replace values given in ‘to_replace’ with ‘value’. |
| `df.resample(rule[, how, axis, fill_method, ...])` | Convenience method for frequency conversion and resampling of time series. |
| `df.reset_index([level, drop, inplace, ...])` | For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the  index names, defaulting to ‘level_0’, ‘level_1’, etc. |
| `df.rfloordiv(other[, axis, level, fill_value])` | Integer division of dataframe and other, element-wise (binary operator rfloordiv). |
| `df.rmod(other[, axis, level, fill_value])` | Modulo of dataframe and other, element-wise (binary operator rmod). |
| `df.rmul(other[, axis, level, fill_value])` | Multiplication of dataframe and other, element-wise (binary operator rmul). |
| `df.rolling(window[, min_periods, freq, center, ...])` | Provides rolling window calculations. |
| `df.round([decimals])` | Round a DataFrame to a variable number of decimal places. |
| `df.rpow(other[, axis, level, fill_value])` | Exponential power of dataframe and other, element-wise (binary operator rpow). |
| `df.rsub(other[, axis, level, fill_value])` | Subtraction of dataframe and other, element-wise (binary operator rsub). |
| `df.rtruediv(other[, axis, level, fill_value])` | Floating division of dataframe and other, element-wise (binary operator rtruediv). |
| `df.sample([n, frac, replace, weights, ...])` | Returns a random sample of items from an axis of object. |
| `df.select(crit[, axis])` | Return data corresponding to axis labels matching criteria |
| `df.select_dtypes([include, exclude])` | Return a subset of a DataFrame including/excluding columns based on their dtype. |
| `df.sem([axis, skipna, level, ddof, numeric_only])` | Return unbiased standard error of the mean over requested axis. |
| `df.set_axis(labels[, axis, inplace])` | Assign desired index to given axis |
| `df.set_index(keys[, drop, append, inplace, ...])` | Set the DataFrame index (row labels) using one or more existing columns. |
| `df.set_value(index, col, value[, takeable])` | Put single value at passed column and index |
| `df.shift([periods, freq, axis])` | Shift index by desired number of periods with an optional time freq |
| `df.skew([axis, skipna, level, numeric_only])` | Return unbiased skew over requested axis |
| `df.slice_shift([periods, axis])` | Equivalent to shift without copying data. |
| `df.sort_index([axis, level, ascending, ...])` | Sort object by labels (along an axis) |
| `df.sort_values(by[, axis, ascending, inplace, ...])` | Sort by the values along either axis |
| `df.sortlevel([level, axis, ascending, inplace, ...])` | DEPRECATED: use DataFrame.sort_index() |
| `df.squeeze([axis])` | Squeeze length 1 dimensions. |
| `df.stack([level, dropna])` | Pivot a level of the (possibly hierarchical) column labels, returning a DataFrame (or Series in the case of an object with a single level of column labels) having a hierarchical index with a new inner-most level of row labels. |
| `df.std([axis, skipna, level, ddof, numeric_only])` | Return sample standard deviation over requested axis. |
| `df.sub(other[, axis, level, fill_value])` | Subtraction of dataframe and other, element-wise (binary operator sub). |
| `df.subtract(other[, axis, level, fill_value])` | Subtraction of dataframe and other, element-wise (binary operator sub). |
| `df.sum([axis, skipna, level, numeric_only, ...])` | Return the sum of the values for the requested axis |
| `df.swapaxes(axis1, axis2[, copy])` | Interchange axes and swap values axes appropriately |
| `df.swaplevel([i, j, axis])` | Swap levels i and j in a MultiIndex on a particular axis |
| `df.tail([n])` | Return the last n rows. |
| `df.take(indices[, axis, convert, is_copy])` | Return the elements in the given positional indices along an axis. |
| `df.to_clipboard([excel, sep])` | Attempt to write text representation of object to the system clipboard This can be pasted into Excel, for example. |
| `df.to_csv([path_or_buf, sep, na_rep, ...])` | Write DataFrame to a comma-separated values (csv) file |
| `df.to_dense()` | Return dense representation of NDFrame (as opposed to sparse) |
| `df.to_dict([orient, into])` | Convert DataFrame to dictionary. |
| `df.to_excel(excel_writer[, sheet_name, na_rep, ...])` | Write DataFrame to an excel sheet |
| `df.to_feather(fname)` | write out the binary feather-format for DataFrames |
| `df.to_gbq(destination_table, project_id[, ...])` | Write a DataFrame to a Google BigQuery table. |
| `df.to_hdf(path_or_buf, key, **kwargs)` | Write the contained data to an HDF5 file using HDFStore. |
| `df.to_html([buf, columns, col_space, header, ...])` | Render a DataFrame as an HTML table. |
| `df.to_json([path_or_buf, orient, date_format, ...])` | Convert the object to a JSON string. |
| `df.to_latex([buf, columns, col_space, header, ...])` | Render an object to a tabular environment table. |
| `df.to_msgpack([path_or_buf, encoding])` | msgpack (serialize) object to input file path |
| `df.to_panel()` | Transform long (stacked) format (DataFrame) into wide (3D, Panel) format. |
| `df.to_parquet(fname[, engine, compression])` | Write a DataFrame to the binary parquet format. |
| `df.to_period([freq, axis, copy])` | Convert DataFrame from DatetimeIndex to PeriodIndex with desired |
| `df.to_pickle(path[, compression, protocol])` | Pickle (serialize) object to input file path. |
| `df.to_records([index, convert_datetime64])` | Convert DataFrame to record array. |
| `df.to_sparse([fill_value, kind])` | Convert to SparseDataFrame |
| `df.to_sql(name, con[, flavor, schema, ...])` | Write records stored in a DataFrame to a SQL database. |
| `df.to_stata(fname[, convert_dates, ...])` | A class for writing Stata binary dta files from array-like objects |
| `df.to_string([buf, columns, col_space, header, ...])` | Render a DataFrame to a console-friendly tabular output. |
| `df.to_timestamp([freq, how, axis, copy])` | Cast to DatetimeIndex of timestamps, at beginning of period |
| `df.to_xarray()` | Return an xarray object from the pandas object. |
| `df.transform(func, *args, **kwargs)` | Call function producing a like-indexed NDFrame |
| `df.transpose(*args, **kwargs)` | Transpose index and columns |
| `df.truediv(other[, axis, level, fill_value])` | Floating division of dataframe and other, element-wise (binary operator truediv). |
| `df.truncate([before, after, axis, copy])` | Truncates a sorted DataFrame/Series before and/or after some particular index value. |
| `df.tshift([periods, freq, axis])` | Shift the time index, using the index’s frequency if available. |
| `df.tz_convert(tz[, axis, level, copy])` | Convert tz-aware axis to target time zone. |
| `df.tz_localize(tz[, axis, level, copy, ambiguous])` | Localize tz-naive TimeSeries to target time zone. |
| `df.unstack([level, fill_value])` | Pivot a level of the (necessarily hierarchical) index labels, returning a DataFrame having a new level of column labels  whose inner-most level consists of the pivoted index labels. |
| `df.update(other[, join, overwrite, ...])` | Modify DataFrame in place using non-NA values from passed DataFrame. |
| `df.var([axis, skipna, level, ddof, numeric_only])` | Return unbiased variance over requested axis. |
| `df.where(cond[, other, inplace, axis, level, ...])` | Return an object of same shape as self and whose corresponding entries are from self where cond is True and otherwise are from other. |
| `df.xs(key[, axis, level, drop_level])` | Returns a cross-section (row(s) or column(s)) from the Series/DataFrame. |

[TOC](#table-of-contents)

#### Indexing & Slicing

| Method | Description | Link |
|--------|-------------|------|
| `df[lbl]` | Column of given `lbl` | [DataFrame][008] |
| `df.loc[lbl]` | Purely label-location based indexer for selection by label. Series of row w/ `lbl` | [DataFrame][008] |
| `df.loc[rlbl, clbl]` | Purely label-location based indexer for selection by label. Value at position (`rlbl`, `clbl`) | [DataFrame][008] |
| `df.loc[rlbl][clbl, ...]` | Purely label-location based indexer for selection by label. Value(s) at position (`rlbl`, `clbl`), ... | [DataFrame][008] |
| `df.iloc[idx]` | Purely integer-location based indexing for selection by position, Series of `idx` row | [DataFrame][008] |
| `df.set_index(keys)` | Set the DataFrame index (row labels) using one or more existing columns. By default yields a new object. | [Indexing DF][012] |
| `df.reset_index(level=None)` | For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the index names, defaulting to 'level_0', 'level_1', etc. if any are None. `level`: int, str, tuple, or list. Only remove the given levels from the index. Removes all levels by default| [Indexing DF][012] |


#### Lecture Methods

| Method | Description | Link |
|--------|-------------|------|
| `df(data, index=None)` | 2-dim size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). `data`: numpy ndarray (structured or homogeneous), dict, or DataFrame Dict can contain Series, arrays, constants, or list-like objects; `index`: Index or array-like. Index to use for resulting frame. Will default to np.arange(n); | [DataFrame][008] |
| `df.head(n=5)` | Return the first n rows  | [DataFrame][008] |
| `df.drop(labels=None, axis=0, index=None, columns=None)` | Return new object with labels in requested axis removed. | [DataFrame][008] |
| `pd.read_csv(fPathName, index_col=None, skiprows=None)` | Read CSV (comma-separated) file into DataFrame, `index_col`: int or sequence or False. Column to use as the row labels of the DataFrame, `skiprows`: list-like or integer or callable. Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file | [DF Index & Load][009] |
| `df.columns` | Index of column labels | [DF Index & Load][009] |
| `df.rename(columns=None, axis=None, inplace=False)` | Alter axes labels; `columns`: columns_mapper, e.g., {"A": "a", "C": "c"}, `axis`: int or str. Axis to target with `mapper`, `inplace`: boolean. Whether to return a new %(klass)s | [DF Index & Load][009] |
| `df.where(cond)` | Return an object of same shape as self and whose corresponding entries are from self where `cond` is True and otherwise are from `other`; `cond`: boolean NDFrame, array-like, or callable. Where `cond` is True, keep the original value. Where False, replace with corresponding value from `other` | [DF Query][010]; [Pandas Idioms][016] |
| `df.count(axis=0)` | Return Series with number of non-NA/null observations over requested axis. Works with non-floating point data as well (detects NaN and None); `axis`: {0 or 'index', 1 or 'columns'}, default 0 or 'index' for row-wise, 1 or 'columns' for column-wise | [DF Query][010] |
| `df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)` | Return object with labels on given axis omitted where alternately any or all of the data are missing; `axis`: {0 or 'index', 1 or 'columns'}, or tuple/list thereof. Pass tuple or list to drop on multiple axes; `how`: {'any', 'all'}, `any`: if any NA values are present, drop that label; `all` if all values are NA, drop that label; `thresh`: int, default None; int value require that many non-NA values; `subset` array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include; `inplace`: boolean, default False, f True, do operation inplace and return None. | [DF Query][010] |
| `df.fillna(value=None, method=None)` | Fill NA/NaN values using the specified method | [Missing Values][014] |
| `df.merge(right, how='inner', left_on=None, right_on=None, left_index=False, right_index=False)` | Merge DataFrame objects by performing a database-style join operation by columns or indexes. `how`: {'left', 'right', 'outer', 'inner'}; `left_on`/`right_on`: label from left/right; `left_index`/`right_index`: indexes from left/right | [Merge DFs][015] |
| `df.applymap(func)` | Apply a function to a DataFrame that is intended to operate elementwise, all elements | [Pandas Idioms][016] |
| `df.apply(func, axis=0)` | Applies function along input axis of DataFrame; `axis`: {0 or 'index', 1 or 'columns'} | [Pandas Idioms][016] |





[TOC](#table-of-contents)






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
[010]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#querying-a-dataFrame
[011]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[012]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#missing-valuesindexing-dataframes
[013]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html
[014]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#missing-values
[015]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#merging-dataframes
[016]: ../AppliedDS-UMich/1-IntroDS/03-AdvPandas.md#pandas-idioms
[017]: 
[018]: 
[019]: 

