# Python for Data Science


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
| `func = lambda var1, ... : expr` | anonymous function, usage: `func(vara, ...)` | [Lambda & List Comprehension][004] |
| `enumerate(iterable[, start])` | Return an enumerate object, obtaining an indexed list:  `(0, seq[0]), (1, seq[1]), (2, seq[2]), ...` | [NumPy][005] |
| `zip(iter1 [,iter2 [...]])` | Return a zip object whose `.__next__()` method returns a tuple where the i-th element comes from the i-th iterable argument. | [NumPy][005] |



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
| `t.second` | Second of `dt` | [Dates and Times][002] |


### Methods

| Method | Description | Link |
|--------|-------------|------|
| `tm.time()` | returns the current time in seconds since the Epoch. (January 1st, 1970) | [Dates and Times][002] |
| `dt.datetime.fromtimestamp(ts)` | Convert the timestamp `ts` to datetime | [Dates and Times][002] |
| `dt.timedelta(arg=val)` | a duration expressing the difference between `val` `arg`, `arg` = `<days|seconds|microseconds>` and `val` = <int> | [Dates and Times][002] |
| `dt.date.today()` | returns the current local date | [Dates and Times][002] |




## Numpy

### Environment and Packages

```python
import numpy as np
```

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
| `np.hstack(tup)` | Stack arrays in sequence horiziontally (column wise). `tup`: sequence of ndarray | [NumPy][005] |


### Array Operations

| Method | Description | Link |
|--------|-------------|------|
| `np.arrayA {+|-|*|/} np.arrayB` | elementwise add/subtract/multiply/divide | [NumPy][005] |
| `np.arrayA.dot(np.arrayB)` | Dot product of two arrays. 1-D - inner product, 2-D - matrix multiplication (`matmul` or `aryA @ aryB` preferred), 0-D (sclar) - multiply, $N \times M$-D - `dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])` | [NumPy][005] |
| `np.array.T` | Transpose of `np.array`  | [NumPy][005] |
| `np.array.dtype` | View the data type of the elements in the array | [NumPy][005] |
| `np.array.astype(typ)` | Cast to a specifica type `typ`  | [NumPy][005] |
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



## Pandas

### Import file

```python
import pandas as pd
```

### Series

| Method | Description | Link |
|--------|-------------|------|
| `pd.Series(data=None, Index=None)` | One-dimensional ndarray with axis labels (including time series). Labels need not be unique but must be a hashable type. `data`: array-like, dict, or scalar, `Index`: labels | [Series][006] |
| `pd.Series.iloc[idx]` | Purely integer-location based indexing for selection by position | [Querying Series][007] |
| `pd.Series.loc[label]` | Purely label-location based indexer for selection by label, or adding w/ assign | [Querying Series][007] |
| `pd.Series.head(n=5)` | Return the first n rows | [Querying Series][007] |
| `pd.Series.set_value(label, value)` | Quickly set single value at passed label.  If label not existed, create and append. | [Querying Series][007] |
| `pd.Series.iteritems()` | Lazily iterate over (index, value) tuples | [Querying Series][007] |
| `pd.Series.append(ser)` | Concatenate two or more Series; `ser`: Series or list/tuple of Series  | [Querying Series][007] |



-------------------------------------

[000]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#python-types-and-sequences
[001]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#python-demonstration-reading-and-writing-csv-files
[002]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#python-dates-and-times
[003]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#advanced-python-objects-map
[004]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#advanced-python-lambda-and-list-comprehensions
[005]: ../AppliedDS-UMich/1-IntroDS/01-PythonFund.md#advanced-python-demonstration-the-numerical-python-library-numPy
[006]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#the-series-data-structure
[007]: ../AppliedDS-UMich/1-IntroDS/02-Pandas.md#querying-a-series
[008]: 
[009]: 
[000]: 
[011]: 
[012]: 
[013]: 
[014]: 
[015]: 
[016]: 
[017]: 
[018]: 
[019]: 

