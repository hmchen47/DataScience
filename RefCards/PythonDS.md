# Python for Data Science

## Table of Contents

+ [General](#general)
    + [Open CSV File](open-cvs-file)
    + [Method](methods)
+ [Regular Expressions](regular-expressions)
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
+ [NLTK](natural-language-toolkit)


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
| `str.split('char')` | separates string at `char` w/o keeping `char` | [Types and Sequence][000], [Handle Text][048] |
| `map(func, iterable, ...)` | return an iterator that applies `func` to every iterable | [Objects & map][003] |
| `func = lambda var1, ...: expr` | anonymous function, usage: `func(var1, ...)` | [Lambda & List Comprehension][004] |
| `enumerate(iterable[, start])` | Return an enumerate object, obtaining an indexed list:  `(0, seq[0]), (1, seq[1]), (2, seq[2]), ...` | [NumPy][005] |
| `zip(iter1 [,iter2 [...]])` | Return a zip object whose `.__next__()` method returns a tuple where the i-th element comes from the i-th iterable argument. | [NumPy][005] [Scatterplots][026] |
| `str.startswith(prefix[, start[, end]])` | Return True if `S` starts with the specified prefix, False otherwise. | [Handle Text][048] |
| `str.endswith(suffix[, start[, end]])` | Return True if `S` ends with the specified suffix, False otherwise. | [Handle Text][048] |
| `str.isupper()` | Return True if all cased characters in `S` are uppercase and there is at least one cased character in `S`, False otherwise. | [Handle Text][048] |
| `str.islower()` | Return True if all cased characters in `S` are lowercase and there is at least one cased character in `S`, False otherwise. | [Handle Text][048] |
| `str.istitle()` | Return True if `S` is a titlecased string and there is at least one character in `S`, i.e. upper- and titlecase characters may only follow uncased characters and lowercase characters only cased ones. Return False otherwise. | [Handle Text][048] |
| `str.isalpha()` | Return True if all characters in `S` are alphabetic and there is at least one character in `S`, False otherwise. | [Handle Text][048] |
| `str.isdigit()` | Return True if all characters in `S` are digits and there is at least one character in `S`, False otherwise. | [Handle Text][048] |
| `str.isalnum()` | Return True if all characters in `S` are alphanumeric and there is at least one character in `S`, False otherwise. | [Handle Text][048] |
| `str.lower()` | Return a copy of the string `S` converted to lowercase. | [Handle Text][048] |
| `str.upper()` | Return a copy of `S` converted to uppercase. | [Handle Text][048] |
| `str.splitlines([keepends])` | Return a list of the lines in `S`, breaking at line boundaries. Line breaks are not included in the resulting list unless keepends is given and true. | [Handle Text][048] |
| `str.join(iterable)` | Return a string which is the concatenation of the strings in the iterable.  The separator between elements is `S`. | [Handle Text][048] |
| `str.rstrip([chars])` | Return a copy of the string `S` with trailing whitespace removed. If `chars` is given and not None, remove characters in chars instead. | [Handle Text][048] |
| `str.find(sub[, start[, end]])` | Return the lowest index in S where substring sub is found, such that sub is contained within `S[start:end]`.  Optional arguments start and end are interpreted as in slice notation. Return `-1` on failure. | [Handle Text][048] |
| `str.rfind(sub[, start[, end]])` | Return the highest index in `S` where substring `sub` is found, such that `sub` is contained within `S[start:end]`. | [Handle Text][048] |
| `str.replace(old, new[, count])` | Return a copy of `S` with all occurrences of substring old replaced by new. | [Handle Text][048] |
| `fd = open(file, mode='r', buffering=-1, args*)` | Open file and return a stream.  Raise `IOErro`r upon failure. | [Handle Text][048] |
| `fd.read(size=-1, /)` | Read at most n characters from stream. Read from underlying buffer until we have n characters or we hit EOF. | [Handle Text][048] |
| `fd.readline(self, size=-1, /)` | Read until newline or EOF. Returns an empty string if EOF is hit immediately. | [Handle Text][048] |
| `fd.seek(self, cookie, whence=0, /)` | Change stream position. Change the stream position to the given byte offset. | [Handle Text][048] |
| `fd.write(self, text, /)` | Write string to stream. Returns the number of characters written (which is always equal to the length of the string). | [Handle Text][048] |
| `fd.close(self, /)` | Flush and close the IO object. This method has no effect if the file is already closed. | [Handle Text][048] |


[TOC](#table-of-contents)


## [Regular Expression][049]

```python
import re
```

| Attribute | Description | Link |
|--------|-------------|------|
| `compile(pattern, flags=0)` | Compile a regular expression pattern, returning a pattern object. | [RegEx][049] |
| `escape(pattern)` |  Escape all the characters in pattern except ASCII letters, numbers and '_'. | [RegEx][049] |
| `findall(pattern, string, flags=0)` | Return a list of all non-overlapping matches in the string. | [RegEx][049] |
| `finditer(pattern, string, flags=0)` | Return an iterator over all non-overlapping matches in the string. | [RegEx][049] |
| `fullmatch(pattern, string, flags=0)` | Try to apply the pattern to all of the string, returning a match object, or None if no match was found. | [RegEx][049] |
| `match(pattern, string, flags=0)` | Try to apply the pattern at the start of the string, returning a match object, or None if no match was found. | [RegEx][049] |
| `purge()` | Clear the regular expression caches | [RegEx][049] |
| `search(pattern, string, flags=0)` | Scan through string looking for a match to the pattern, returning a match object, or None if no match was found. | [RegEx][049] |
| `split(pattern, string, maxsplit=0, flags=0)` | Split the source string by the occurrences of the pattern, returning a list containing the resulting substrings. | [RegEx][049] |
| `sub(pattern, repl, string, count=0, flags=0)` | Return the string obtained by replacing the leftmost non-overlapping occurrences of the pattern in string by the replacement repl. | [RegEx][049] |
| `subn(pattern, repl, string, count=0, flags=0)` | Return a 2-tuple containing (new_string, number). | [RegEx][049] |
| `template(pattern, flags=0)` | Compile a template pattern, returning a pattern object | [RegEx][049] |

<br/>
[Regular Expressions 101](https://regex101.com/)

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
| `kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate')` | Kurtosis is the fourth central moment divided by the square of the variance. If Fisher's definition is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution; `a`: array data; `axis`: Axis along which the kurtosis is calculated.; `fisher`: True=Fisher's definition (normal ==> 0.0), False=Pearson's definition (normal ==> 3.0); `bias`: False=statistical bias; `nan_policy`: {'propagate', 'raise', 'omit'} | [More Distribution][022] |
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
| `np.bincount(x, weights=None, minlength=0)` | Count number of occurrences of each value in array of non-negative ints | [Model Evaluation][052] |



### Array Creation

| Method | Description | Link |
|--------|-------------|------|
| `np.array(object, ndmin=0)` | Create an array; `object`: array_like, `ndim`: minimum dimensions | [NumPy][005] |
| `np.arange([start,] stop[, step,])` | Return evenly spaced values within a given interval | [NumPy][005] |
| `np.reshape(ary, newshape, order='C')` | Gives a new shape to an array without changing its data. | [NumPy][005]; [KNN Reg][052] |
| `np.linspace(start, stop)` | Returns `num` evenly spaced samples, calculated over the interval [`start`, `stop`]. | [NumPy][005] |
| `np.resize(ary, new_shape)` | Return a new array with the specified shape. | [NumPy][005]; [KNN Reg][052] |
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
| `pd.cut(x, bins, right=True, labels=None)` | Return indices of half-open bins to which each value of `x` belongs. Useful for creating bins | [Scales][018], [Asgn04-3][055] |
| `pd.concat(objs, axis=0, join='outer', join_axes=None, *args)` | Concatenate pandas objects along a particular axis with optional set logic along the other axes | [Asgn01][054] |
| `get_dummies(data, *args)` | Convert categorical variable into dummy/indicator variables | [Asgn04-3][055] |




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


<br/>


<table style="margin: 0 auto; border: 1px solid black; border-collapse: collapse; width: 60vw;">
  <thead>
  <tr style="border-bottom: double black;">
    <th style="width: 5vw; font-size: 1.2em; border-right: double back; background-color: #4CAF50;"> Alias </th>
    <th style="width: 25vw; text-align: left; font-size: 1.2em; background-color: #4CAF50; border-right: double white;"> Description </th>
    <th style="width: 5vw; font-size: 1.2em; border-right: double back; background-color: #4CAF50;"> Alias </th>
    <th style="width: 25vw; text-align: left; font-size: 1.2em; background-color: #4CAF50;"> Description </th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> B </td>
    <td style="padding: 0.3em;"> business day frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> C </td>
    <td style="padding: 0.3em;"> custom business day frequency (experimental) </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> D </td>
    <td style="padding: 0.3em;"> calendar day frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> W </td>
    <td style="padding: 0.3em;"> weekly frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> M </td>
    <td style="padding: 0.3em;"> month end frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> SM </td>
    <td style="padding: 0.3em;"> semi-month end frequency (15th and end of month) </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> BM </td>
    <td style="padding: 0.3em;"> business month end frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> CBM </td>
    <td style="padding: 0.3em;"> custom business month end frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> MS </td>
    <td style="padding: 0.3em;"> month start frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> SMS </td>
    <td style="padding: 0.3em;"> semi-month start frequency (1st and 15th) </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> BMS </td>
    <td style="padding: 0.3em;"> business month start frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> CBMS </td>
    <td style="padding: 0.3em;"> custom business month start frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> Q </td>
    <td style="padding: 0.3em;"> quarter end frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> BQ </td>
    <td style="padding: 0.3em;"> business quarter end frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> QS </td>
    <td style="padding: 0.3em;"> quarter start frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> BQS </td>
    <td style="padding: 0.3em;"> business quarter start frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> A </td>
    <td style="padding: 0.3em;"> year end frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> BA </td>
    <td style="padding: 0.3em;"> business year end frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> AS </td>
    <td style="padding: 0.3em;"> year start frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> BAS </td>
    <td style="padding: 0.3em;"> business year start frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> BH </td>
    <td style="padding: 0.3em;"> business hour frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> H </td>
    <td style="padding: 0.3em;"> hourly frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> T </td>
    <td style="padding: 0.3em;"> minutely frequency </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> S </td>
    <td style="padding: 0.3em;"> secondly frequency </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> L </td>
    <td style="padding: 0.3em;"> milliseconds </td>
    <td style="padding: 0.3em; font-weight: bold; padding-left: 0.5em;"> U </td>
    <td style="padding: 0.3em;"> microseconds </td>
  </tr>
  <tr>
    <td style="padding: 0.3em; font-weight: bold;"> N </td>
    <td style="padding: 0.3em;"> nanoseconds </td>
    <td style="padding: 0.3em; font-weight: bold;">  </td>
    <td style="padding: 0.3em;">  </td>
  </tr>
  </tbody>
</table>  



## Series

| Attribute | Description |
|-----------|-------------|
| `s.index` | array-like or Index (1d),  Values must be hashable and have the same length as data. Non-unique index values are allowed. Will default to RangeIndex (len(data)) if not provided. If both a dict and index sequence are used, the index will override the keys found in the dict. |
| `s.at` | Fast label-based scalar accessor |
| `s.axes` | Return a list of the row axis labels |
| `s.base` | return the base object if the memory of the underlying data is |
| `s.blocks` | Internal property, property synonym for as_blocks() |
| `s.data` | return the data pointer of the underlying data |
| `s.dtype` | return the dtype object of the underlying data |
| `s.dtypes` | return the dtype object of the underlying data |
| `s.empty` |  |
| `s.flags` |  |
| `s.ftype` | return if the data is sparse|dense |
| `s.ftypes` | return if the data is sparse|dense |
| `s.hasnans` |  |
| `s.iat` | Fast integer location scalar accessor. |
| `s.iloc` | Purely integer-location based indexing for selection by position. |
| `s.imag` |  |
| `s.is_copy` |  |
| `s.is_monotonic` | Return boolean if values in the object are |
| `s.is_monotonic_decreasing` | Return boolean if values in the object are |
| `s.is_monotonic_increasing` | Return boolean if values in the object are |
| `s.is_unique` | Return boolean if values in the object are unique |
| `s.itemsize` | return the size of the dtype of the item of the underlying data |
| `s.ix` | A primarily label-location based indexer, with integer position fallback. |
| `s.loc` | Purely label-location based indexer for selection by label. |
| `s.name` |  |
| `s.nbytes` | return the number of bytes in the underlying data |
| `s.ndim` | return the number of dimensions of the underlying data, |
| `s.real` |  |
| `s.shape` | return a tuple of the shape of the underlying data |
| `s.size` | return the number of elements in the underlying data |
| `s.strides` | return the strides of the underlying data |
| `s.values` | Return Series as ndarray or ndarray-like |


### Lecture Methods

| Method | Description | Link |
|--------|-------------|------|
| `pd.Series(data=None, Index=None)` | One-dimensional ndarray with axis labels (including time series). Labels need not be unique but must be a hashable type | [Series][006] |
| `s.iloc[idx]` | Purely integer-location based indexing for selection by position | [Querying Series][007] |
| `s.loc[label]` | Purely label-location based indexer for selection by label, or adding w/ assign | [Querying Series][007] |
| `s.head(n=5)` | Return the first n rows | [Querying Series][007] |
| `s.set_value(label, value)` | Quickly set single value at passed label.  If label not existed, create and append. | [Querying Series][007] |
| `s.iteritems()` | Lazily iterate over (index, value) tuples | [Querying Series][007] |
| `s.append(ser)` | Concatenate two or more Series; `ser`: Series or list/tuple of Series  | [Querying Series][007] |
| `s.rename(index=None, **kwargs)` | Alter axes input function or functions | [Asgn04-3][055] |


[TOC](#table-of-contents)


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
| `df.index.duplicated(keep='first')` | Return boolean np.ndarray denoting duplicate values | [Asgn01][054] |
| `df.sort_values(axis=0, ascending=True, *args)` | Sort by the values along either axis | [Asgn01][054] |
| `df.diff(periods=1, axis=0)` | 1st discrete difference of object; `periods`: Periods to shift for forming difference; `axis`: {0 or 'index', 1 or 'columns'} | [Date Functionality][020] |
| `df.plot.box(by=None, **kwds)` | Boxplot | [Plotting w/ Pandas][044] |
| `df.plot.hist(by=None, bins=10, **kwds)` | Histogram | [Plotting w/ Pandas][044] |
| `df.plot.kde(**kwds)` | Kernel Density Estimate plot | [Plotting w/ Pandas][044] |


[TOC](#table-of-contents)

### `pandas.core.strings.StringMethods` Methods

| Method | Description | Link |
|--------|-------------|------|
| `len()` | Compute length of each string in the Series/Index. | [Pandas Regex][050] |
| `split(pat=None, n=-1, expand=False)` | Split each string (a la re.split) in the Series/Index by given pattern, propagating NA values. | [Pandas Regex][050], [Asgn01][054] |
| `contains(pat, case=True, flags=0, na=nan, regex=True)` | Return boolean Series/array whether given pattern/regex is contained in each string in the Series/Index | [Pandas Regex][050] |
| `str_count(pat, flags=0, **kwargs)` | Count occurrences of pattern in each string of the Series/Index | [Pandas Regex][050] |
| `df.str.find(sub, start=0, end=None)` | Return lowest indexes in each strings in the Series/Index where the substring is fully contained between [start:end]. Return -1 on failure | [Asgn01][054] |
| `str_findall(pat, flags=0, **kwargs)` | Find all occurrences of pattern or regular expression in the Series/Index | [Pandas Regex][050], [Asgn01][054] |
| `replace(pat, repl, n=-1, case=None, flags=0)` | Replace occurrences of pattern/regex in the Series/Index with some other string | [Pandas Regex][050], [Asgn04-3][055] |
| `extract(pat, flags=0, expand=None)` | For each subject string in the Series, extract groups from the first match of regular expression `pat` | [Pandas Regex][050], [Asgn01][054] |
| `df.str.extractall(pat, flags=0)` | For each subject string in the Series, extract groups from all matches of regular expression pat. When each subject string in the Series has exactly one match, extractall(pat).xs(0, level='match') is the same as extract(pat). | [Asgn01][054] |
| 


[TOC](#table-of-contents)


## Natural Language Toolkit

```python
import nltk

nltk.download()     # fownload corpora
```

### Methods

| Method | Description | Link |
|--------|-------------|------|
| `FreqDist(samples=None)` | A frequency distribution for the outcomes of an experiment | [Basic NLP][051] |
| `nltk.corpus` | NLTK corpus readers, create a LazyModule instance wrapping module name | [Basic NLP][051] |
| `nltk.corpus.udhr` | Construct a new plaintext corpus reader for a set of documents located at the given root directory | [Basic NLP][051] |
| `nltk.corpus.udhr.words(fileids=None)` | return the given file(s) as a list of words and punctuation symbols | [Basic NLP][051] |
| `nltk.PorterStemmer(mode='NLTK_EXTENSIONS')` | A word stemmer based on the Porter stemming algorithm | [Basic NLP][051] |
| `nltk.PorterStemmer.stem(word)` | Strip affixes from the token and return the stem | [Basic NLP][051] |
| `nltk.WordNetLemmatizer()` | Initizlize lemmatization by using WordNet's built-in morphy function | [Basic NLP][051] |
| `nltk.WordNetLemmatizer.lemmatize(word, pos='n')` | Lemmatize using WordNet's built-in morphy function | [Basic NLP][051] |
| `nltk.word_tokenize(text, *args)` | Return a tokenized copy of text, using NLTK's recommended word tokenizer | [Basic NLP][051] |
| `nltk.sent_tokenize(text, language='english')`| Return a sentence-tokenized copy of text, using NLTK's recommended sentence tokenizer | [Basic NLP][051] |
| `nltk.help.upenn_tagset(tagpattern=None)` | detail explanation of given tag | [Adv NLP][052] |
| `nltk.pos_tag(tokens, tagset=None, lang='eng')` | Use NLTK's currently recommended part of speech tagger to tag the given list of tokens | [Adv NLP][052] |
| `nltk.CFG(start, productions, calculate_leftcorners=True)` | A context-free grammar.  A grammar consists of a start state and a set of productions. | [Adv NLP][052] |
| `nltk.CFG.fromstring(input, encoding=None)` | Return the `CFG` corresponding to the input string(s) | [Adv NLP][052] |
| `nltk.ChartParser(grammar, *args)` | A generic chart parser | [Adv NLP][052] |
| `nltk.ChartParser.parse_all(sent, *args, **kwargs)` | list of tree structure | [Adv NLP][052] |
| `nltk.data.load(resource_url, *args)` | Load a given resource from the NLTK data package | [Adv NLP][052] |
| `nltk.corpus.treebank()` | Reader for corpora that consist of parenthesis-delineated parse trees | [Adv NLP][052] |
| `nltk.corpus.treebank.parsed_sents(fileids=None)` | parsing sentences with parenthesis-delineated parse trees | [Adv NLP][052] |
| `nltk.edit_distance(s1, s2, *args)` | Calculate the Levenshtein edit-distance between two strings | [Asgn02][053] |
| `nltk.jaccard_distance(label1, label2)` | Distance metric comparing set-similarity | [Asgn02][053] |
| `nltk.ngrams(sequence, n, *args)` | Return the ngrams generated from a sequence of items, as an iterator | [Asgn02][053] |


-------------------------------------

<!--
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
[090]: 
[091]: 
[092]: 
[093]: 
[094]: 
[095]: 
[096]: 
[097]: 
[098]: 
[099]: 
-->

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
[052]: ../AppliedDS-UMich/4-TextMining/02-BasicNLP.md#basic-nlp-tasks-with-nltk
[053]: ../AppliedDS-UMich/4-TextMining/02-BasicNLP.md#advanced-nlp-tasks-with-nltk
[054]: ../AppliedDS-UMich/4-TextMining/asgn02.md#edit-distance-and-jaccard-distance-calculation-with-nltk
[055]: ../AppliedDS-UMich/4-TextMining/asgn01.md#solution
[056]: ../AppliedDS-UMich/3-AML/asgn04.md#solution-3
