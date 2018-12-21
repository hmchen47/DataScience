# Assignment 1

## [Extra resources on Regex](https://www.coursera.org/learn/python-text-mining/discussions/weeks/1/threads/HyeJPFWWEei_yQqv26nkEA)

### YW Init

If you need more complicated expressions of regex for the assignment or just in general want to know more, below are some useful resources. Enjoy!

[Regex expression reference and tester](https://regexr.com)

[Regex cheatsheet](http://www.rexegg.com/regex-quickstart.html)

Another [regex tester](https://www.debuggex.com/) (I highly recommend this one as it has relationship graph to demonstrate the relationship of different groups and show you which group is working and which is not)

[Regex tutorial](https://www.guru99.com/python-regular-expressions-complete-tutorial.html) (if you want to know more, I strongly recommend reading it through, you will know more about the difference among search, match, and findall functions and when to use each of them)

### Jo Are By Amended

For the curious-minded, let me add a few references to how regular expressions work behind the scenes (it helped me deepen my understanding of regular expressions, and I wish someone would have told me sooner).

https://en.wikipedia.org/wiki/Regular_expression#Implementations_and_running_times

https://en.wikipedia.org/wiki/Finite-state_machine

https://en.wikipedia.org/wiki/Deterministic_finite_automaton

https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton

https://www.youtube.com/results?search_query=regular+expression+finite+automata



## [Something wrong with the grader](https://www.coursera.org/learn/python-text-mining/discussions/weeks/1/threads/AoLX8rSbEeiqnRI0WnAb-A)

### Uwe F Mayer Replay 1

First up, your code is not good, sorry to say. You are getting zero because the autograder cannot run your code. That's what the message "Unable to load student data file" means. Here are the most common causes for that:

1. Notebook imports matplotlib or has anything else related to matplotlib such as %matplotlib notebook
2. Notebook plots
3. Notebook prints (using print)
4. Notebook has a syntax error
5. Notebook imports a library not available to the grader
6. Notebook uses an option / parameter to a more recent version of a library that the autograder's older library version does not support

The autograder works by first running the entire notebook, and only afterwards calls the assigned functions to check if the expected answer is produced. If it cannot run the entire notebook you'll get the message about unable to load student data file.

The steps to fix are to make sure none of the above points happen. You may want to use Kernel>Restart and Run all to check for 4. You can upload your notebook to the online system and run it there to check for 5. The first 3 you check by simply looking at your notebook and making sure you don't do any of that (after all, where would the autograder display the output?).


### Uwe F Mayer Replay 2

Yeifer, first run the notebook on the Coursera online system. Then add a cell at the end that calls your assigned function. Run the entire notebook again via Kernel > Restart and Run All. This simulates what the autograder does, it first runs your entire notebook, and only then calls your assigned function. In that cell you added at the end, does it still produce the correct answer?

Another issue I didn't list above is that some learners change the function definition. The assignment asks to implement the function date_sorter(), however quite a few learner don't do that and instead implement date_sorter(df), which as far as the grader is concerned is something entirely different.

Does any of that help?

### Uwe F Mayer Replay 3

We don't know what you are submitting, but I am guessing you misunderstood the instructions. The correct answer should look like:

```python
0 9
1 84
2 2
3 53
4 28
5 474
6 153
7 13
8 129
9 98
10 111
...
490 152
491 235
492 464
493 253
494 231
495 427
496 141
497 186
498 161
499 413
Length: 500, dtype: int32
```

I have seen it repeatedly that learners switched index and value in their submission, or sorted by the wrong thing. Please check that your submission matched (or mostly matches) the above. You don't have to get exactly what I got to get 100%, there's a bit of a fudge factor built into the grader.

Please update this thread with what you find out.

Anurag, the first two lines in my posted answer say that the records with the earliest date in the dataset is record 9, then record with the second-earliest date in the dataset is record 84. Your post relates to the first two records of the dataset, which do not contain the two earliest dates in the dataset, and hence will not be the first two indices of the answer. It appears you misunderstood what the answer is supposed to contain: Extract all the dates, then sort the records by those dates, and return the resorted original index. As it turns out record #9 (that's the 10th record) has the earliest date, and so on.


### Uwe F Mayer Replay 4

Miroslav, start with a brand new unedited assignment notebook, and submit it. It should load, but of course you won't get any points. Then start adding your code to it, piece by piece. Submit after each piece (saving your notebook of course before submitting and always doing a Kernel>Restart and Run All to capture errors in the notebook). At some point or the other the submission will fail to load. Look at that code and chances are you will find an overlooked print or plot or import statement. Should you not have such a statement, then rewrite that code until the grader can run it.

Of course if you work offline you should do the same thing with the online system. Your libraries might differ, and some libraries now have migrated far enough away so that code that runs locally no longer runs on the online system.

Please let us know what you find.


### Uwe F Mayer Replay 5

Sahil, do you get "cannot load student data file" as part of the grader's messages? If so there's something in your notebook the autograder cannot handle, and it never even gets to your answer. As a recap, submissions must:

+ not use print
+ not use matplotlib (not even import)
+ have no syntax or other error
+ after a kernel restart be able to be run all the way through and then in function calls afterwards (added by the autograder) produce desired answers
+ not expect the autograder to run the same new library versions as learners run offline (the autograder is using the same ones as the online Coursera notebook, and they are about 1 year old)

### Uwe F Mayer Replay 6

Well, you asked for it: You are wrong :-).

There are indeed identical dates, but the scoring accounts for that. It computes a proximity score of your returned record order to the one the autograder has, and if it's close enough you'll get 100%. You might want to write your parsed dates along with the original records into a file and open that with Excel. The data is tiny (500 records) so you can easily check what you've missed. Quite often incorrectly parsed dates have the day as 1. Another common issue is to make sure you parse dates in the US fashion, that is month is first in the raw data. Finally make sure you sort correctly, I'd suggest to transform all dates into a single YYYYMMDD string and sort by that.


### Uwe F Mayer Replay 7

or good measure I resubmitted with a tie-breaking sort. Specifically if I had multiple records with the same date extracted I sorted each such group of records with identical date by the index in increasing order for one submission, and by decreasing index order in the other submission. That resulted in 92 differences. Of course most differences are simply by one position or so. Both submissions got 100% from Grader version 2017.10.16a. So as said, as long as the extracted dates are correct and the sorting is done correctly the sort order of records with identical dates does not matter.


## [Notice for Auditing Learners: Assignment Submission](https://www.coursera.org/learn/python-text-mining/discussions/weeks/1/threads/kx2S5rKrEeefpw51rAEiYg)

Please note: only verified learners can submit assignments. If you are auditing this course, you will be able to go through the quizzes or assignments, but you will not be able to submit your assignment for a grade. If you wish to have your assignments graded and receive a course certificate, we encourage you to upgrade to the Certified Learner track for this course. Coursera has provided [information about purchasing a certificate](https://learner.coursera.help/hc/en-us/articles/208280146-Pay-for-a-course-or-Specialization), and you can also get help from the [Coursera Help Center](https://learner.coursera.help/hc/en-us).


## Solution

```python
def date_sorter():
    import re
    
    df_dates = pd.DataFrame([])

    # Your code here
    pat1 = re.compile(r"((?P<mon>(?:[0-9]|[0-1][0-9]))[\/-](?P<day>(?:[1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1]))[\/-](?P<year>(?:(?:(?:19|20)[0-9][0-9]|[0-9][0-9]))))")
    pat2 = re.compile(r"((?P<day>(?:[1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])) (?P<mon>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z|\s|.|,]+(?P<year>(?:(?:(?:19|20)[0-9][0-9]|[0-9][0-9]))))")
    pat3 = re.compile(r"((?P<mon>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z-\s.,]+(?P<day>(?:[1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1]))[,|.|\s|t|h|n|d|s|t|-]+(?P<year>(?:(?:(?:19|20)[0-9][0-9]|[0-9][0-9]))))")
    pat4 = re.compile(r"((?P<mon>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z|\s|.|,]+(?P<year>(?:(?:19|20)[0-9][0-9])))")
    pat5 = re.compile(r"((?P<mon>(?:[1-9]|[0-1][0-9]))[\/](?P<year>(?:(?:19|20)[0-9][0-9])))")
    pat6 = re.compile(r"((?P<year>(?:(?:19|20)[0-9][0-9])))")

    df_dates1 = df.str.extractall(pat1)
    df_dates2 = df.str.extractall(pat2)
    df_dates = pd.concat([df_dates1, df_dates2])

    df_dates3 = df.str.extractall(pat3)
    df_dates = pd.concat([df_dates, df_dates3])

    df_dates4 = df.str.extractall(pat4)
    df_dates4['day'] = '1'
    df_dates = pd.concat([df_dates, df_dates4])
    df_dates = df_dates[~df_dates.index.duplicated(keep='first')]

    df_dates5 = df.str.extractall(pat5)
    df_dates5['day'] = '1'
    df_dates = pd.concat([df_dates, df_dates5])
    df_dates = df_dates[~df_dates.index.duplicated(keep='first')]

    df_dates6 = df.str.extractall(pat6)
    df_dates6['day'] = '1'
    df_dates6['mon'] = '1'
    df_dates = pd.concat([df_dates, df_dates6])
    df_dates = df_dates[~df_dates.index.duplicated(keep='first')]

    df_dates['day'] = df_dates.day.astype(int)

    import calendar
    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}

    df_dates['mon'] = df_dates['mon'].apply(lambda x: abbr_to_num[x] if x in list(calendar.month_abbr) else int(x))
    df_dates['year'] = df_dates.year.astype(int)
    df_dates['year'] = df_dates.year.apply(lambda x: x + 1900 if x < 100 else x)

    import datetime as dt
    df_dates['date'] = pd.to_datetime((df_dates['year']*10000+df_dates['mon']*100+df_dates['day']).apply(str))

    df_dates = df_dates.sort_values(by='date')

    index = df_dates.index.labels[0]

    return pd.Series(index) # Your answer here

date_sorter()
```

+ `df.str.extractall` method
    + Signature: `df.str.extractall(pat, flags=0)`
    + Docstring: For each subject string in the Series, extract groups from all matches of regular expression pat. When each subject string in the Series has exactly one match, extractall(pat).xs(0, level='match') is the same as extract(pat).
    + Parameters
        + `pat` (string): Regular expression pattern with capturing groups
        + `flags` (int, default 0 (no flags)): re module flags, e.g. `re.IGNORECASE`
    + Returns: A DataFrame with one row for each match, and one column for each group. Its rows have a MultiIndex with first levels that come from the subject Series. The last level is named 'match' and indicates the order in the subject. Any capture group names in regular expression pat will be used for column names; otherwise capture group numbers will be used.

+ `df.str.extract` method
    + Signature: `df.str.extract(pat, flags=0, expand=None)`
    + Docstring: For each subject string in the Series, extract groups from the first match of regular expression pat.
    + Parameters
        + `pat` (string): Regular expression pattern with capturing groups
        + `flags` (int, default 0 (no flags)):  re module flags, e.g. re.IGNORECASE
        + `expand` (bool, default False): 
            + If True, return DataFrame.
            + If False, return Series/Index/DataFrame.
    + Returns: DataFrame with one row for each subject string, and one column for each group. Any capture group names in regular expression pat will be used for column names; otherwise capture group numbers will be used. The dtype of each result column is always object, even when no match is found. If expand=False and pat has only one capture group, then return a Series (if subject is a Series) or Index (if subject is an Index).

+ `df.str.split` method
    + Signature: `df.str.split(pat=None, n=-1, expand=False)`
    + Docstring: Split each string (a la re.split) in the Series/Index by given pattern, propagating NA values. Equivalent to `str.split`.
    + Parameters
        + `pat` (string, default None): String or regular expression to split on. If None, splits on whitespace
        + `n` (int, default -1 (all)): None, 0 and -1 will be interpreted as return all splits
        + `expand` (bool, default False): 
            + If True, return DataFrame/MultiIndex expanding dimensionality.
            + If False, return Series/Index
    + Return: `return_type`: deprecated, use `expand`

+ `df.str.find` method
    + Signature: `df.str.find(sub, start=0, end=None)`
    + Docstring: Return lowest indexes in each strings in the Series/Index where the substring is fully contained between [start:end]. Return -1 on failure. Equivalent to standard `str.find`.
    + Parameters
        + `sub` (str): Substring being searched
        + `start` (int): Left edge index
        + `end` (int): Right edge index
    + Returns: `found`: Series/Index of integer values

+ `df.str.findall` method
    + Signature: `df.str.findall(pat, flags=0, **kwargs)`
    + Docstring: Find all occurrences of pattern or regular expression in the Series/Index. Equivalent to `re.findall`.
    + Parameters
        + `pat` (string): Pattern or regular expression
        + `flags` (int, default 0 (no flags)): re module flags, e.g. re.IGNORECASE
    + Returns: `matches`: Series/Index of lists

+ `df.index.duplicated` method
    + Signature: `df.index.duplicated(keep='first')`
    + Docstring: Return boolean np.ndarray denoting duplicate values
    + Parameters
        + `keep` ({'first', 'last', False}, default 'first'): 
            + `first` : Mark duplicates as `True` except for the first occurrence.
            + `last` : Mark duplicates as `True` except for the last occurrence.
            + False : Mark all duplicates as `True`.
    + Returns: `duplicated` (np.ndarray)

+ `df.sort_values` method
    + Signature: `df.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')`
    + Docstring: Sort by the values along either axis
    + Parameters
        + `axis` ({0, 'index'}, default 0): Axis to direct sorting
        + `ascending` (bool or list of bool, default True): Sort ascending vs. descending. Specify list for multiple sort orders.  If this is a list of bools, must match the length of the by.
        + `inplace` (bool, default False): if True, perform operation in-place
        + `kind` ({'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'): Choice of sorting algorithm. See also ndarray.np.sort for more information.  `mergesort` is the only stable algorithm. For DataFrames, this option is only applied when sorting on a single column or label.
        + `na_position` ({'first', 'last'}, default 'last'): `first` puts NaNs at the beginning, `last` puts NaNs at the end
    + Returns: `sorted_obj` (Series)

+ `pd.concat` method
    + Signature: `pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)`
    + Docstring: Concatenate pandas objects along a particular axis with optional set logic along the other axes.
    + Notes:Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
    + Parameters
        + `objs` (a sequence or mapping of Series, DataFrame, or Panel objects): If a dict is passed, the sorted keys will be used as the `keys` argument, unless it is passed, in which case the values will be selected (see below). Any None objects will be dropped silently unless they are all None in which case a ValueError will be raised
        + `axis` ({0/'index', 1/'columns'}, default 0):  The axis to concatenate along
        + `join` ({'inner', 'outer'}, default 'outer'): How to handle indexes on other axis(es)
        + `join_axes` (list of Index objects): Specific indexes to use for the other $n - 1$ axes instead of performing inner/outer set logic
        + `ignore_index` (boolean, default False): If True, do not use the index values along the concatenation axis. The resulting axis will be labeled 0, ..., n - 1. This is useful if you are concatenating objects where the concatenation axis does not have meaningful indexing information. Note the index values on the other axes are still respected in the join.
        + `keys` (sequence, default None): If multiple levels passed, should contain tuples. Construct hierarchical index using the passed keys as the outermost level
        + `levels` (list of sequences, default None): Specific levels (unique values) to use for constructing a MultiIndex. Otherwise they will be inferred from the keys
        + `names` (list, default None): Names for the levels in the resulting hierarchical index
        + `verify_integrity` (boolean, default False): Check whether the new concatenated axis contains duplicates. This can be very expensive relative to the actual data concatenation
        + `copy` (boolean, default True): If False, do not copy data unnecessarily
    + Returns: `concatenated` (type of objects)





