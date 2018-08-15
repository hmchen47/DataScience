# Section 1: Center and Spread (Lec 1.1 - Lec 1.4)

+ [Launching Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/7bba8d29a20946e5be64e508fd3481b2/15f6f5cf64944c379af5a6b1f12f8af4/)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec1.ipynb)
+ [Local Notebook](./notebook/lec1.ipynb)
+ [Local Python](./notebook/lec1.py)

## Lec 1.1 Introduction

### Notes

+ Prediction
  + One of the major responsibility of data science
  + To predict the value of one or more variables given the values of other related variables
  + Distributions: empirical, probability, data, etc.

+ Goals
  + Quantify natural concepts like "center" and "variability"
  + Examine bell shaped distributions
  + Understand why many of the empirical distributions that we have generated are bell shaped

+ The Average (or Mean) <br/>
    Data: 2, 3, 3, 9 -> Average = $(2+3+3+9)/4 = 4.25$
    + Need not be a value in the collection
    + Need not be an integer even if the data are integers
    + Somewhere between min and max, but not necessarily halfway in between
    + Same units as the data
    + Smoothing operator: collect all the contributions in one big pot, then split evenly

+ Relation to the Histogram
  + The average of a list depends only on the __proportions__ in which the distinct values appear, not on the number of entries in the list
  + The average is the __center of gravity__ of the histogram
  + The point on the horizontal axis where the histogram balances

+ Demo
  ```python
  values = make_array(2, 3, 3, 9)
  sum(values) / len(values), np.average(values), np.mean(values)
  # (4.25, 4.25, 4.25

  (2 + 3 + 3 + 9) / 4                 # 4.25
  2 * (1/4) + 3 * (2/4) + 9 * (1/4)   # 4.25
  2 * 0.25 + 3 * 0.5 + 9 * 0.25       # 4.25

  values_table = Table().with_columns('Value', values)
  values_table                        # Value: 2, 3, 3, 9

  bins_for_display = np.arange(0.5, 10.6, 1)

  values_table.hist(bins = bins_for_display, ec = 'w')

  2 * np.ones(10)   # array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

  twos = 2 * np.ones(10)
  threes = 3 * np.ones(20)
  nines = 9 * np.ones(10)

  new_values = np.append(np.append(twos, threes), nines)

  len(new_values)     # 40

  new_values_table = Table().with_column('Value', new_values)
  new_values_table.hist(bins = bins_for_display)

  np.average(new_values), np.average(values)
  # (4.25, 0.3909710391822828)
  ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000300_DTH.mp4" alt="Lecture 1.1 Introduction" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 1.2 Average and Median

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000100_DTH.mp4" alt="Lec 1.2 Average and Median" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 1.3 Standard Deviation

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000200_DTH.mp4" alt="Lec 1.3 Standard Deviation" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 1.4 Chebyshev's Bounds

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000400_DTH.mp4" alt="Lec 1.4 Chebyshev's Bounds" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading



### Practice



