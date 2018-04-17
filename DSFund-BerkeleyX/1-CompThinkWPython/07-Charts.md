# Section 7: Charts (Lec 7.1 - 7.8)

## Lec 7.1 Line Graphs

### Notes

+ Visualization Relations: line graphs & scatter plots
+ Demo - Line Graphs

    ```python
    # load data
    full_census_table = Table.read_table("census.csv")
    # taken from Sec 06
    partial = full_census_table.select(['SEX', 'AGE', 4, 9])
    us_pop = partial.relabeled(2, '2010').relabeled(3, '2015')
    ratio = (us_pop.column(3) / us_pop.column(2))
    census = us_pop.with_columns(
            'Change', us_pop.column(3) - us_pop.column(2), 
            'Total Growth', ratio - 1,
            'Annual Growth', ratio ** (1/5) - 1)
    census.set_format([2, 3, 4], NumberFormatter)
    census.set_format([5, 6], PercentFormatter)

    # Line graphs - drop SEX col and only display age 0~100
    by_age = census.where('SEX', 0).drop('SEX').where('AGE', are.between(0, 100))
    by_age.plot(0, 2) 
    by_age.plot(0, 3)
    by_age.select(0, 1, 2).plot(0)
    by_age.select(0, 1, 2).plot(0, overlay=False)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/pcEadlLnFBw)


## Lec 7.2 Example 1

### Notes

+ Question  
    The graph shows the change in the U.S. population by age, between 2010 and 2015.  
    How can you explain the peak at 68 years?
+ Demo

    ```python
    by_age.labels   # tuple of table labels
    by_age.plot(0, 3)
    by_age.sort(3, descending=True)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/5-NEr5Pnybk)


## Lec 7.3 Scatter Plots

### Notes

+ Demo

    ```python
    actors = actors.relabeled(5, '#1 Movie Gross')
    actors.scatter(2, 1)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]


## Lec 7.4 Example 2

### Notes

+ Demo

    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]


## Lec 7.5 How to Choose

### Notes

+ Demo

    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]


## Lec 7.6 Types of Data

### Notes

+ Demo

    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]


## Lec 7.7 Distributions

### Notes

+ Demo

    ```python

    ```

### Video


## Lec 7.8 Example 3

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]


##  Reading and Practice for Section 7

### Reading


### Practice



