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

+ Interpreting Scatter Plots  
    Interpretations involve all three features of the plot:
    + the individuals corresponding to the point; that is, those on whom the two variables are measured
    + the variable on the horizontal axis
    + the variable on the vertical axis
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
    actors.scatter(2, 3)
    actors.where(2, are.below(10))  # Anthony Daniel
    actors.where(2, are.above(60))  # Samuel Jackson, Morgan Freeman, ...
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/WxrsPBNklks)


## Lec 7.5 How to Choose

### Notes

+ When to use Line Graph and Scatter Plot
    + Line graph (`plot`): one-to-one mapping
    + Scatter plot (`scatter`): one-to-many mapping

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/CQIc1pjkyEM)


## Lec 7.6 Types of Data

### Notes

+ Two Important Types
    + __Numerical__ - Each value is from a numerical scale
        + Ordered, because they are numbers
        + Differences, averages, etc. are meaningful
    + __Categorical__ - Each value is from a fixed inventory
        + May or may not have an ordering
+ Categorical, Not Numerical  
    Just because the assigned values are numbers, that doesn't mean the variable is numerical
    + Census example has numerical; `SEX` code (0, 1, and 2)
    + It doesn't make sense to perform arithmetic on these "numbers", e.g. $1 - 0$ or $(0+1+2)/3$ have no meaning
    + The variable `SEX` is still categorical, even though numbers were used for the categories
+ Terminology
    + __Individuals__: those whose features are recorded 
    + __Variable__: a feature
    + A variable has different __values__
    + Values can be __numerical__ or __categorical__, and of many syb-types within these
    + Each __individual has exactly one value__ of the variable
    + __Distribution__: for each different value of the variable , the frequency of individuals that have that value
+ Demo

    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]


## Lec 7.7 Distributions

### Notes

+ Use of Bar Charts
    + To display the relation between a categorical variable and a numerical variable
    + To display the distribution of a categorical variable
+ Demo

    ```python
    top = Table.read_table('top_movies.csv')
    top10 = top.take(np.arange(10))
    top10.barh(0, 2)

    studios = top.group('Studio')
    sum(studios.column(1))
    studios.barh(0)
    studios.sort(1, descending=True).barh(0)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/ME3LjCrvxik)


## Lec 7.8 Example 3

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]


##  Reading and Practice for Section 7

### Reading


### Practice



