# Section 10a: Groups (Lec 10.1 - Lec 10.3)

## Lec 10.1 One Attribute Group

### Notes

+ Grouping by one Column
    + The `group` method aggregates all rows with the same value a column into a single row in the resulting table.
    + 1st argument: which column to group by
    + 2nd column: (optional) how to combine values
        + `len` - number of grouped values (default)
        + `list` - list of all grouped values
        + `sum` - total of all grouped values
        + ...
+ Demo
    ```python
    all_cones = Table.read_table('cones.csv')
    cones = all_cones.drop('Color').exclude(5)

    cones.group('Flavor')
    cones.group('Flavor', list)
    cones.group('Flavor', len)
    cones.group('Flavor', min)

    min(cones.where('Flavor', 'chocolate').column('Price'))
    cones.group('Flavor', np.average)

    def data_range(x):
        return max(x) - min(x)
    cones.group('Flavor', data_range)

    nba = Table.read_table('nba_salaries.csv').relabeled(3, 'SALARY')
    teams_and_money = nba.select('TEAM', 'SALARY')
    teams_and_money.group('TEAM', sum).sort(1, descending=True)
    nba.group('TEAM', sum)
    position_and_money = nba.select('POSITION', 'SALARY')
    position_and_money.group('POSITION')
    position_and_money.group('POSITION', np.average)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/n0jAwei6zGY){:target="_blank"}


## Lec 10.2 Cross Classification

### Notes

+ Grouping by Multiple Columns
    + The `group` method can also aggregate all rows that share the combination of values in multiple columns
    + 1st argument: a list of which columns to group by
    + 2nd argument: (optional) how to combine values
+ Demo
    ```python
    all_cones.group('Flavor')
    all_cones.group(['Flavor', 'Color'])
    all_cones.group(['Flavor', 'Color'], max)

    nba.drop(0).group(['TEAM', 'POSITION'], np.average)
    nba.drop(0, 2).group('POSITION', np.average)

    full_table = Table.read_table('educ_inc.csv')
    ca_2014 = full_table.where('Year', are.equal_to('1/1/14 0:00')).where('Age', are.not_equal_to('00 to 17')).drop(0).sort('Population Count')
    no_ages = ca_2014.drop(0)
    no_ages.group([0, 1, 2], sum)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/9NGa2MaDPxw){:target="_blank"}


## Lec 10.3 Example 1

### Notes

+ Question   
    Which NBA teams spent the most on their "starters" in 2015-2016?  
    Assume the "starter" for a team & position in the player with the highest salary on that team in that position
    + Ans: 
        ```python
        starter_salaries = nba.drop(0).group(['TEAM', 'POSITION'], max)
        starter_salaries.drop(1).group('TEAM', sum).sort(1, descending=True)
        ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/8MZW99WJcXs){:target="_blank"}


## Reading and Practice for Section 10a

### Reading

This guide assumes that you have watched section 10 (video lecture segments Lec 10.1, Lec 10.2, Lec 10.3) in Courseware.

This corresponds to textbook sections:

+ [Chapter 8.2: Classifying by one Variable](https://www.inferentialthinking.com/chapters/08/2/classifying-by-one-variable.html)
+ [Chapter 8.3: Cross Classifying by More than One Variable](https://www.inferentialthinking.com/chapters/08/3/cross-classifying-by-more-than-one-variable.html)

In section 10, we learned an advanced table method, group. This powerful table operation allow you to analyze and summarize large quantities of data.

Here are the descriptions for group

`tbl.group(column_or_columns)` returns a table with the counts of rows grouped by unique values or combinations of values in a column or columns

`tbl.group(column_or_columns, func)` returns a table that groups rows by unique values or combinations of values in a column or columns. The other values are aggregated by func

Practice your understanding of group with the following practice problems.

### Practice

Data 8X has opened up a marble store where we sell small bags of marbles in groups of different amounts. Each bag contains marbles of one color. Each row is a bag of marbles. Our table marbles is as follows:

marbles

| Color | Amount | Price ($) |
|-------|--------|-----------|
| Red | 4 | 1.30 |
| Green | 6 | 1.20 |
| Blue | 12 | 2.00 |
| Red | 7 | 1.75 |
| Green | 9 | 1.40 |
| Green | 2 | 1.00 |

The marbles table currently has three columns: the color, the amount in each bag, and the price in dollars of each bag of marbles available for purchase. Let's see what we can do with this data!

Q1. Which line of code returns a new table which displays the total number of marbles and the total cost for each unique color?

    a. marbles.group('Color', max)
    b. marbles.group('Color')
    c. marbles.group('Color', sum)
    d. None of the above.

    Ans: c

Now assume we have updated our inventory of marbles, and have begun selling marbles of different shapes as well as colors. Each bag contains marbles of one color and one shape. Each row is a bag of marbles. A table representing our new inventory is as follows:

marbles

| Color | Shape | Amount | Price ($) |
|-------|-------|--------|-----------|
| Red | Round | 4 | 1.30 |
| Green | Rectangular | 6 | 1.20 |
| Blue | Rectangular | 12 | 2.00 |
| Red | Round | 7 | 1.75 |
| Green | Rectangular | 9 | 1.40 |
| Green | Round | 2 | 1.00 |

The following table results from calling marbles.group(['Color', 'Shape'], sum). Fill in the missing entries.

```python
marbles.group(['Color', 'Shape'], sum)
```

| Color | Shape | Amount sum | Price ($) [D] |`
| Red | Round | 11 | 3.05 |`
| Green | Rectangular | [A] | [E] |`
| Blue | Rectangular | [B] | 3.05 |`
| Green | Round | [C] | [F] |`

Q2. What should go in placeholder [A]?

    Ans: 15

Q3. What should go in placeholder [B]?

    Ans: 12

Q4. What should go in placeholder [C]?

    Ans: 2

Q5. What should go in placeholder [D]?

    Ans: Sum

Q6. What should go in placeholder [E]?

    Ans: 2.6

Q7. What should go in placeholder [F]?

    Ans: 1.0


# Section 10b: Pivot (Lec 10.4 - Lec 10.6)

## Lec 10.4 Pivot Tables

### Notes

+ Pivot
    + Cross-classifies according to two categorical variables
    + Produces a grid of counts or aggregated values
    + Two required arguments:
        + 1st: variable that forms column labels of grid
        + 2nd: variable that forms row labels of grid
    + Two optional arguments (include both or neither)
        + `values` = `column_lebel_to_arggregate`
        + `collect` = function _with_which_to_aggregate
+ Demo
    ```python
    all_cones.group(['Flavor', 'Color'])
    all_cones.pivot('Flavor', 'Color')   # pivot table, contingency table

    all_cones.pivot('Color', 'Flavor')
    all_cones.pivot('Color', 'Flavor', values = 'Price', collect = max)

    nba.drop(0).group(['TEAM', 'POSITION'], np.average)
    nba.pivot('POSITION', 'TEAM', 'SALARY', np.average)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/JSgaMnExiCY){:target="_blank"}


## Lec 10.5 Example 2

### Notes

+ Question (by `pivot`)   
    Which NBA teams spent the most on their "starters" in 2015-2016?  
    Assume the "starter" for a team & position in the player with the highest salary on that team in that position
    + Answer
        ```python
        step_1 = nba.pivot('POSITION', 'TEAM', 'SALARY', max)

        totals = step_1.drop(0).apply(sum)
        step_1.with_columns('TOTAL', totals).sort(6, descending=True)
        ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/e2Bs4SfIBUA){:target="_blank"}


## Lec 10.6 Comparing Distributions

### Notes

+ California, 2014  
    Population crosse-classified by age, gender, educational level, and income.  The final column contains the counts in each combination of attributes.

    | Age | Gender | Educational Attainment | Personal Income | Population Count |
    |-----|--------|------------------------|-----------------|----------------|
    |18 to 64 | Female | No high school diploma | H: 75,000 and over | 2058 |
    | 65 to 80+ | Male | No high school diploma | H: 75,000 and over | 2153 |
    | 65 to 80+ | Female | No high school diploma | G: 50,000 and 74,999 | 4666 |
    | ... | ... | ... | ... | ... |
    + Goal: Compare distributions of personal income at different educational levels.
    + Ans:
        ```python
        educ_income = ca_2014.pivot(2, 3, 4, sum)

        def percent(x):
            """Convert an array of counts into percents"""
            return np.round((x / sum(x)) * 100, 2)
        
        distributions = educ_income.select(0).with_columns(
            'Bachelors or Higher', percent(educ_income.column(1)),
            'High School', percent(educ_income.column(2))
        )
        
        sum(distributions.column(1))    # verify
        distributions.barh(0)
        ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/YqN8OYt8Upw){:target="_blank"}


## Reading and Practice for Section 10b

### Reading

This guide assumes that you have watched section 10 (video lecture segments Lec 10.4, Lec 10.5, Lec 10.6) in Courseware.

This corresponds to textbook section:

[Chapter 8.3: Cross-Classifying by more than One Variable](https://www.inferentialthinking.com/chapters/08/3/cross-classifying-by-more-than-one-variable.html)

In section 10b, we learned another complex table method, pivot.

Here are the descriptions for pivot.

`tbl.pivot(col1, col2)` returns a pivot table where each unique value in `col1` has its own column and each unique value in `col2` has its own row. Each cell of the pivot table contains the number of rows that have the combination of values in `col1` and `col2`

`tbl.pivot(col1, col2, values, collect)` returns a pivot table where each unique value in `col1` has its own column and each unique value in `col2` has its own row. The cells of the grid contain row counts (two arguments) or the `values` from a third column, aggregated by the `collect` function

Practice your understanding of pivot with the following practice problem.

### Practice

Back to our Data 8X marble store where we sell small bags of marbles. Each bag contains marbles of one color and one shape. Each row is a bag of marbles. The marbles table currently has four columns: the color, the shape, the amount in each bag, and the price in dollars of each bag of marbles available for purchase. Let's see what we can do with this data! This is the table marbles.

marbles

| Color | Shape | Amount | Price ($) |
|-------|-------|--------|-----------|
| Red | Round | 4 | 1.30 |
| Green | Rectangular | 6 | 1.20 |
| Blue | Rectangular | 12 | 2.00 |
| Red | Round | 7 | 1.75 |
| Green | Rectangular | 9 | 1.40 |
| Green | Round | 2 | 1.00 |

The following table results from calling `marbles.pivot(’Shape’, ’Color’, ’Amount’, max)`. Fill in the missing entries.

```python
marbles.pivot(’Shape’, ’Color’, ’Amount’, max)
```

| Color | Rectangular | Round |
|-------|-------------|-------|
| Blue | [B] | 0 |
| [A] | [C] | [D] |
| Red | 0 | 7 |

Q1. What should go in placeholder [A]?

    Ans: Green

Q2. What should go in placeholder [B]?

    Ans: 12

Q3. What should go in placeholder [C]?

    Ans: 9

Q4. What should go in placeholder [D]?

    Ans: 2



