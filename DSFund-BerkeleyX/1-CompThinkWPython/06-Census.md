# Section 6: Census (Lec 6.1 - Lec 6.4)

## Lec 6.1 Census

### Notes

+ The Decennial Census
    + Every ten years, the Census Bureau counts how many people there in the U.S.
    + In between censuses, the Bureau s=estimates how many people there are each year
    + Article 1, Section 2 of the Constitution:  
        "Representatives and direct Taxes shall be apportioned among the several States ... according to their respective Numbers ..."
+ Public Census Data
    + Analyzing census data can lead to the discovery of interesting features and trends in the population.
    + [Data](http://www2.census.org/programs-surveys)
+ Demo

    ```python
    data = 'http://www2.census.gov/programs-surveys/popest/datasets/2010-2015/national/asrh/nc-est2015-agesex-res.csv'

    partial = full.select('SEX', 'AGE', 4, 9)
    us_pop = partial.relabeled(2, '2010').relabeled(3, '2015')
    us_pop.set_format([2, 3], NumberFormatter)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/b29OrPn5ofw)

## Lec 6.2 Column Arithmetic

### Notes

+ Demo
    ```python
    us_pop.column('2015') - us_pop.column('2010')

    change = us_pop.column('2015') - us_pop.column('2010')
    census = us_pop.with_columns(
        'Change', change,
        'Total Growth', change / us_pop.column('2010')
    )
    census.set_format('Change', NumberFormatter)
    census.set_format('Total Growth', PercentFormatter)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/4SLry5hYcmE)

## Lec 6.3 Accessing Values

### Notes

+ Census Table Description
    + Values are column-dependent interpretations
    + The `SEX` column: 1 = Male, 2 = Female
    + The `POPESTIMATE2010` column: 7/1/2010 estimate
+ In this tab;le, some rows are sums of other rows
    + The `SEX` column: 0 is $Total$ (of Male + Femal)
    + The `AGE`   column: 999 is $Total$ of all ages
+ Numeric codes are often used for storage efficiency
+ Demo

    ```python
    census.sort('Change', descending=True)
    sum(census.column('2010')) 
    everyone = census.sort('Change', descending=True).row(0)    # 1st row
    five_year_growth = everyone.item(5)

    census = census.with_column(    # must reassign to original table
        'Annual Growth', (census.column(5) + 1) ** (1/5) - 1
    )
    census.set_format(6, PercentFormatter)
    census.sort('Change', descending=True)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/EOrAk4r9kck)

## Lec 6.4 Males and Females

### Notes

+ Demo
    ```python
    us_pop_2015 = us_pop.drop('2010').where('AGE', are.below(999)).where('SEX', are.above(0))
    sum(us_pop_2015.column(2))

    males = us_pop_2015.where('SEX', 1).column('2015')
    by_sex = us_pop_2015.where('SEX', 2).drop('SEX').relabeled('2015', 'Females').with_column('Males', males)
    by_sex.set_format('Males', NumberFormatter)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/SAJavz58uHk)

## Reading and Practice for Section 6

### Reading

This guide assumes that you have watched section 6 (video lecture segments Lec 6.1, Lec 6.2, Lec 6.3, Lec 6.4) in Courseware.

This corresponds to textbook sections:  

+ [Chapter 6.3: Example Trends in the U.S. Population](https://www.inferentialthinking.com/chapters/06/3/example-trends-in-the-population-of-the-united-states.html)
+ [Chapter 6.4: Trends in Gender Ratios](https://www.inferentialthinking.com/chapters/06/4/example-gender-ratio-in-the-us-population.html)

In Section 6, we applied table methods to manipulate US Census data. Proficiency in manipulating tables requires knowing table methods and how they combine in sequence, as well as comfort with array arithmetic.  A list of all the existing table methods and how to use them can be found in the [Tables Documentation](http://data8.org/datascience/tables.html). Knowing your table methods will be extremely helpful for lab this week!

Test your understanding with these practice questions.

### Practice

Match each of the following table commands with a description of its effect. Each answer may only be used once. Assume that t is a table that has already been loaded.

A. `t.select(...)`
B. `t.sort(...)`
C. `Table.read_table(...)`
D. `t.column(...).item(...)`
E. `t.barh(...)`
F. `t.column(...)`
G. `t.show(...)` 

Q1. Displays the first few rows of `t`.

    Ans: G

Q2. Reads a table from a file or webpage.

    Ans: C

Q3. Returns a table with some of the columns of `t`.

    Ans: A [x]

Q4. Returns a table with the same contents as `t`, but with the rows ordered by the values in a certain column.

    Ans: B

Q5. Returns an array containing the values in a column of `t`.

    Ans: F [x]

Q6. Displays a bar chart based on the columns of `t`.

    Ans: E

Q7. Returns a single value from a cell in `t`.

    Ans: D


