# Section 3a: Python (Lec 3.1 - Lec 3.3)

## Lec 3.1 Python

### Notes

+ Top Box Office Hit - The highest grossing movie of all time is:
    + Avatar
    + Jaws
    + Titanic
    + Star Wars
    + Star War: The Force Awakens
+ Programming Languages
    + Python is popular both for data science and general software development
    + Mastering the language fundamentals is critical
    + Learn through practice, not by reading or listening
    + Follow along  online via link to demo
+ Demo: [link](./notebooks/lec03.ipynb) & [external link](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.1x+1T2018/courseware/413fff9cb76c471fa0ccb32d7d08ace6/d09d78f97bbe4174a78df2b3846779f4/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.1x%2B1T2018%2Btype%40vertical%2Bblock%40d518a9ada2594f298e3c3972b574a39d)
    + Programming environment
        ```python
        from datascience import *
        import numpy as np

        %matplotlib inline
        import matplotlib.pyplot as plots
        plots.style.use('fivethirtyeight')
        ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/zEN_zpTGjsY)

## Lec 3.2 Names

### Notes

+ Assignment Statements
    + Name = any expression
    + Statements don't have a value; they perform an action
    + An assignment statement changes the meaning of the name to the left of the = symbol
    + The name is bound to a value (not an equation)
+ Demo
    ```python
    hours_per_wk = 24 * 7
    hours_per_week
    hours_per_wk * 60
    seconds_per_year = 356 * 24 * 60 * 60
    seconds_per_hour = 60 * 60
    hours_per_year = 24 * 365
    seconds_per_year = seconds_per_hour * hours_per_year
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/-b1peYEib_A)


## Lec 3.3 Call Expressions

### Notes

+ Anatomy of a Call Expression
    + $f(27)$:
        + `f`: What function to call
        + `27`: Argument to the function
        + read: Call f on 27
    + $max(15, 27)$: two arguments
+ Demo:
    ```python
    abs(5)      # 5
    abs(5-8)    # 3
    max(3, 4)   # 4
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/fLNavUrZXSc)


## Reading and Practice for Section 3a

### Readings

This guide assumes that you have watched the section 3a videos.

This corresponds to textbook sections: 

+ [Chapter 3.1: Expressions Opens in new window](https://www.inferentialthinking.com/chapters/03/1/expressions.html)
+ [Chapter 3.2: Names Opens in new window](https://www.inferentialthinking.com/chapters/03/2/names.html)
+ [Chapter 3.3: Call Expressions Opens in new window](https://www.inferentialthinking.com/chapters/03/3/call-expressions.html)

In section 3a, we learned some of the basics of the programming language Python 3 (the most recent version available).  Python is a popular programming language both for data science & general software development.  

Learning a programming language requires considerable practice in reading and writing code. You will start writing your own Python code very soon, but before that, try checking your understanding of Python with the following practice problems.

### Practices

After executing the following lines of code:
```python
x = 8
y = 2
z = x / y
```

Q1 What will `x + y` evaluate to?

    Ans: 10

Q2. What will `y + z` evaluate to?

    Ans: 6

Consider the following lines of code about volume measurements. If you're not familiar with US Standard Volume measurements, review the website, [US Standard Volume measurements](https://www.mathsisfun.com/measure/us-standard-volume.html)

```python
fluid_ounces_per_cup = 8
cups_per_quart = 4
quarts_per_gallon = 4
```

Q3. Which name represents the number of cups in a quart?

    a. fluid_ounces_per_cup
    b. cup_per_quart
    c. cup
    d. quart
    e. None of the above

    Ans: e

Q4. Suppose you wanted to calculate the number of fluid ounces in a gallon. Select all of the following option(s) that correctly calculates the number of fluid ounces in a gallon.

    a. fluid_ounces_per_cup * fluid_ounces_per_cup
    b. fluid_ounces_per_cup * cups_per_quart * quarts_per_gallon
    c. fluid_ounces_per_cup * cups_per_quart * 2
    d. quarts_per_gallon * 4 * 8
    e. None of the above

    Ans: b, d

Q5. For the following questions, fill in the expected output of the expression.

    ```python
    max(1, 7, -8)
    abs(min(-9, -12))
    ```

    Ans: 7, 12

# Section 3b: Tables (Lec 3.4 - Lec 3.7)

## Lec 3.4 Tables

### Notes

+ Table Structure
    + organize data in tables
    + a table is a sequence of labeled columns
    + data within a column should be the same "type"
    + terms: Label, Column, Row
+ Demo
    ```python
    Table.read_table('flowers.csv')             # load data set
    flowers = Table.read_table('flowers.csv')   # assign name for the table
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/3KcQVMZI-Ro)

## Lec 3.5 Select

### Notes

+ Table Operations
    + `t.select(label)`: constructs a new table with just the specified columns
    + `t.drop(label)`: constructs a new table without the specified columns
    + `t.sort(label)`: constructs a new table, with rows sorted by the specified column

+ `t.select` method
    + Signature: `select(*column_or_columns)`
    + Docstring: Return a table with only the columns in `column_or_columns`
    + Args: 
        + `column_or_columns`: Columns to select from the `Table` as either column labels (`str`) or column indices (`int`).

+ `t.drop` method
    + Signature: `drop(*column_or_columns)`
    + Docstring: Return a Table with only columns other than selected label or labels.
    + Args:
        + `column_or_columns` (string or list of strings): The header names or indices of the columns to be dropped.
    + `column_or_columns`: must be an existing header name, or a valid column index.

+ `t.sort` method:
    + Signature: `sort(column_or_label, descending=False, distinct=False)`
    + Docstring: Return a Table of rows sorted according to the values in a column.
    + Args:
        + `column_or_label`: the column whose values are used for sorting.
        + `descending` (boolean)
        + `distinct` (boolean): omitt repeated values in `column_or_label`

+ Demo
    ```python
    petals = flowers.select('Petals')
    flowers.drop('Color')
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/_ojpEI8z5rI)

## Lec 3.6 Sorting

### Notes

+ Demo
    ```python
    movies = Table.read_table('top_movies_by_title.csv')
    movies.show(3)
    movies.sort('Gross')
    movies.sort('Gross', descending=True)
    sorted_by_gross = movies.sort('Gross', descending=True)
    sorted_by_gross.sort('Studio')
    sorted_by_gross.sort('Studio', distinct=True)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/RGy3BM9nNW8)

## Lec 3.7 Bar Charts

### Notes

+ Demo
    ```python
    top_per_studio = sorted_by_gross.sort('Studio', distinct=True)
    top_per_studio.barh('Studio', 'Gross')

    top_studios = top_per_studio.sort('Gross', descending=True)
    top_studios.barh('Studio', 'Gross')

    just_revenues = top_studios.select('Studio', 'Gross', 'Gross (Adjusted)')
    just_revenues
    just_revenues.barh('Studio')

    sorted_by_year = top_studios.sort('Year')
    revenues_by_year = sorted_by_year.select('Studio', 'Gross', 'Gross (Adjusted)')
    revenues_by_year.barh('Studio')

    sorted_by_year
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/vl8CJVoZpsA)


## Reading and Practice for Section 3b

### Readings

This guide assumes that you have watched the videos for section 3b.

This corresponds to textbook section:

[Chapter 3.4: Introduction to Tables](https://www.inferentialthinking.com/chapters/03/4/intro-to-tables.html)

In section 3b, we used Python to load, manipulate, and visualize data.  We introduced tables, along with operations to select columns and sort rows.  The table operations in this section are just a small subset of the many table operations available to you! The following weeks will introduce the rest.

Below are the table operations that you learned in this section. If t is the name of a table that has a column called label, then

`t.select("label")` constructs a new table with just that column.

`t.sort("label")` constructs a new table with all columns, but with the rows sorted by the values in that column.

Try practicing your understanding of tables and operations with the following practice problems.

### Practice

Suppose you have a table called `students` with columns labeled `name`, `age`, and `email`. Select the best table expression for the following statements.

Q1. A table with just the names and emails of all students.

    a. students.select('name', 'age')
    b. students.select('name').select('email')
    c. students.select(name, email)
    d. students.select('name', 'email')

    Ans: d

Q2. A table ordered by age from oldest to youngest.

    a. students.sort('age')
    b. students.sort('age', descending=False)
    c. students.sort('age', descending=True)
    d. students.sort('age', ascending=False)

    Ans: c

Q3. A table with just the names of the students, ordered from the youngest to oldest.

    a. students.select('name').sort('age')
    b. students.sort('age').select('name')
    c. students.select('name', 'age').sort('age')
    d. students.sort('age').select('name', 'age')

    Ans: b

