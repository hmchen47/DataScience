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


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 3.5 Select

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 3.6 Sorting

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 3.7 Bar Charts

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

