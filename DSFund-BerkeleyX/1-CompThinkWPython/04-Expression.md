# Section 4: Expressions (Lec 4.1 - Lec 4.5)

## Lec 4.1 Arithmetic

### Notes

+ Arithmetic Operators
    | Operation | Operator | Example | Value |
    |-----------|:----------:|---------|-------|
    | Addition | `+` | `2 + 3` | 5 |
    | Subtraction | `-` | `2 - 3` | -1 |
    | Multiplication | `*` | `2 * 3` | 6 |
    | Division | `/` | `8 / 3` | 2.6666666666666665 |
    | Remainder | `%` | `8 % 3` | 2 |
    | Exponentiation | `**` | `2 ** 0.5` | 1.4142135623730951 |
+ Ints and Floats
    + Python has two real number types:
        + `int`: an integer of any size
        + `float`: a number with a fractional aprt that may be wero
    + An `int` never has a decimal point; a `float` alwaus does
    + A `float` might be printed using scientific notation.
    + Three limitations of float values:
        + limited size (but the limit is huge)
        + limited precision of 15-16 decimal places
        + after arithmetic, the final few decimal places can be wrong
+ Demo
    + Programming environment
        ```python
        from datascience import *
        import numpy as np

        %matplotlib inline
        import matplotlib.pyplot as plots
        plots.style.use('fivethirtyeight')
        ```
    + precision: 15 or 16 significant digits
        ```python
        (2 ** 0.5) * (2 ** 0.5)         # != 2
        (2 ** 0.5) * (2 ** 0.5) - 2     # != 0
        ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/hWY_LGmzwkU)

## Lec 4.2 Arithmetic Question

### Notes

+ Rank the results of the following expression in order from least to greatest

    a. 3 * 10 ** 10  
    b. 10 * 3 ** 10  
    c. (10 * 3) ** 10  
    d. 10 / 3 / 10  
    e. 10 / (3 / 10)  

    Ans: c (4904900000000) > a (30000000000) > b (590490) > e (0.33333333333333337) > d (33.333333333333336)

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/F3KVaWfrwn0)

## Lec 4.3 Exponential Growth

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 4.4 Arrays

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 4.5 Columns

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]
