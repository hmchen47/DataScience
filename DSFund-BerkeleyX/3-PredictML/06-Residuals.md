# Section 6: Residuals (Lec 6.1 - Lec 6.4)

+ [Launching Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/9e3318ad6da44461990e1d4e3a64986f/6f26c6f9a58842ab81654f2c01994558/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%40c8a5435d8a5d45f39b0a327e1d239d50)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec6.ipynb#)
+ [Local notebook](./notebook/lec6.ipynb)
+ [Local Python Code](./notebook/lec6.py)

+ Common Functions for Demo
    ```python
    def standard_units(any_numbers):
        """Convert any array of numbers to standard units."""
        return (any_numbers - np.average(any_numbers)) / np.std(any_numbers)

    # Below, t is a table; x and y are column indices or labels.

    def correlation(t, x, y):
        """Return the correlation coefficient (r) of two variables."""
        return np.mean(standard_units(t.column(x)) * standard_units(t.column(y)))

    def slope(t, x, y):
        """The slope of the regression line (original units)."""
        r = correlation(t, x, y)
        return r * np.std(t.column(y)) / np.std(t.column(x))

    def intercept(t, x, y):
        """The intercept of the regression line (original units)."""
        return np.mean(t.column(y)) - slope(t, x, y) * np.mean(t.column(x))

    def fitted_values(t, x, y):
        """The fitted values along the regression line."""
        a = slope(t, x, y)
        b = intercept(t, x, y)
        return a * t.column(x) + b
    ```

## Lec 6.1 Introduction

### Note

+ Residuals
    + Error in regression estimate
    + One residual corresponding to each point (x, y)
    + residual <br/>
        = <span style="color:blue" b> observed y - regression estimate of y </span><br/>
        = observed y - height of regression line at x <br/>
        = vertical distance between the point and the best line

+ Demo
    ```python
    # ## Residuals
    galton = Table.read_table('galton.csv')

    heights = Table().with_columns(
        'MidParent', galton.column('midparentHeight'),
        'Child', galton.column('childHeight')
    )
    # MidParent     Child
    # 75.43         73.2
    # 75.43         69.2
    # 75.43         69
    # ... (rows omitted)

    heights = heights.with_columns('Fitted', fitted_values(heights, 0, 1))
    # MidParent     Child   Fitted
    # 75.43         73.2    70.7124
    # 75.43         69.2    70.7124
    # 75.43         69      70.7124
    # ... (rows omitted)

    heights.scatter(0)

    def residuals(t, x, y):
        return t.column(y) - fitted_values(t, x, y)

    heights = heights.with_columns('Residual', residuals(heights, 'MidParent', 'Child'))
    # MidParent     Child   Fitted      Residual
    # 75.43         73.2    70.7124     2.48763
    # 75.43         69.2    70.7124     -1.51237
    # 75.43         69      70.7124     -1.71237
    # ... (rows omitted)

    heights.scatter(0)

    def plot_residuals(t, x, y):
        with_residuals = t.with_columns(
            'Fitted', fitted_values(t, x, y),
            'Residual', residuals(t, x, y)
        )
        with_residuals.select(x, y, 'Fitted').scatter(0)
        with_residuals.scatter(x, 'Residual')

    plot_residuals(heights, 'MidParent', 'Child')
    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V002100_DTH.mp4" alt="Lec 6.1 Introduction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.2 Regression Diagnostics

### Note


+ Demo
    ```python

    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V002200_DTH.mp4" alt="Lec 6.2 Regression Diagnostics" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.3 Properties of Residuals

### Note


+ Demo
    ```python

    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V002300_DTH.mp4" alt="Lec 6.3 Properties of Residuals" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.4 Discussion Question

### Note


+ Demo
    ```python

    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V002000_DTH.mp4" alt="Lec 6.4 Discussion Question" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading



### Practice

