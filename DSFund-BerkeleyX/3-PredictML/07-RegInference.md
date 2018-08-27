# Section 7: Regression Inference (Lec 7.1 - Lec 7.3)

+ [Launching Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/9e3318ad6da44461990e1d4e3a64986f/4f90f7d56ed74420b5b641cdb0fedc65/?child=first)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec7.ipynb#)
+ [Local notebook](./notebook/lec7.ipynb)
+ [Local Python Code](./notebook/lec7.py)

+ Common Function for Demo
    ```python
    def standard_units(any_numbers):
        """Convert any array of numbers to standard units."""
        return (any_numbers - np.average(any_numbers)) / np.std(any_numbers)

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

    def fit(t, x, y):
        """The fitted values along the regression line."""
        a = slope(t, x, y)
        b = intercept(t, x, y)
        return a * t.column(x) + b

    def plot_residuals(t, x, y):
        """Plot a scatter diagram and residuals."""
        t.scatter(x, y, fit_line=True)
        actual = t.column(y)
        fitted = fit(t, x, y)
        residuals = actual - fitted
        print('r:', correlation(t, x, y))
        print('RMSE:', np.mean(residuals**2)**0.5)
        t.select(x).with_column('Residual', residuals).scatter(0, 1)
    ```

## Lec 7.1 Regression Model

### Note


+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V002600_DTH.mp4" alt="Lec 7.1 Regression Model" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Lec 7.2 Prediction Variability

### Note


+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V002700_DTH.mp4" alt="Lec 7.2 Prediction Variability" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Lec 7.3 The True Slope

### Note


+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V002500_DTH.mp4" alt="Lec 7.3 The True Slope" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Reading and Practice

### Reading



### Practice


