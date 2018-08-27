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

+ Residual Plot: A scatter diagram of residuals
    + Should look like an unassociated blob for linear relations
    + But will show patterns for non-linear relations
    + Used to check whether linear regression is appropriate

+ Demo
    ```python
    # ### Nonlinearity
    dugong = Table.read_table('dugong.csv')
    # Length  Age
    # 1.8     1
    # 1.85    1.5
    # 1.87    1.5
    # ... (rows omitted)

    correlation(dugong, 'Length', 'Age')        # 0.8296474554905714

    plot_residuals(dugong, 'Length', 'Age')

    height_vs_average_weight = Table.read_table('us_women.csv')
    # height  ave weight
    # 58      115
    # 59      117
    # 60      120
    # ... (rows omitted)

    height_vs_average_weight

    correlation(height_vs_average_weight, 0, 1)     # 0.9954947677842161

    plot_residuals(height_vs_average_weight, 0, 1)
    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V002200_DTH.mp4" alt="Lec 6.2 Regression Diagnostics" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.3 Properties of Residuals

### Note

+ Residual Variance
    + The mean of residuals is always $0$, regardless of the original data
    + Variance of standard deviation squared: mean squared deviation
    + $(\text{Variance of residuals}) / (\text{Variance of y}) = 1 - r^2$
    + $(\text{Variance of fitted values}) / (\text{Variance of y}) = r^2$
    + $\text{Variance of y} = (\text{Variance of fitted values}) + (\text{Variance of residuals})$

+ No matter what the shape of the scatter plot, the SD of the fitted values is a fraction of the SD of the observed values of $y$. The fraction is |r|.

    $$\frac{\text{SD of fitted values}}{\text{SD of }y} = |r| $$ 

    That is, $\text{SD of fitted values} = |r| \cdot \text{SD of }y$$

+ Average of Residuals
    + The average of the residuals is always $0$
    + No matter what the scatter looks like
    + Just as the average of the deviations from mean is always $0$
    + No matter what the data look like

+ Correlation, Revisited
    + “The correlation measures how clustered the points are about a straight line.”
    + We can now quantify this statement.

+ SD of Fitted Values

    $$\frac{\text{SD of fitted values}}{\text{SD of } y } = |r|$$

    $$\text{SD of fitted values} = |r| \cdot (\text{SD of } y) $$

+ Variance of Fitted Values
    + Variance = Square of the SD = Mean Square of the Deviations
    + Variance has bad units, but good math properties
    + $$\frac{\text{Variance of fitted values}}{{\text{Variance of } y}} = r^2 $$

+ A Variance Decomposition
    + $$\frac{\text{Variance of fitted values}}{\text{Variance of }y} = r^2$$
    + $$\frac{\text{Variance of residuals}}{\text{Variance of }y} = 1 - r^2$$

+ Residual Average and SD
    + The average of residuals is always $0$
    + $$\frac{\text{Variance of residuals}}{\text{Variance of }y} = 1 - r^2$$
    + $$\text{SD of residuals} = \sqrt{(1 - r²)} \text{  SD of } y$$

+ Demo
    ```python
    # ### A Measure of Clustering
    def plot_fitted(t, x, y):
        tbl = t.select(x, y)
        tbl.with_columns('Fitted Value', fitted_values(t, x, y)).scatter(0)

    plot_fitted(heights, 'MidParent', 'Child')

    correlation(heights, 'MidParent', 'Child')          # 0.3209498960639592

    np.var(fitted_values(heights, 'MidParent', 'Child')) / np.var(heights.column('Child'))
                                                        # 0.10300883578346642

    correlation(heights, 'MidParent', 'Child') ** 2     # 0.10300883578346624

    np.std(fitted_values(heights, 'MidParent', 'Child')) / np.std(heights.column('Child'))
                                                        # 0.32094989606395957

    correlation(dugong, 'Length', 'Age')                # 0.8296474554905714

    np.std(fitted_values(dugong, 0, 1)) / np.std(dugong.column(1))
                                                        # 0.8296474554905713

    plot_fitted(dugong, 'Length', 'Age')

    hybrid = Table.read_table('hybrid.csv')
    # vehicle           year    msrp        acceleration    mpg     class
    # Prius (1st Gen)   1997    24509.7     7.46            41.26   Compact
    # Tino              2000    35355       8.2             54.1    Compact
    # Prius (2nd Gen)   2000    26832.2     7.97            45.23   Compact

    plot_fitted(hybrid, 'acceleration', 'mpg')

    correlation(hybrid, 'acceleration', 'mpg')          # -0.5060703843771186

    np.std(fitted_values(hybrid, 3, 4)) / np.std(hybrid.column(4))
                                                        # 0.5060703843771186
    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V002300_DTH.mp4" alt="Lec 6.3 Properties of Residuals" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.4 Discussion Question

### Note

+ Discussion Question

    How does the SD of then fitted values related to $r$?<br/>
    A. $(\text{SD of fitted}) / (\text{SD of y}) = r$<br/>
    B. $(\text{SD of fitted}) / (\text{SD of y}) = |r|$<br/>
    C. $(\text{SD of fitted}) / (\text{SD of residuals}) = r$<br/>
    D. $(\text{SD of fitted}) / (\text{SD of residuals}) = |r|$

    Ans: B

+ No matter what the shape of the scatter plot, the SD of the residuals is a fraction of the SD of the observed values of $y$. The fraction is  $\sqrt{1-r^2}$.

$$\text{SD of residuals} = \sqrt{1 - r^2} \cdot \text{SD of }y$$

+ Demo
    ```python
    np.std(residuals(heights, 'MidParent', 'Child'))        # 3.3880799163953426

    r = correlation(heights, 'MidParent', 'Child')          # 0.32094989606395924

    np.sqrt(1 - r**2) * np.std(heights.column('Child'))     # 3.388079916395342

    np.std(residuals(hybrid, 'acceleration', 'mpg'))        # 9.43273683343029

    r = correlation(hybrid, 'acceleration', 'mpg')          # -0.5060703843771186

    np.sqrt(1 - r**2) * np.std(hybrid.column('mpg'))        # 9.43273683343029
    ```

### Video 

<a href="https://edx-video.net/BERD83FD2018-V002000_DTH.mp4" alt="Lec 6.4 Discussion Question" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading



### Practice

