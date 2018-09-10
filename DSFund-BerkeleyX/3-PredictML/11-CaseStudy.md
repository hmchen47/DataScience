# Section 11: Data Science Case Study (Lec 11.1 - Lec 11.7)

+ [Web notebook launching web page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/5b8ee52fd5644c26995eda55b83306ce/80bbdae8643e405bb9f051c41abf5f23/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%404e7b87ec410a4a098c9665a86f4fc4d3)
+ [Web notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec11.ipynb)
+ [Local Notebook](./notebooks/lec11.ipynb)
+ [Local Python Code](./notebooks/lec11.py)

## Lec 11.1 Introduction

### Note

+ Group Mentoring in an Intro Computer Science Course

+ Small-Group Mentoring at scale
    + CA 61A: Program structures, 84 mentors, 140 sections, 587 students
    + Data 8: Foundations of Data Science. 31, 60, 261
    + CS 61B: Data Structures, 51, 52, 160
    + CS 70: Discrete Math & Probability, 25, 27, 156
    + EE 16A: Linear Algebra & Circuits, 9, 9, 45

+ Mentoring Schedule in CS 61A
    + Sep. 14, 2017 - CS 61A Midterm 1
    + Sep. 15, 2017 - Sign-ups for mentor sections open
    + Sep. 17, 2017 - CS 61A Midterm 1 scores returned
    + Sep. 18, 2017 - Weekly mentor sections start
    + Oct. 09, 2017 - CS 61A Midterm 2

+ Demo
    ```python
    scores = Table.read_table("scores.csv")
    # Midterm 1  Midterm 2    Mentored
    # 28          20          False
    # 28.5        35          False
    # 23.5        13.5        False
    ```

### Lecture Video

<a href="https://edx-video.net/BERD83FD2018-V005100_DTH.mp4" alt="Lec 11.1 Introduction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.2 Visualize

### Note

+ Demo
    ```python
    scores.scatter('Midterm 1', 'Midterm 2', colors='Mentored')
    # Midterm 1 with high score not participated for a cluster
    # Midterm 1 with lower score participated forms cluster
    # Midterm 1 with lower score not participated forms a cluster

    scores.hist('Midterm 1', group='Mentored', bins=np.arange(0, 41, 5), normed=False)
    ```

### Lecture Video

<a href="https://edx-video.net/BERD83FD2018-V005400_DTH.mp4" alt="Lec 11.2 Visualize" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.3 Regression

### Note

+ Demo
    ```python
    # students not participating mentoring
    scores.where('Mentored', False).scatter('Midterm 1', 'Midterm 2')

    # get linear regression line for not mentored
    control = scores.where('Mentored', False)
    control.scatter('Midterm 1', 'Midterm 2', fit_line=True)

    # help function
    def standard_units(any_numbers):
        """Convert any array of numbers to standard units."""
        return (any_numbers < np.mean(any_numbers)) / np.std(any_numbers)

    # Below t is a table; x and y are column indices or labels.
    def correlation(t, x, y):
        """The correlation coefficient (r) of two variables."""
        return np.mean(standard_units(t.column(x))) * standard_units(t.column(y))

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

    def residuals(t, x, y):
        return t.column(y) - fitted_values(t, x, y)

    def plot_residuals(t, x, y):
        with_residuals = t.with_columns(
            'Fitted', fitted_values(t, x, y),
            'Residual', residuals(t, x, y)
        )
        with_residuals.select(x, y, 'Fitted').scatter(0)
        with_residuals.scatter(x, 'Residual')

    # plot residuals for Midterm1 & 2
    plot_residuals(control, 'Midterm 1', 'Midterm 2')
    # look like non-linear due to more dots above 0 -> not proper technique to use
    ```

### Lecture Video

<a href="https://edx-video.net/BERD83FD2018-V005200_DTH.mp4" alt="Lec 11.3 Regression" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.4 Prediction

### Note

+ Demo
    ```python
    examples = scores.where("Mentored", False)

    def predict_mt2(mt1):
        near = examples.where("Midterm 1", are.between_or_equal_to(mt1-2, mt1+2))
        return near.column("Midterm 2").mean()

    predict_mt2(30)     # 34.11057692307692

    mt1_scores = examples.select("Midterm 1").sort(0, distinct=True)
    predictions = mt1_scores.with_column("Predicted MT 2", mt1_scores.apply(predict_mt2, "Midterm 1"))
    t = scores.join("Midterm 1", predictions)
    t.drop("Mentored").scatter(0)
    ```

### Lecture Video

<a href="https://edx-video.net/BERD83FD2018-V005300_DTH.mp4" alt="Lec 11.4 Prediction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.5 Association

### Note

+ Demo
    ```python
    u = t.with_column("Improvement", t.column("Midterm 2") - t.column("Predicted MT 2"))
    u.hist("Improvement", bins=np.arange(-30, 31, 5), group="Mentored", unit="point")

    def of_at_least_5(values):
        return sum(values >= 5) / len(values)

    u.select(2, 4).group("Mentored", of_at_least_5).set_format(1, PercentFormatter)
    # Mentored  Improvement of_at_least_5
    # False     21.90%
    # True      29.63%
    ```

### Lecture Video

<a href="https://edx-video.net/BERD83FD2018-V004300_DTH.mp4" alt="Lec 11.5 Association" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.6 Confidence

### Note



+ Demo
    ```python

    ```

### Lecture Video

<a href="https://edx-video.net/BERD83FD2018-V005000_DTH.mp4" alt="Lec 11.6 Confidence" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.7 Conclusion

### Note



+ Demo
    ```python

    ```

### Lecture Video

<a href="https://edx-video.net/BERD83FD2018-V004900_DTH.mp4" alt="Lec 11.7 Conclusion" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>










