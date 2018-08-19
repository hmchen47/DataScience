# Section 5: Least Squares (Lec 5.1 - Lec 5.4)

## Lec 5.1 Linear Regression Review

### Note

+ Regression Line
    <a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/11f472f1d45d411993d1f696435f7d51/ec06aa8ad4eb4e30ae9bea5c093e0454/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%4083cd119349084de2915411d56e9e3056">
        <br/><img src="./diagrams/lec4-1.png" alt="Regression Line" width="400">
    </a>

+ Regression Line Equation <br/>
    In original units, the regression line has this equation:

    $$\frac{\text{estimate of } y - \text{average of } y}{|text{SD of }y} = r \times \frac{\text{the given } x - \text{average of } x}{\text{SD of } x}$$

    where the left-hand formula is "y in standard units" and the right-hand formula is "x in standard units".

    $$y = \text{slope } \times x + \text{intercept}$$

    $$\text{slope of the regression line} = r \times \frac{\text{SD of} y}{\text{SD of } x}$$

    $$\text{intercept of the regression line} = \text{average of } y - \text{slope } \times \text{average of } x$$


### Video

<a href="https://edx-video.net/BERD83FD2018-V001800_DTH.mp4" alt="Lec 5.1 Linear Regression Review" target="blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 5.2 Discussion Question

### Note

+ Discussion Question
    + A course has a midterm (average $70$; standard deviation $10$) and a really hard final (average $50$; standard deviation $12$)
    + If the scatter diagram comparing midterm & final scores for students has a typical oval shape with correlation $0.75$, then...
    + What do you expect the average final score would be for students who scored $90$ on the midterm?
    + How about $60$ on the midterm?

+ Demo
    ```python
    x_mean = 70
    x_sd = 10
    y_mean = 50
    y_sd = 12
    r = 0.75

    midterm_score = 90
    x = midterm_score
    (((x - x_mean) / x_sd) * r * y_sd) + y_mean     # 68

    midterm_score = 60
    x = midterm_score
    (((x - x_mean) / x_sd) * r * y_sd) + y_mean     # 41

    # y = a * x + b
    a = r * (y_sd / x_sd)
    b = y_mean - a * x_mean

    a * 90 + b                                      # 68
    a * 60 + b                                      # 41
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001700_DTH.mp4" alt="Lec 5.2 Discussion Question" target="blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 5.3 Squared Error

### Note

+ Error in Estimation
    + $\text{error} = \text{actual value} − \text{estimate}$
    + Typically, some errors are positive and some negative
    + To measure the rough size of the errors
        + _square_ the _errors_ to eliminate cancellation
        + take the _mean_ of the squared errors
        + take the square _root_ to fix the units
        + __root mean square error__ (rmse)

    <a href="url">
        <br/><img src="url" alt="text" width="450">
    </a>


+ Demo
    ```python
    little_women = Table.read_table('little_women.csv')
    little_women = little_women.move_to_start('Periods')
    little_women.show(3)
    # Periods     Characters
    # 189         21759
    # 188         22148
    # 231         20558
    # ... (row omitted)

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

    little_women.scatter(0, 1)

    correlation(little_women, 0, 1)     # 0.9229576895854816

    a = slope(little_women, 0, 1)
    b = intercept(little_women, 0, 1)
    a * little_women.column(0) + b
    # array([21183.59679439, 21096.61895313, 24836.66612724, 21705.46384194,
    #        26924.13431744, 16921.68257274, 16138.88200141, 23358.04282585,
    #        34056.31730062, 20835.68542936, 21531.50815943, 42058.27869638,
    #        19965.90701678, 20400.79622307, 20487.77406433, 19704.973493  ,
    #        18226.35019161, 17269.59393777, 15269.10358883, 17356.57177903,
    #        28141.82409506, 15182.12574757, 26228.31158737, 20574.75190559,
    #        12659.76835108, 17791.46098532, 14225.36949373, 28315.77977757,
    #        25010.62180976, 23705.95419088, 20226.84054055, 24227.82123843,
    #        24923.6439685 , 27098.08999996, 22227.33088949, 13442.5689224 ,
    #        18400.30587413, 22662.22009578, 27619.95704751, 10050.43311333,
    #        21009.64111188, 15008.17006505, 31273.02638036, 13007.67971611,
    #        13094.65755737, 25097.59965102, 38840.09856983])

    def linear_fit(t, x, y):
        a = slope(t, x, y)
        b = intercept(t, x, y)
        return a * t.column(x) + b

    lw_fitted = little_women.with_column(
        'fitted',
        linear_fit(little_women, 0, 1)
    )
    lw_fitted.scatter(0)

    # Squared Error
    sample = [[131, 14431], [231, 20558], [392, 40935], [157, 23524]]
    def lw_errors(slope, intercept):
        print('Slope:    ', np.round(slope), 'characters per period')
        print('Intercept:', np.round(intercept), 'characters')
        little_women.scatter('Periods', 'Characters')
        xlims = np.array([50, 450])
        plots.plot(xlims, slope * xlims + intercept, lw=2)
        for x, y in sample:
            plots.plot([x, x], [y, slope * x + intercept], color='r', lw=2)

    lw_errors(50, 1000)     # Slope: 50 characters per period; Intercept: 1000 characters
    lw_errors(-50, 20000)   # Slope: -50 characters per period; Intercept: 20000 characters

    def lw_rmse(slope, intercept):
        lw_errors(slope, intercept)
        x = little_women.column('Periods')
        y = little_women.column('Characters')
        predicted = slope * x + intercept
        mse = np.mean((y - predicted) ** 2)
        print("Root mean squared error:", mse ** 0.5)

    lw_rmse(50, 10000)
    # Slope:     50 characters per period
    # Intercept: 10000 characters
    # Root mean squared error: 4322.167831766537
    lw_rmse(-50, 20000)
    # Slope:     -50 characters per period
    # Intercept: 20000 characters
    # Root mean squared error: 15556.958991519832
    lw_rmse(90, 4000)
    # Slope:     90 characters per period
    # Intercept: 4000 characters
    # Root mean squared error: 2715.5391063834586
    lw_rmse(slope(little_women, 0, 1), intercept(little_women, 0, 1))
    # Slope:     87.0 characters per period
    # Intercept: 4745.0 characters
    # Root mean squared error: 2701.690785311856
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001900_DTH.mp4" alt="Lec 5.3 Squared Error" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 5.4 Least Squares

### Note

    <a href="url">
        <br/><img src="url" alt="text" width="450">
    </a>


+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V002400_DTH.mp4" alt="Lec 5.4 Least Squares" target="blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading



### Practice



