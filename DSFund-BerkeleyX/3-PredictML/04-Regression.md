# Section 4: Regression (Lec 4.1 - Lec 4.5)

+ [Launching Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/11f472f1d45d411993d1f696435f7d51/b9a43abd58fa47579e95d24fed03db8e/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%4096b729c15ebc4778abd9c403bfebdb14)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec4.ipynb)
+ [Local Notebook](./notebook/lec4.ipynb)
+ [Local Python Code](./notebook/lec4.py)

## Lec 4.1 Prediction

### Notes

+ Prediction a Numerical Variable - Approach to predicting an outcome for an individual
    + Find others who are like that individual
    + and whose outcomes you know
    + Use those outcomes as the basis of your prediction.

+ Galon's Data
    + Goal: Predict the height of a new child, based in that child's midparent height.
    <a href="http://genomicsclass.github.io/book/pages/intro_using_regression.html">
        <br/><img src="http://genomicsclass.github.io/book/pages/figure/intro_using_regression-galton_data-1.png" alt="Father & son heights" width="300">
    </a>

+ Making a Prediction
    + How to predict a child's height, given a midparent height of 68 inches?
    + __Idea__: Use the average height of the children of all the families in which the midparent height is close to 68 inches.

+ Galton's Heights
    <a href="url">
        <br/><img src="./diagrams/lec4-2.png" alt="Estimate of 68 in" width="300">
    </a>
    <a href="url">
        <img src="./diagrams/lec4-3.png" alt="Estimates of all heights" width="300">
    </a>

+ Nearest Neighbor Regression
    + A method for prediction:
        + Group each x with a representative x value (rounding)
        + Average the corresponding y values for each group
    + For each representative x value, the corresponding prediction is the average of the y values in the group.
    + Graph these predictions.
    + If the association between x and y is linear, then points in the graph of averages tend to fall on the regression line.

+ Graph of Averages
    + For each value of x, the predicted value of y is the average of the y values of the nearest neighbors.
    + Graph these predictions for all the values of x.  That's the __graph of averages__.
    + If the association between the two variables is linear, then points on the graph of averages tend to fall on or near a straight line.  That's the __regression line__.

+ Demo
    ```python
    galton = Table.read_table('galton.csv')

    heights = Table().with_column(
        'MidParent', galton.column('midparentHeight'),
        'Child', galton.column('childHeight')
        )
    # MidParent     Child
    # 75.43         73.2
    # 75.43         69.2
    # 75.43         69
    # ... (rows omitted)

    heights.scatter('MidParent')

    def predict_child(x):
        chosen = heights.where('MidParent', are.between(x - 0.5, x + 0.5))
        return np.average(chosen.column('Child'))

    predictions = heights.apply(predict_child, 'MidParent')

    heights = heights.with_column(
        'Original Prediction', predictions
    )
    # MidParent  Child   Original Prediction
    # 75.43      73.2    70.1
    # 75.43      69.2    70.1
    # 75.43      69      70.1
    # ... (rows omitted)

    heights.scatter('MidParent')
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001400_DTH.mp4" alt="Lec 4.1 Prediction" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Lec 4.2 Linear Regression

### Notes


+ Where is the prediction line?
    <a href="url">
        <br/><img src="./diagrams/lec3-1.png" alt="Various correlation coefficients" width="600">
    </a>

+ Identifying the Line <br/>
    If the scatter plot is "football shaped" (oval), then we can spot an important feature of the regression line.

+ Regression Estimate - To find the regression estimate of y:
    + Convert the given x to standard units
    + Multiply by r
    + That’s the regression estimate of y, but:
        + It’s in standard units: $\text{estimat of} y_{(su)} = r \times \text{given } x_{(su)}$
        + So covert it back to the original units of y

+ Question <br/>
    + The scatter plot of the lengths and wights of adult mountain lions is roughly football shaped.
        + Weights: average 84 inches, SD 8 inches
        + Weights: average 125 pounds, SD 15 pounds
        + Correlation between length and wight: 0.6
    + Q: Find the regression estimate the length of a mountain lion that weights 155 pounds.
    + Ans: 
        + Step 1:  The weight of 155 pounds is $(155 - 125) / 15 = 2$ standard units
        + Step 2: The estimate of length is $0.6 x 2 = 1.2$ standard units
        + Step 3: $1.2$ SDs above average in length is $1.2 x 8 + 84 = 93.6$ inches


+ Demo
    ```python
    r = 0.6
    x_demo = np.random.normal(0, 1, 10000)
    z_demo = np.random.normal(0, 1, 10000)
    y_demo = r*x_demo + np.sqrt(1 - r**2)*z_demo

    def trial_line():
        plots.figure(figsize=(7,7))
        plots.xlim(-4, 4)
        plots.ylim(-4, 4)
        plots.scatter(x_demo, y_demo, s=10)
        #plots.plot([-4, 4], [-4*0.6,4*0.6], color='g', lw=2)
        plots.plot([-4,4],[-4,4], color='r', lw=2)
        #plots.plot([1.5,1.5], [-4,4], color='k', lw=2)
        plots.xlabel('x in standard units')
        plots.ylabel('y in standard units');

    def trial_with_vertical():
        trial_line()
        plots.plot([1.5,1.5], [-4,4], color='k', lw=2)

    def both_with_vertical():
        trial_line()
        plots.plot([1.5,1.5], [-4,4], color='k', lw=2)
        plots.plot([-4, 4], [-4*0.6,4*0.6], color='g', lw=2)

    def regression_line(r):
        x = np.random.normal(0, 1, 10000)
        z = np.random.normal(0, 1, 10000)
        y = r*x + (np.sqrt(1-r**2))*z
        plots.figure(figsize=(7, 7))
        plots.xlim(-4, 4)
        plots.ylim(-4, 4)
        plots.scatter(x, y, s=10)
        plots.plot([-4, 4], [-4*r,4*r], color='g', lw=2)
        if r >= 0:
            plots.plot([-4,4],[-4,4], lw=2, color='r')
        else:
            plots.plot([-4,4], [4,-4], lw=2, color='r')

    trial_line()

    trial_with_vertical()

    both_with_vertical()

    r = 0.7     # try various r values to observe the relationship of two lines
    regression_line(r)
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001500_DTH.mp4" alt="Lec 4.2 Linear Regression" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 4.3 Regression to the Mean

### Notes


    <a href="url">
        <br/><img src="url" alt="text" width="450">
    </a>

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001300_DTH.mp4" alt="Lec 4.3 Regression to the Mean" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 4.4 Regression Equation

### Notes


    <a href="url">
        <br/><img src="url" alt="text" width="450">
    </a>

+ Demo
    ```python

    ```

### Video


<a href="https://edx-video.net/BERD83FD2018-V001600_DTH.mp4" alt="Lec 4.4 Regression Equation" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 4.5 Interpreting the Slope

### Notes



    <a href="url">
        <br/><img src="url" alt="text" width="450">
    </a>


+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001200_DTH.mp4" alt="Lec 4.5 Interpreting the Slope" target="_blank">
    <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading



### Practice



