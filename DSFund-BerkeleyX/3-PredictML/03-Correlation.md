# Section 3: Correlation (Lec 3.1 - Lec 3.3)

+ [Alunching Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/7bba8d29a20946e5be64e508fd3481b2/7c8a2a1d0e8241e99bff43c05c23b011/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%402d9f38b83b784ebc9cb99d1d459549a7)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec3.ipynb)
+ [Local Notebook](./notebook/lec3.ipynb)
+[Local Python Code](./notebook/lec3.py)

## Lec 3.1 Visualization

### Notes

+ Prediction - To predict the value of a variable,
    + Identify attributes that are associated with that variable and that you can measure
    + Describe the relation between the attributes and the variable you want to predict
    + Use the relation to make your prediction

+ Guessing the Future
    + Based on incomplete information
    + One way of making predictions:
        + To predict an outcome for an individual,
        + find others who are like that individual
        + and whose outcomes you know.
        + Use those outcomes as the basis of your prediction.

+ Two Numerical Variables
    + Trend
        + Positive association
        + Negative association
    + Pattern
        + Any discernible “shape” in the scatter
        + Linear
        + Non-linear

    + __Visualize, then quantify__

+ Demo
    ```python
    def r_scatter(r):
        plots.figure(figsize=(5,5))
        "Generate a scatter plot with a correlation approximately r"
        x = np.random.normal(0, 1, 1000)
        z = np.random.normal(0, 1, 1000)
        y = r*x + (np.sqrt(1-r**2))*z
        plots.scatter(x, y, color='darkblue', s=20)
        plots.xlim(-4, 4)
        plots.ylim(-4, 4)

    galton = Table.read_table('galton.csv')

    heights = Table().with_columns(
        'MidParent', galton.column('midparentHeight'),
        'Child', galton.column('childHeight')
        )
    # MidParent   Child
    # 75.43       73.2
    # 75.43       69.2
    # 75.43       69
    # ...(row omitted)

    heights.scatter('MidParent')            # positive; linear

    hybrid = Table.read_table('hybrid.csv')
    # vehicle           year    msrp    acceleration    mpg     class
    # Prius (1st Gen)   1997    24509.7	7.46            41.26   Compact
    # Tino              2000    35355   8.2             54.1    Compact
    # Prius (2nd Gen)   2000    26832.2 7.97            45.23   Compact
    # ... (rows omitted)

    hybrid.scatter('mpg', 'msrp')           # negative; non-linear
    hybrid.scatter('acceleration', 'msrp')  # positive; non-linear
    suv = hybrid.where('class', 'SUV')
    suv.num_rows                            # 39
    suv.scatter('mpg', 'msrp')              # negative; linear

    def standard_units(x):
        "Convert any array of numbers to standard units."
        return (x - np.average(x)) / np.std(x)

    Table().with_columns(
        'mpg (standard units)',  standard_units(suv.column('mpg')), 
        'msrp (standard units)', standard_units(suv.column('msrp'))
    ).scatter(0, 1)                         # negative; linear
    plots.xlim(-3, 3)
    plots.ylim(-3, 3);
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001000_DTH.mp4" alt="Lec 3.1 Visualization" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Lec 3.2 Calculation

### Notes


+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V001100_DTH.mp4" alt="Lec 3.2 Calculation" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Lec 3.3 Interpretation

### Notes


+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000900_DTH.mp4" alt="Lec 3.3 Interpretation" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>

## Reading and Practice

### Reading



### Practice


