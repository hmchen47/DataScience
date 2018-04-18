# Section 8: Histograms (Lec 8.1 - Lec 8.7)

+ Displaying a Categorical Distribution
    + The distribution of a variable (a column, e.g. Studios) describes the frequencies of its different values
    + the `group` method counts the number of rows for each value in the column (e.g. the number of top movie released by each studio)
    + Bar charts can display the distribution of a categorical variable (e.g. studios)
        + One bar for each category
        + Length of bar is the count of individuals in that category

## Lec 8.1 Area Principle

### Notes

+ What's Wrong with This Picture?  
    ![Diagram](https://i1.wp.com/flowingdata.com/wp-content/uploads/2012/03/ipad-expanded-battery.jpg?w=960&ssl=1)
    + 70% bigger for iPad 2 battery, but double in diagram due to (1.7 x 1.7)
    + Flowing [data](https://flowingdata.com/2012/03/16/new-ipad-battery-size-is-huge/)
+ Area Principle  
    __Areas__ should be proportional to the values they represent.  
    For example:
    + If represent $20%$ of a population: A
    + Then $40%$ can be represented by: AA
    + But noy by AAAA
+ Question:  
    Is this consistent with the area principle? (Only the Right bottom one)

    ![diagram](http://www.princeton.edu/~ina/images/infographics/starbucks_small.jpg)

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/qEYz6D0MKq8){:target="_blank"}


## Lec 8.2 Binning

### Notes

+ Binning Numerical Values  
    Binning is counting the number of numerical valuesthat lie within ranges, called __bins__.
    + Bins are defined by their lower bounds (inclusive)
    + The upper bound is the lower bound of the next bin
    + e.g., 188, 170, 189, 163, 183, 171, 185, 168, 173, ...
        + $[160, 165)$: 163
        + $[165, 170)$: 168
        + $[170, 175)$: 170, 171, 173
        + ...
+ Demo
    ```python
    top = Table.read_table('top_movies.csv')
    # add age of movies
    age = 2017 - top.column('Year')
    top = top.with_column('Age', age)
    
    min(top.column('Age')), max(top.column('Age'))  # get max and min ages
    my_bins = make_array(0, 5, 10, 15, 25, 40, 65, 100) # uneven intervals
    top.bin('Age', bins = my_bins)  # create bins with given intervals
    top.bin('Age', bins = np.arange(0, 101, 25))
    top.bin('Age', bins = np.arange(0, 60, 25))
    # the last bin is [50, 50), the value 50 thrown into the last 2nd bin
    top.where('Age', 50)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/kREoWbByNZs){:target="_blank"}


## Lec 8.3 Example 1

### Notes

+ Question: 


    | Table A | bin | Score Count | | Table B | bin | Score Count |
    |--|-----|-------------|--|--|--|-----|-------------|
    | |  70 | 1 | | |  70 | 10 |
    | |  80 | 2 | | |  80 | 20 |
    | |  90 | 3 | | |  90 | 40 |
    | | 100 | 0 | | | 100 |  0 |

    | Table C | bin | Score Count | | Table D | bin | Score Count |
    |--|-----|-------------|--|--|--|-----|-------------|
    | |  70 | 10 | | |  70 | 321 |
    | |  80 | 20 | | |  80 | 642 |
    | |  90 | 30 | | |  90 | 963 |
    | | 100 | 0 | | | 100 |  0 |

    + Which of A, B, C, and D are consistent with this distribution of test scores?

        Ans: A, C, and D


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/vz5VLqrw-tA){:target="_blank"}


## Lec 8.4 Drawing Histograms

### Notes

+ Histogram
    + Chart that displays the distribution of a numerical variable
    + Use bins; there is one bar corresponding to each bin
    + Use the area principle: the __area__ of each bar is the percent of individuals in the corresponding bin
+ Demo
    ```python
    top.bin('Age', bins = my_bins)      # uneven bins
    top.hist('Age', bins = my_bins, unit = 'Year')
    top.hist('Age', bins = my_bins, unit = 'Year', normed = False) # not normalized and not follow area principle
    top.hist('Age', bins = np.arange(0, 110, 10), unit = 'Year')
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/xPv7VNSBJZQ){:target="_blank"}


## Lec 8.5 Density

### Notes

+ Histogram Axes
    + By default, `hist` uses a scale (`normed=True`) that ensures the area of the chart sums to 100%
    + The area of each bar is a percentage of the whole
    + The horizontal axis is a number line (e.g., years), and the bins sizes don't have to be equal to each other
    + The vertical axis is a rate (e.g., percent per year)
+ How to calculate Height  
    The $[25, 40)$ bin contains 42 out of 200 movies
    + $42/200 = 21\%$
    + The bin is $40 - 25 = 15$ years wide
    + $\text{Height of bar} = 21\% / 15 \text{ years} = 1.4$ percent per year
+ Height Measures Density  
    $$ \text{Height} = \frac{\% \text{in bin}}{\text{width of bin}}$$  
    + The height measures the percent of data in the bin __relative to the amount of space in the bin.__
    + Height measures crowdedness or __density__.
    + Units: percent per unit on the horizontal axis
+ Area Measures Percent  
    $$\text{Area of bar} = \% \text{ in bin} = \text{Height x width in bin}$$
    + How many individuals in the bin? --> Use __area__
    + How crowded is the bin? --> Use __height__
+ Bar Chart or Histogram?
    + Bar Chart
        + Distribution of categorical variable
        + Bars have arbitrary (but equal) widths and spacing
        + height (or length) of bars proportional to the percent of individuals
    + Histograms
        + Distribution of numerical variable
        + Horizontal axis is numerical , hence to scale with no gaps
        + Area of bars proportional to the percent of individuals; height measures density

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/F8Pv0DWqPls){:target="_blank"}


## Lec 8.6 Example 2

### Notes

+ Question  
    What is the height of each bar in this histogram?  
    `incomes.hist(1, bins=[0, 15,25, 85])`  
    What are the vertical axis units? 
+ Answer
    + Vertical axis: Percent per million
    + `incomes.hist(1, bins=[0, 15,25, 85], unit='million')`
    + $[0, 15)$: $(45\%) / (15 \text{ million}) = 3 \% \text{ per million}$
    + $[15, 25)$: $(40\%) / (10 \text{ million}) = 4 \% \text{ per million}$
    + $[25, 85)$: $(15\%) / (60 \text{ million}) = .25 \% \text{ per million}$

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/ZwvovAbWUyY){:target="_blank"}


## Lec 8.7 Example 3

### Notes

+ Question:  
    Here is a numerical description of a histogram.  The height of one of the bar is missing.  Can you fill it in?  

    | Wight bin (pounds) | Height of Bar (percent per pound) |
    |---|---|
    | $[0, 5)$ | 2 |
    | $[5, 10)$ | 3 |
    | $[10, 20)$ | ? |
    | $[20, 50)$ | 1 |
+ Answer:  
    Total area of visible bars: $ (5 x 2) + (5 x 3) + (30 x 1) = 55\%$  
    Area of hidden bar = $1 - 55\% = 45\%$  
    Height of hidden bar = $45\% / 10 = 4.5\% \text{ per pound}$

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/Jl5fNPkEcDI){:target="_blank"}


## Reading and Practice for Section 8

### Reading

This guide assumes that you have watched section 8 (video lecture segments Lec 8.1, Lec 8.2, Lec 8.3, Lec 8.4, Lec 8.5, Lec 8.6, Lec 8.7) in Courseware.

This corresponds to textbook section:

[Chapter 7.2: Visualizing Numerical Distributions](https://www.inferentialthinking.com/chapters/07/2/visualizing-numerical-distributions.html){:target="_blank"}

A histogram visualizes a single numerical variable. A histogram of a numerical dataset looks very much like a bar chart, though it has some important differences. 

Two defining properties of Histograms are:

1. The bins are drawn to scale and contiguous (though some might be empty), because the values on the horizontal axis form a continuous number line.
2. The area of each bar is proportional to the number of entries in the bin.

Histograms are often drawn using the density scale, where the area of a bar is equal to the percent of entries in that bin. The density scale is advantageous because the areas are interpretable, and the histogram areas are drawn to scale even if the widths of the bars are different.

Computing the bar heights uses that fact that a bar is a rectangle: 

> (area of the bar) = (height of the bar) * (width of the bar).

Here are some practice problems about histograms.

### Practice

The table `nba` has a column labeled `salary` containing the 2015-2016 salaries of NBA players. The following histogram was generated by calling `nba.hist(...)`. Also included below is a table with the bins and their corresponding heights.

![Diagram](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/ba69e1581c19bdb329e1e6c22fc71623/asset-v1:BerkeleyX+Data8.1x+1T2018+type@asset+block/q8_histogram.png)

bin widths and heights for `nba.hist(...)`

bin (million dollars): [0, 2), [2, 4), [4, 12), [12, 18), [18, 26)  
height (percent per million dollars): 17.64, 11.39, 3.60, 1.60, 0.45

Q1. Which bin contains the most number of players?

    a. bin [0, 2)
    b. bin [2, 4)
    c. bin [4, 12)
    d. bin [12, 18)
    e. bin [18, 26)

    Ans: a

Q2. Which bin contains the least number of players?

    a. bin [0, 2)
    b. bin [2, 4)
    c. bin [4, 12)
    d. bin [12, 18)
    e. bin [18, 26)

    Ans: e

Q3. What proportion of players have a salary between 4 and 12 million dollars?

    a. Somewhere between 3% and 4%
    b. Somewhere between 10% and 12%
    c. Somewhere between 28% and 30%

    Ans: c

