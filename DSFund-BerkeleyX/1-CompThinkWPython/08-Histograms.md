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

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/qEYz6D0MKq8)


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

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/kREoWbByNZs)


## Lec 8.3 Example 1

### Notes

+ Question: 
    + Table A

        | bin | Score Count |
        |-----|-------------|
        |  70 | 1 |
        |  80 | 2 |
        |  90 | 3 |
        | 100 | 0 |

    + Table B

        | bin | Score Count |
        |-----|-------------|
        |  70 | 10 |
        |  80 | 20 |
        |  90 | 40 |
        | 100 |  0 |

    + Table C

        | bin | Score Count |
        |-----|-------------|
        |  70 | 10 |
        |  80 | 20 |
        |  90 | 30 |
        | 100 |  0 |

    + Table D

        | bin | Score Count |
        |-----|-------------|
        |  70 | 321 |
        |  80 | 642 |
        |  90 | 963 |
        | 100 |   0 |
    + Which of A, B, C, and D are consistent with this distribution of test scores?

        Ans: A, C, and D


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/vz5VLqrw-tA)


## Lec 8.4 Drawing Histograms

### Notes

+ Demo
    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/xPv7VNSBJZQ)


## Lec 8.5 Density

### Notes

+ Demo
    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/F8Pv0DWqPls)


## Lec 8.6 Example 2

### Notes

+ Demo
    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/ZwvovAbWUyY)


## Lec 8.7 Example 3

### Notes

+ Demo
    ```python

    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/Jl5fNPkEcDI)


## Reading and Practice for Section 8

### Reading


### Practice



