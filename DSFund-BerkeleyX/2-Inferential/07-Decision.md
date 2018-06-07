# Section 7: Decisions and Uncertainty (Lec 7.1 - Lec 7.5)

+ Environment Initiation
    ```python
    from datascience import *
    import numpy as np

    import matplotlib.pyplot as plots
    ```
+ [Web Link](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/0eeac50995794429b04ca715f4effd91/4e7f40f6c2c94a6eafb0331b83aabc32/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%402751e24db49049acb477eccb15aca6e8)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/59d217c894d11dbd21d2d37ef6ae9675/notebooks/materials-x18/lec/x18/2/lec7.ipynb#)
+ [Local Notebook](./notebooks/lec7.ipynb)
+ [Local Python Code](./notebooks/lec7.py)

## Lec 7.1 Introduction and Terminology

### Notes

+ Incomplete Information
    + We are trying to choose between two views of the world, based on data in a sample.
    + It is not always clear whether the data are consistent with one view or the other.
    + Random samples can turn out quite extreme. It is unlikely, but possible.

+ Testing Hypotheses
    + A test chooses between two views of how data were generated
    + The views are called __hypotheses__
    + The test picks the hypothesis that is better supported by the observed data

+ Null and Alternative <br/>
    The method only works if we can simulate data under one of the hypotheses.
    + Null hypothesis
        + A well defined chance model about how the data were generated
        + We can simulate data under the assumptions of this model – “under the null hypothesis”
    + Alternative hypothesis
    + A different view about the origin of the data

+ Test Statistic
    The statistic that we choose to simulate, to decide between the two hypotheses <br/>
    Questions before choosing the statistic:<br/>
    + What values of the statistic will make us lean towards the null hypothesis?
    + What values will make us lean towards the alternative?
        + Preferably, the answer should be just “high” or just “low”. Try to avoid “both high and low”.

+ Prediction Under the Null Hypothesis
    + Simulate the test statistic under the null hypothesis; draw the histogram of the simulated values
    + This displays __the empirical distribution of the statistic under the null hypothesis__
    + It is a prediction about the statistic, made by the null hypothesis
        + It shows all the likely values of the statistic
        + Also how likely they are (if the null hypothesis is true)
    + The probabilities are approximate, because we can’t generate all the possible random samples

+ Conclusion of the Test <br/>
    + Resolve choice between null and alternative hypotheses
        + Compare the __observed test statistic__ and its empirical distribution under the null hypothesis
        + If the observed value is _not consistent_ with the distribution, then the test favors the alternative – “rejects the null hypothesis”
    + Whether a value is consistent with a distribution:
        + A visualization may be sufficient
        + If not, there are conventions about “consistency”

### Video

<a href="https://edx-video.net/BERD82FD2018-V002100_DTH.mp4" alt="Introduction and Terminology" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 7.2 Performing a Test

### Notes

+ The Problem
    + Large Statistics class divided into 12 discussion sections
    + Graduate Student Instructors (GSIs) lead the sections
    + After the midterm, students in Section 3 notice that the average score in their section is lower than in others

+ The GSI’s Defense
    + GSI’s position (Null Hypothesis): <br/>
        If we had picked my section at random from the whole class, we could have got an average like this one.
    + Alternative: <br/>
        No, the average score is too low. Randomness is not the only reason for the low scores.

+ Demo
    ```python
    scores = Table.read_table('scores_by_section.csv')
    scores.group('Section')
    scores.group('Section', np.average).show()

    # Null: The Section 3 average is like the average of 27 random scores from the class.
    # Alternative: No, it's too low.

    # observed statistic
    observerd_average = 13.6667
    np.average(scores.sample(27, with_replacement=False).column('Midterm'))

    averages = make_array()
    repetitions = 50000
    for i in np.arange(repetitions):
        new_average = np.average(scores.sample(27, with_replacement=False).column('Midterm'))
        averages = np.append(averages, new_average)

    Table().with_column('Random Sample Average', averages).hist(bins = 25, ec='w')
    plots.scatter(observerd_average, 0, color='red', s=30);
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002400_DTH.mp4" alt="Performing a Test" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 7.3 Statistical Significance

### Notes

+ Tail Areas
    <br/>[<img src="https://saylordotorg.github.io/text_introductory-statistics/section_09/bbb58b35dd040bcf4d8b4d3f7c72b679.jpg" alt="Right and Left Tails of a Distribution" width="300">](https://saylordotorg.github.io/text_introductory-statistics/s09-04-areas-of-tails-of-distribution.html#fwk-shafer-ch05_s04_f01)
    [<img src="https://saylordotorg.github.io/text_introductory-statistics/section_09/e4499588e283aa8c3339ac767a95ccef.jpg" alt="text" width="220">](https://saylordotorg.github.io/text_introductory-statistics/s09-04-areas-of-tails-of-distribution.html#fwk-shafer-ch05_s04_f01)
    [<img src="https://saylordotorg.github.io/text_introductory-statistics/section_09/c01e6ac6766ff9e8602e238df3d23be1.jpg" alt="Z Value that Produces a Known Area" width="155">](https://saylordotorg.github.io/text_introductory-statistics/s09-04-areas-of-tails-of-distribution.html#fwk-shafer-ch05_s04_f01)

+ Conventions About Inconsistency
    + __"Inconsistent"__: The test statistic is in the tail of the empirical distribution under the null hypothesis
    + __"In the tail,” first convention__:
        + The area in the tail is less than $5\%$
        + The result is _statistically significant_
    + __“In the tail,” second convention__:
        + The area in the tail is less than $1\%$
        + The result is _highly statistically 
+ Definition of the P-value
    + Formal name: __observed significance level__
    + The P-value is the chance,
        + under the null hypothesis,
        + that the test statistic
        + is equal to the value that was observed in the data
        + or is even further in the direction of the alternative.

+ Demo
    ```python
    np.count_nonzero(averages <= observerd_average) / repetitions
    # 0.05652 > 5%  
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002300_DTH.mp4" alt="Lec 7.3 Statistical Significance" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 7.4 An Error Probability

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002000_DTH.mp4" alt="An Error Probability" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 7.5 Origin of the Conventions

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002200_DTH.mp4" alt="Lec 7.5 Origin of the Conventions
" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice for Section 7

### Reading


### Practice






