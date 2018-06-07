# Section 6: Comparing Distributions (Lec 6.1 - Lec 6.4)

+ Environment Initiation
    ```python
    from datascience import *
    import numpy as np

    import matplotlib.pyplot as plots
    plots.style.use('fivethirtyeight')
    get_ipython().run_line_magic('matplotlib', 'inline')

    np.set_printoptions(legacy='1.13')
    ```

+ [Lecture Notebook](https://hub.data8x.berkeley.edu/hub/user/59d217c894d11dbd21d2d37ef6ae9675/git-sync?repo=git://reposync/materials-x18&subPath=lec/x18/2/lec6.ipynb)
+ [Local Notebook](./notebooks/lec6.ipynb)
+ [local Python Code](./notebooks/lec6.py)

## Lec 6.1 Introduction

### Notes

+ Jury Selection in Alameda  <br/>
    Radical and Ethic Disparities in Alameda Country Jury Pools <br/>
    &nbsp;&nbsp; - A Report  by the ACLU of North California

+ Jury Panels
    <br/><img src="./diagrams/juryFlow.png" alt="Jury Selection Flow" width="600"><br/>
    Section 197 of California's Code of Civil Procedure says, "All persons selected for jury service shall be selected at random, from a source or sources inclusive of a representative cross section of the population of the area served by the court."

+ Demo
    ```python
    jury = Table().with_columns(
        'Ethnicity', make_array('Asian', 'Black', 'Latino', 'White', 'Other'),
        'Eligible', make_array(0.15, 0.18, 0.12, 0.54, 0.01),
        'Panels', make_array(0.26, 0.08, 0.08, 0.54, 0.04)
    )
    jury

    jury.barh('Ethnicity')
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V001700_DTH.mp4" alt="Introduction" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.2 Total Variation Distance

### Notes

+ Model and Alternative
    + _Model_: The people on the jury panels were selected at random from the eligible population
    + _Alternative viewpoint_: No, they weren’t

+ Distance Between Distributions
    + People on the panels are of multiple ethnicities
    + Distribution of ethnicities is categorical
    + To see whether the the distribution of ethnicities of the panels is close to that of the eligible jurors, we have to measure the distance between two categorical distributions
    
+ Total Variation Distance <br/>
    Every distance has a computational recipe __Total Variation Distance (TVD)__:
    + For each category, compute the difference in proportions between two distributions
    + Take the absolute value of each difference
    + Sum, and then divide the sum by 2

+ Demo
    ```python
    jury_with_diffs = jury.with_column('Difference', jury.column('Panels') - jury.column('Eligible'))

    jury_with_diffs = jury_with_diffs.with_column('Absolute Difference', np.abs(jury_with_diffs.column('Difference')))

    sum(jury_with_diffs.column('Absolute Difference'))

    sum(jury_with_diffs.column('Absolute Difference')) / 2

    def total_variation_distance(distribution_1, distribution_2):
        return sum(np.abs(distribution_1 - distribution_2)) / 2

    total_variation_distance(jury.column('Panels'), jury.column('Eligible'))
    ```

### Video


<a href="https://edx-video.net/BERD82FD2018-V001800_DTH.mp4" alt="Total Variation Distance" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.3 Assessment

### Notes

+ Demo
    ```python
    eligible = jury.column('Eligible')

    panels_and_sample = jury.with_column('Random Sample', sample_proportions(1453, eligible))
    
    panels_and_sample.barh('Ethnicity')

    total_variation_distance(panels_and_sample.column('Random Sample'), eligible)
    total_variation_distance(jury.column('Panels'), eligible)
    tvds = make_array()

    repetitions = 10000
    for i in np.arange(repetitions):
        sample_distribution = sample_proportions(1453, eligible)
        new_tvd = total_variation_distance(sample_distribution, eligible)
        tvds = np.append(tvds, new_tvd)

    Table().with_column('Total Variation Distance', tvds).hist(bins = np.arange(0, 0.2, 0.005), ec='w')

    total_variation_distance(jury.column('Panels'), eligible)
    ```

### Video


<a href="https://edx-video.net/BERD82FD2018-V001900_DTH.mp4" alt="Assessment" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.4 Summary

### Notes

+ Summary of the Method <br/>
    To assess whether a sample was drawn randomly from a known categorical distribution:
    + Use TVD as the statistic because it measures the distance between categorical distributions
    + Sample at random from the population and compute the TVD from the random sample; repeat numerous times
    + Compare:
        + Empirical distribution of simulated TVDs
        + Actual TVD from the sample in the study

### Video


<a href="https://edx-video.net/BERD82FD2018-V001600_DTH.mp4" alt="Summary" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading

This guide assumes that you have watched the videos for Section 6.

This corresponds to textbook sections:

[Chapter 11.2: Multiple Categories](https://www.inferentialthinking.com/chapters/11/2/multiple-categories.html)

In section 6, we examined jury selection in Alameda County to investigate if certain ethnic groups are underrepresented among jury panelists. This example showed us how to assess whether a sample was drawn randomly from a known categorical distribution. It is important to realize that we use total variation distance (tvd) as the test statistic because it measures distance between categorical distributions. 

Try applying what you learned in the following practice questions.

### Practice

Let's revisit the example of jury selection in Alameda County from lecture. Like from lecutre, one distribtuion represents, for each ethnicity, the proportion of all eligible juror candidates of that ethnicity. The other distribution represents the proportion of people of a particular ethnicity that were selected to be on the jury panel. The bar chart of the two distributions can be found below.

<br/><img src="https://prod-edxapp.edx-cdn.org/assets/courseware/v1/43e88ee5f5594e454485e80b52e87659/asset-v1:BerkeleyX+Data8.2x+1T2018+type@asset+block/jury.png" alt="text" width="450">

Histogram of ethnicity distribution of jurors selected in jurys in Alameda County. The distribution of eligible jurors is Asian: 0.15, Black: 0.18, Latino: 0.12, White: 0.54, Other: 0.01. The actual distribution of the jurors is Asian: 0.26, Black: 0.08, Latino: 0.08, White: 0.54, Other: 0.04. 

Q1. From the histogram, which ethnicities are underrepresented in the jury panel? Select all that apply

    a. Asian
    b. Black
    c. Latino
    d. White
    e. Other

    Ans: b, c

Q2. Which ethnicities are overrepresented in the jury panel? Select all that apply

    a. Asian
    b. Black
    c. Latino
    d. White
    e. Other

    Ans: a, e

Q3. What is the total combined length of the bars for the eligible juror distribution?

    Ans: 1.0
 
Q4. What is the total combined length of the bars for the panel juror distribution?

    Ans:1.0
 
The table of the exact proportions corresponding to the chart above can be found below.

| Ethnicity | Eligible | Panel|
|-----------|----------|------|
| Asian | 0.15 | 0.26 |
| Black | 0.18 | 0.08 |
| Latino | 0.12 | 0.08 |
| White | 0.54 | 0.54 |
| White | 0.01 | 0.04 |
    
Your friend suggests the following statistic for calculating the difference between the two distributions.

1. Subtract the height of one bar in the distribution of eligible jurors from its corresponding bar in the distribution of jurors as they appeared in the panels
2. Add all those differences together

Q5. What is the value of the statistic proposed by your friend?

Ans: 0
 
Q6. True or False: your friend chose a good statistic.

Ans: False

Q7. Instead, calculate the total variation distance for the two distritbutions shown above. What is the total variation distance? (Don't forget to divide by 2.)

Ans: 0.14



