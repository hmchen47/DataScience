# Section 8a: A/B Testing (Lec 8.1 - Lec 8.3)

[Web Notebook](https://hub.data8x.berkeley.edu/user/59d217c894d11dbd21d2d37ef6ae9675/notebooks/materials-x18/lec/x18/2/lec8.ipynb)

[Local Notebook](./notebooks/lec8.ipynb)

[Local Python Code](./notebooks/lec8.py)

## Lec 8.1 Introduction

### Notes

+ Comparing Two Samples
    + Compare values of sampled individuals in Group A with values of sampled individuals in Group B.
    + Question: Do the two sets of values come from the same underlying distribution?
    + Answering this question by performing a statistical test is called __A/B testing.__

+ Demo
    ```python
    from datascience import *
    import numpy as np

    import matplotlib.pyplot as plots
    plots.style.use('fivethirtyeight')
    get_ipython().run_line_magic('matplotlib', 'inline')

    baby = Table.read_table('baby.csv')

    smoking_and_birthweight = baby.select('Maternal Smoker', 'Birth Weight')

    smoking_and_birthweight.group('Maternal Smoker')
    # Maternal Smoker	count
    #          False	715
    #          True	    459

    smoking_and_birthweight.hist('Birth Weight', group='Maternal Smoker')

    means_tbl = smoking_and_birthweight.group('Maternal Smoker', np.average)
    # Maternal Smoker	Birth Weight average
    #           False	123.085
    #           True	113.819

    means = means_tbl.column(1)
    observed_difference = means.item(0) - means.item(1)
    # 9.266142572024918
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002600_DTH.mp4" alt="Lec 8.1 Introduction" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 8.2 Hypotheses and Statistic

### Notes

+ The Groups and the Question
    + Random sample of mothers of newborns. Compare:
        + (A) Birth weights of babies of mothers who smoked during pregnancy
        + (B) Birth weights of babies of mothers who didn’t smoke
    + Question: Could the difference be due to chance alone?
+ Comparing Two Samples
    + Compare values of sampled individuals in Group A with values of sampled individuals in Group B.
    + Question: Do the two sets of values come from the same underlying distribution?
    + Answering this question by performing a statistical test is called A/B testing.

+ The Groups and the Question
    + Random sample of mothers of newborns. Compare:
        + (A) Birth weights of babies of mothers who smoked during pregnancy
        + (B) Birth weights of babies of mothers who didn’t smoke
    + Question: Could the difference be due to chance alone?

+ Hypotheses
    + __Null__: <br/>
        In the population, the distributions of the birth weights of the babies in the two groups are the same. (They are different in the sample just due to chance.)
    + __Alternative__: <br/>
        In the population, the babies of the mothers who didn’t smoke were heavier, on average, than the babies of the smokers.

+ Test Statistic
    + Group A: smokers
    + Group B: non-smokers
    + Statistic: Difference between average weights <br/> $\text{Group B average} - \text{Group A average}$
    + Large values of this statistic favor the alternative

+ Simulating Under the Null

    + If the null is true, all rearrangements of the birth weights among the two groups are equally likely - __permutation test__
    + Plan:
        + Shuffle all the birth weights
        + Assign some to “Group A” and the rest to “Group B”, maintaining the two sample sizes
        + Find the difference between the averages of the two shuffled groups
        + Repeat



### Video

<a href="https://edx-video.net/BERD82FD2018-V002500_DTH.mp4" alt="Lec 8.2 Hypotheses and Statistic" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 8.3 Performing the Test

### Notes

+ Demo
    ```python
    weights = smoking_and_birthweight.select('Birth Weight')
    weights

    weights.sample(with_replacement=False)

    shuffled_weights = weights.sample(with_replacement=False).column(0)

    original_and_shuffled = smoking_and_birthweight.with_column(
        'Shuffled Birth Weight', shuffled_weights
    )

    original_and_shuffled.group('Maternal Smoker', np.average)

    group_labels = baby.select('Maternal Smoker')

    # Procedure: 
    # 1. array of shuffled weights
    # 2. table with shuffled weights assigned to group labels
    # 3. array of means of the two groups
    # 4. difference between means of the two groups
    shuffled_weights = weights.sample(with_replacement=False).column(0)
    shuffled_tbl = group_labels.with_column('Shuffled Weight', shuffled_weights)
    means = shuffled_tbl.group('Maternal Smoker', np.average).column(1)
    new_difference = means.item(0) - means.item(1)
    # -1.1400795283148284 -> randomly

    differences = make_array()

    for i in np.arange(5000):
        shuffled_weights = weights.sample(with_replacement = False).column(0)
        shuffled_tbl = group_labels.with_column('Shuffled Weight', shuffled_weights)
        means = shuffled_tbl.group('Maternal Smoker', np.average).column(1)
        new_difference = means.item(0) - means.item(1)
        differences = np.append(differences, new_difference)

    Table().with_column('Difference Between Means', differences).hist(bins=20, ec='w')

    observed_difference
    # 9.266142572024918
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002800_DTH.mp4" alt="Lec 8.3 Performing the Test" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>




## Reading and Practice for Section 8a

### Reading


### Practice



