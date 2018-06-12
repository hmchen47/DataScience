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

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002500_DTH.mp4" alt="Lec 8.2 Hypotheses and Statistic" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 8.3 Performing the Test

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V002800_DTH.mp4" alt="Lec 8.3 Performing the Test" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice for Section 8a

### Reading


### Practice



