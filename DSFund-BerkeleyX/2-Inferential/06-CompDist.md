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

<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.2 Total Variation Distance

### Notes

+ Demo
    ```python

    ```

### Video


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.3 Assessment

### Notes

+ Demo
    ```python

    ```

### Video


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 6.4 Summary

### Notes

+ Demo
    ```python

    ```

### Video


<a href="url" alt="text" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice for Section 6

### Reading



### Practice



