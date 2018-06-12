# Section 9: Causality (Lec 9.1 - Lec 9.4)

## Lec 9.1 Introduction

### Notes

+ Randomized Controlled Experiment
    + Sample A: control group
    + Sample B: treatment group
    + If the treatment and control groups are selected at random, then you can make causal conclusions.
    + Any difference in outcomes between the two groups could be due to
        + chance
        + the treatment
+ Demo
    ```python
    bta = Table.read_table('bta.csv')

    bta.group('Group')
    # Group       count
    # Control     16
    # Treatment   15

    bta.group('Group', sum)
    # Group       Result sum
    # Control     2
    # Treatment   9

    bta.group('Group', np.average)
    # Group       Result average
    # Control     0.125
    # Treatment   0.6
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003300_DTH.mp4" alt="Lec 9.1 Introduction" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 9.2 Hypotheses

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003100_DTH.mp4" alt="Lec 9.2 Hypotheses" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 9.3 Test Statistic

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003000_DTH.mp4" alt="Lec 9.3 Test Statistic" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 9.4 Performing a Test

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003200_DTH.mp4" alt="Lec 9.4 Performing a Test" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice for Section 9

### Reading


### Practice




