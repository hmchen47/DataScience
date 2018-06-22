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

+ Before the Randomization
    + In the population there is one imaginary ticket for each of the 31 participants in the experiment.
    + Each participant’s ticket looks like this: (Potential Outcome)
        + Treatment group: outcome if assigned to treatment group
        + Control group: outcome if assigned to control group

+ The Data 
    + 16 randomly picked tickets show: Outcome if assigned to control group
    + The remaining 15 tickets show: Outcome if assigned to treatment group

+ The Hypotheses
    + Null: <br/>
        The distribution of all 31 potential control scores is the same as the distribution of all 31 potential treatment scores.
    + Alternative: <br/>
        The distribution of all 31 potential control scores is different from the distribution of all 31 potential treatment scores.

+ Demo
    ```python
    observed_outcomes = Table.read_table('observed_outcomes.csv')
    # Group         Outcome if assigned treatment   Outcome if assigned control
    # Control       Unknown                         1
    #   ...
    # Treatment     1                               Unknown
    # Treatment     1                               Unknown
    #   ...
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003100_DTH.mp4" alt="Lec 9.2 Hypotheses" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 9.3 Test Statistic

### Notes

+ Test Statistic
    + Compare the proportions of 1's in the two groups
    + The alternative says the distributions in the two groups are different
    + Test statistic:
        + __Distance__ between the two proportions
        + $ | \text{control proportion} - \text{treatment proportion} |$
    + Large values of the statistic favor the alternative
+ Demo
    ```python
    obs_proportions = bta.group('Group', np.average).column(1)
    # array([0.125, 0.6  ])

    observed_distance = abs(obs_proportions.item(0) - obs_proportions.item(1))
    # 0.475
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003000_DTH.mp4" alt="Lec 9.3 Test Statistic" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 9.4 Performing a Test

### Notes

+ Demo
    ```python
    labels = bta.select('Group')
    results = bta.select('Result')

    shuffled_results = results.sample(with_replacement=False).column(0)
    shuffled_tbl = labels.with_column('Shuffled Result', shuffled_results)
    proportions = shuffled_tbl.group('Group', np.average).column(1)
    new_distance = abs(proportions.item(0) - proportions.item(1))
    # 0.041666666666666685 -> random result

    distances = make_array()
    for i in np.arange(20000):
        shuffled_results = results.sample(with_replacement=False).column(0)
        shuffled_tbl = labels.with_column('Shuffled Result', shuffled_results)
        proportions = shuffled_tbl.group('Group', np.average).column(1)
        new_distance = abs(proportions.item(0) - proportions.item(1))
        distances = np.append(distances, new_distance)

    Table().with_column('Distance', distances).hist(bins=np.arange(0, 1, 0.1), ec='w')
    plots.scatter(observed_distance, 0, color='red', s=40);

    np.count_nonzero(distances >= observed_distance) / 20000
    # 0.0074
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003200_DTH.mp4" alt="Lec 9.4 Performing a Test" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice for Section 9

### Reading

This guide assumes that you have watched the videos for Section 9.

This corresponds to textbook section:

[Chapter 12.3: Causality](https://www.inferentialthinking.com/chapters/12/3/causality.html)

In section 9, we learned that our methods for comparing two samples have a powerful use in the analysis of randomized controlled experiments. Since the treatment and control groups are assigned randomly in randomized controlled experiments, differences in their outcomes can be compared to what would happen just due to chance if the treatment had no effect at all. Within a randomized controlled experiment, if all the observed differences are more marked than what we would predict as purely due to chance, we will have evidence of causality.

Complete the practice questions below to refresh your understanding of causality.

### Practice

In 2013, a Berkeley chemical engineering professor, Jay Keasling, discovered a synthetic version of artemisinin - a chemical crucial to producing antimalarial drugs. A group of researchers and statisticians come together to test the effectiveness of this new chemical, which is being used to treat malaria in developing countries.

A random sample of 500 malaria patients are randomly assigned to take either the antimalarial drug or a placebo pill. It turns out that among the patients who took the drug, a much larger proportion recovered from malaria than among those who took the placebo. Meanwhile, cases of malaria are decreasing in developing nations, and some suggest that this might be due to the increased effectiveness of mosquito nets being distributed.

Q1. True of False: This is a randomized controlled experiment.

    Ans: True


Q2. True of False: The experiment indicates that taking the drug causes an increase in recovery rates among malaria patients.

    Ans: True

Q3. True of False: The improvements in mosquito nets are a confounding factor in this experiment.

    Ans: True






