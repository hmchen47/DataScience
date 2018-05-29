# Sampling and Simulation

##  Section 4a: Sampling (Lec 4.1 - Lec 4.2)

## Lec 4.1 Probability & Sampling

### Notes

+ Discussion Question

    A population has 100 people, including Mo and Jo. We sample two people at random without replacement.

    (a) $P(\text{both Mo and Jo are in the sample}) = P(\text{first Mo, then Jo}) + P(\text{first Jo, then Mo}) = (1/100) * (1/99) + (1/100) * (1/99) = 0.0002 $

    (b) $P(\text{neither Mo nor Jo is in the sample}) = (98/100) * (97/99) = 0.9602$
 
### Video

<a href="https://youtu.be/VNeKoGu6T2A" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>



## Lec 4.2 Sampling

### Notes

+ Sampling

    + Deterministic sample:
        + Sampling scheme doesn’t involve chance
    + Probability sample:
        + Before the sample is drawn, you have to know the selection probability of every group of people in the population
        + Not all individuals have to have equal chance of being selected
    + Uniform random sample:
        + Each individual has an equal chance of being selected
        + A "simple random sample" is uniforma & without replacement

+ Sample of Convenience

    + Example: sample consists of whoever walks by
    + Just because you think you’re sampling “at random”, doesn’t mean you are.
    + If you can’t figure out ahead of time
        + what’s the population
        + what’s the chance of selection, for each group in the population 

        then you don’t have a random sample


+ `Table.sample` method
    + Signature: `Table.sample(k=None, with_replacement=True, weights=None)`
    + Return a new table where k rows are randomly sampled from the original table.
    + Args:
        + `k`: specifies the number of rows (`int`) to be sampled from the table. Default is k equal to number of rows in the table.
        + `with_replacement`: `bool`
        + `weights`: Array specifying probability the ith row of the table is sampled.
+ `Table.group` method
    + Signature: `Table.group(column_or_label, collect=None)`
    + Group rows by unique values in a column; count or aggregate others.
    + Args:
        + `column_or_label`: values to group (column label or index, or array)
        + `collect`: a function applied to values in other columns for each group

+ Demo
    ```python
    top = Table.read_table('top_movies_2017.csv')
    top = top.with_column('Row Index', np.arange(top.num_rows)).move_to_start('Row Index')
    top.set_format(['Gross', 'Gross (Adjusted)'], NumberFormatter)

    top.take([3, 5, 7])         # take sample sw/ specific rows

    top.where('Title', are.containing('and the'))   # get samples with criteria

    start = np.random.choice(np.arange(10)) # random samples from a given list
    top.take(np.arange(start, start + 5))

    top.sample(5)       

    top.sample(50).group("Title")

    top.sample(500).group('Title')

    top.sample(5, with_replacement=False)
    ```

### Video

<a href="https://youtu.be/YUA7fcT9sXU" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading

This guide assumes that you have watched the videos for Section 4a.

This corresponds to textbook sections:

[Chapter 10: Sampling and Empirical Distributions](https://www.inferentialthinking.com/chapters/10/sampling-and-empirical-distributions.html)

In section 4a, we learned about sampling. Random sampling will allow us to make conclusions about unknowns based on data. Know the difference between a deterministic sample, a probability sample, and an uniform random sample. Also be careful of samples of convenience.

Try to complete the following questions about sampling.

### Practice

The Data 8.2x staff needs your help to survey the Data 8.2x students. The Data 8.2x staff want to get some feedback for the course, but they cannot ask every single student because that would take too much time. The staff comes up with a few sampling methods to determine who to survey.

Given the following sampling techniques, are they deterministic, probability, or uniform random samples? Assume that there are 20,000 students and the staff has a list of all students.

Q1. True or False: surveying every 10th student on the list starting from the first student on the list will result in a determinstic sample.

    Ans: True

Q2. Choose a random number uniformly from 1 to 10 and survey 1000 consecutive students on the list starting from the random number. What type of sample will result from this surveying technique. Select all that apply.

    a. Deterministic Sample
    b. Probability Sample
    c. Uniform Random Sample

    Ans: b

Q3. Choose a random number uniformly from 1 to 10 and survey every 10th student on the list starting from the random number. True of False: each individual has the same chance of being chosen.

    Ans: True

Q4. Suppose for each student, we flip a fair coin. If that coin lands on heads, we survey the student. If tail, we do not survey the student. What is the chance that nobody is surveyed?

    a. $0$
    b. $1/2$
    c. $(1/2)^19999$
    d. $(1/2)^20000$
    e. $1 - (1/2)^20000$

    Ans: d



# Section 4b: Simulation (Lec 4.3 - Lec 4.6)

## Lec 4.3 Distributions

### Notes

+ Probability Distribution

    + Random quantity with various possible values
    + “Probability distribution”:
        + All the possible values of the quantity
        + The probability of each of those values
    + If you can do the math, you can work out the probability distribution can without ever simulating the random quantity

+ Empirical Distribution

    + “Empirical”: based on observations
    + Observations can be from repetitions of an experiment
    + “Empirical Distribution”
        + All observed values
        + The proportion of times each value appears

+ Demo
    ```python
    # Gerneate dice outcome set
    die = Table().with_column('face', np.arange(6)+1)

    # plot dice face historgram func
    def face_hist(t):
        t.hist('face', bins=np.arange(0.5, 7, 1), unit='face')
        plots.xlabel('Face')

    face_hist(die)

    # Try changing the sample size of 1000 to larger and smaller numbers
    face_hist(die.sample(1000))
    ```

### Video

<a href="https://youtu.be/f7z8QSovv10" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 4.4 Large Random Samples

### Notes

+ Law of Averages

    + If a chance experiment is repeated many times, independently and under the same conditions, then the proportion of times that an event occurs gets closer to the theoretical probability of the event
    + As you increase the number of rolls of a die, the proportion of times you see the face with five spots gets closer to 1/6

+ Empirical Distribution of a Sample

    If the sample size is large,

    then the empirical distribution of a uniform random sample

    resembles the distribution of the population,

    with high probability


+ Demo
    ```python
    united = Table.read_table('united.csv') # read united airline data

    def delay_hist(t):
        t.hist('Delay', unit='minute', bins=np.arange(-30, 301, 10))
        
    delay_hist(united)

    delay_hist(united.sample(1000))
    ```

### Video

<a href="https://youtu.be/z6tlWBbhEGM" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 4.5 Simulation

### Notes

+ Calculate & Simulate

    + Roll a fair diw 4 times
    + What is $P(\text{get at least one 6})$?

+ Demo
    ```python
    k = 4                               # a trial w/ 4 rolls
    dice = np.arange(6) + 1
    rolls = np.random.choice(dice, k)

    sum(rolls == 6)

    trials = 10000
    successes = 0

    for _ in np.arange(trials):
        rolls = np.random.choice(dice, k)
        if sum(rolls == 6) > 0:
            successes = successes + 1

    successes / trials
    ```

### Video

<a href="https://youtu.be/bZAH45VowH0" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 4.6 Statistics

### Notes

+ Why Sample?
    + Probability: 
        + a branch of mathematics where if you make assumptions about how the world works
        + compute what will happen when you run an experiment
    + Statistics: 
        + look at the outcome of an experiment and tries to reason about what the world is like
        + a method of understanding from observation
    + Sampling: tie the two together

+ Inference
    + __Statistical Inference__: Making conclusions based on data in random samples
    + Example: 
        + Use the data to guess the valueif an unknown (_fixed_) number
        + Create an __estimate__ (_depends on tehrandom sample_) of the unknown quantity

+ Terminology
    + __Parameter__: A number associated with the population
    + __Statistic__: A number calculated from the sample

    A statistic can be used as an __estimate__ of a parameter

+ Probability Distribution of a Statistic

    + Values of a statistic vary because random samples vary
    + “Sampling distribution” or “probability distribution” of the statistic:
        + All possible values of the statistic,
        + and all the corresponding probabilities
    + Can be hard to calculate
        + Either have to do the math
        + Or have to generate all possible samples and calculate the statistic based on each sample

+ Empirical Distribution of a Statistic

    + Empirical distribution of the statistic:
        + Based on simulated values of the statistic
        + Consists of all the observed values of the statistic,
        + and the proportion of times each value appeared
    + Good approximation to the probability distribution of the statistic, if the number of repetitions in the simulation is __large__

+ Demo
    ```python
    def estimate_by_simulation(trials):
        successes = 0

        for _ in np.arange(trials):
            rolls = np.random.choice(dice, k)
            if sum(rolls == 6) > 0:
                successes = successes + 1

        return successes / trials

    estimates = []
    for _ in np.arange(1000):
        estimates.append(estimate_by_simulation(10000))

    Table().with_column('Estimate', estimates).hist(bins=50, normed=False)
    ```

### Video

<a href="https://youtu.be/6FLXlYSa8NY" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading

This guide assumes that you learned more about sampling. Distributions represent events of chance. We learned about probability distributions and empirical distributions. The law of averages explains why the empirical distribution of a large random sample looks like the probability distribution. With our understanding of different distributions and the law of averages, we can use sampling and simulation to learn things about the world around us with limited data.

Be careful! Before you proceed, do you know the difference between a probability distribution and an empirical distribution? If you don't, try rewatching the previous lecture videos.

Try to complete the following questions. have watched the videos for Section 4b.

This corresponds to textbook sections:

[Chapter 10: Sampling and Empirical Distributions](https://www.inferentialthinking.com/chapters/10/sampling-and-empirical-distributions.html)

In section 4b, we learned more about sampling. Distributions represent events of chance. We learned about probability distributions and empirical distributions. The law of averages explains why the empirical distribution of a large random sample looks like the probability distribution. With our understanding of different distributions and the law of averages, we can use sampling and simulation to learn things about the world around us with limited data.

Be careful! Before you proceed, do you know the difference between a probability distribution and an empirical distribution? If you don't, try rewatching the previous lecture videos.

Try to complete the following questions.

### Practice

Below are four histograms that result from rolling a fair 6-sided die with values 1, 2, 3, 4, 5, and 6. 

Each of these histograms is either the probability histogram of one roll of the die or an empirical histogram of 10, 1000, or 100000 die rolls. What is your best guess of what each histogram represents? Please match the die roll with the corresponding resulting histogram. Try not to use a Jupyter notebook to solve this problem! 

Histogram 1: 

Histogram of 6-sided die rolls. Distribution of 6 values is not perfectly even, but also not very sparse. 

![hist1](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/f50e8137769e20c432e010da842f6597/asset-v1:BerkeleyX+Data8.2x+1T2018+type@asset+block/die1000.png)

Histogram 2: 

Histogram of 6-sided die rolls. Distribution of 6 values is perfectly even (all 6 outcomes exactly the same probability). 

![hist2](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/53598b1d3ba3979f7c0afd3e9827a5f3/asset-v1:BerkeleyX+Data8.2x+1T2018+type@asset+block/dieprob.png)

Histogram 3: 

Histogram of 6-sided die rolls. Distribution of 6 values is very sparse. 

~[hist3](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/b44839ede46fd66ad0ccf686a71ab6a9/asset-v1:BerkeleyX+Data8.2x+1T2018+type@asset+block/die10.png)

Histogram 4: 

Histogram of 6-sided die rolls. Distribution of 6 values is almost perfectly even.

![hist4](https://prod-edxapp.edx-cdn.org/assets/courseware/v1/28dc51f2aef766d759c5cd56ae5b1df6/asset-v1:BerkeleyX+Data8.2x+1T2018+type@asset+block/die100000.png)


Q1. Which histogram is the probabiliy histogram of 1 roll of the die?

    Ans: Histogram 2 

Q2. Which histogram is the empirical histogram of 10 rolls of the die?

    Ans: Histogram 3

Q3. Which histogram is the empirical histogram of 1,000 rolls of the die?

    Ans: Histogram 1

Q4. Which histogram is the empirical histogram of 100,000 rolls of the die?

    ANs: Histogram 4




