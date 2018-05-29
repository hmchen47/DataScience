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


## Reading and Practice for Section

### Reading


### Practice




