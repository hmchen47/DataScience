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

+ Demo
    ```python
    top = Table.read_table('top_movies_2017.csv')
    top = top.with_column('Row Index', np.arange(top.num_rows)).move_to_start('Row Index')
    top.set_format(['Gross', 'Gross (Adjusted)'], NumberFormatter)

    top.take([3, 5, 7])

    top.where('Title', are.containing('and the'))

    start = np.random.choice(np.arange(10))
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




