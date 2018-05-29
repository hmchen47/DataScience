# Section 5: Hypothesis Testing (Lec 5.1 - Lec 5.4)

## Lec 5.1 Assessing Models

### Notes

+ Choosing One of Two Viewpoints

    + Based on data
        + “Chocolate has no effect on cardiac disease.” --> “Yes, it does.”
        + “This jury panel was selected at random from eligible jurors.” --> “No, it has too many people with college degrees.”

+ Models
    + A model is a set of assumptions aboyut the data
    + In data science, many models involve assumption about processes that involve randomnes, e.g., “Chance models”

+ Approach to Assessment

    + If we can simulate data according to the assumptions of the model, we can learn what the model predicts.
    + We can then compare the predictions to the data that were observed.
    + If the data and the model’s predictions are not consistent, that is evidence against the model.

### Video 

<a href="https://youtu.be/wJ9Eov9Mdf0" alt="Lec 5.1 Assessing Models" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>



## Lec 5.2 A Model about Random Selection

### Notes

+ Swain vs. Alabama, 1965

    + Talladega County, Alabama
    + Robert Swain, black man convicted of crime
    + Appeal: one factor was all-white jury
    + Only men 21 years or older were allowed to serve
    + 26% of this population were black
    + Swain’s jury panel consisted of 100 men
    + 8 men on the panel were black

+ Supreme Court Ruling

    + About disparities between the percentages in the eligible population and the jury panel, the Supreme Court wrote:
        > “... the overall percentage disparity has been small and reflects no studied attempt to include or exclude a specified number of Negroes”
    + The Supreme Court denied Robert Swain’s appeal

+ Sampling from a Distribution

    + Sample at random from a categorical distribution <br/>
        `sample_proportions(sample_size, pop_distribution)`
    + Samples at random from the population
    + Returns an array containing the distribution of the categories in the sample

+ `sample_porportions` function
    + Signature: `sample_proportions(sample_size, probabilities)`
    + Return the proportion of random draws for each outcome in a distribution.
    + Args: 
        + `sample_size`: The size of the sample to draw from the distribution.
        + `probabilities`: An array of probabilities that forms a distribution.

+ Demo 
    ````python
    # proportion of two categories
    eligible_population = make_array(0.26, 0.74)
 
    sample_proportions(100, eligible_population)

    both_counts = 100 * (sample_proportions(100, eligible_population))
 
    both_counts.item(0)

    counts = make_array()

    repetitions = 10000
    for i in np.arange(repetitions):
        sample_distribution = sample_proportions(100, eligible_population)
        sampled_count = (100 * sample_distribution).item(0)
        counts = np.append(counts, sampled_count)

    Table().with_column('Random Sample Count', counts).hist(bins = np.arange(5.5, 44.5, 1))
    ```

### Video 

<a href="https://youtu.be/OreWRDOb9fg" alt="Lec 5.2 A Model about Random Selection" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>



## Lec 5.3 A Genetic Model

### Notes

+ Demo 
    ````python

    ```

### Video 

<a href="https://youtu.be/OI4x1i_0kPU" alt="Lec 5.3 A Genetic Model" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>



## Lec 5.4 Example

### Notes

+ Demo 
    ````python

    ```

### Video 

<a href="urhttps://youtu.be/ybDvLbRR4UAl" alt="Lec 5.4 Example" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>



## Reading and Practice for Section 5

### Reading


### Practice 



