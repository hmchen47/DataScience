# Section 1: Center and Spread (Lec 1.1 - Lec 1.4)

+ [Launching Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/7bba8d29a20946e5be64e508fd3481b2/15f6f5cf64944c379af5a6b1f12f8af4/)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/37b80bfacc52ea5dfdad124579807188/notebooks/materials-x18/lec/x18/3/lec1.ipynb)
+ [Local Notebook](./notebook/lec1.ipynb)
+ [Local Python](./notebook/lec1.py)

## Lec 1.1 Introduction

### Notes

+ Prediction
  + One of the major responsibility of data science
  + To predict the value of one or more variables given the values of other related variables
  + Distributions: empirical, probability, data, etc.

+ Goals
  + Quantify natural concepts like "center" and "variability"
  + Examine bell shaped distributions
  + Understand why many of the empirical distributions that we have generated are bell shaped

+ The Average (or Mean) <br/>
    Data: 2, 3, 3, 9 -> Average = $(2+3+3+9)/4 = 4.25$
    + Need not be a value in the collection
    + Need not be an integer even if the data are integers
    + Somewhere between min and max, but not necessarily halfway in between
    + Same units as the data
    + Smoothing operator: collect all the contributions in one big pot, then split evenly

+ Relation to the Histogram
  + The average of a list depends only on the __proportions__ in which the distinct values appear, not on the number of entries in the list
  + The average is the __center of gravity__ of the histogram
  + The point on the horizontal axis where the histogram balances

+ Demo
  ```python
  values = make_array(2, 3, 3, 9)
  sum(values) / len(values), np.average(values), np.mean(values)
  # (4.25, 4.25, 4.25

  (2 + 3 + 3 + 9) / 4                 # 4.25
  2 * (1/4) + 3 * (2/4) + 9 * (1/4)   # 4.25
  2 * 0.25 + 3 * 0.5 + 9 * 0.25       # 4.25

  values_table = Table().with_columns('Value', values)
  values_table                        # Value: 2, 3, 3, 9

  bins_for_display = np.arange(0.5, 10.6, 1)

  values_table.hist(bins = bins_for_display, ec = 'w')

  2 * np.ones(10)   # array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

  twos = 2 * np.ones(10)
  threes = 3 * np.ones(20)
  nines = 9 * np.ones(10)

  new_values = np.append(np.append(twos, threes), nines)

  len(new_values)     # 40

  new_values_table = Table().with_column('Value', new_values)
  new_values_table.hist(bins = bins_for_display)

  np.average(new_values), np.average(values)
  # (4.25, 0.3909710391822828)
  ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000300_DTH.mp4" alt="Lecture 1.1 Introduction" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 1.2 Average and Median

### Notes

+ Discussion Question
    + Create a data set that has this histogram. (You can do it with a short list of whole numbers.)
    + What are its median and mean?
    <a href="url">
        <br/><img src="./diagrams/lec1-1.PNG" alt="text" width="300">
    </a>
      + Answer: data set = {1, 2, 2, 3, 3, 3, 4, 4, 5} -> median = average = 3
    + Are the medians of these two distributions the same or different? Are the means the same or different? If you say “different,” then say which one is bigger.
    <a href="url">
        <br/><img src="./diagrams/lec1-1.PNG" alt="text" width="300">
    </a>
    <a href="url">
        <img src="./diagrams/lec1-3.PNG" alt="text" width="300">
    </a>
      + Answer: List 1 = {1, 2, 2, 3, 3, 3, 4, 4, 5}, List 2 = {1, 2, 3, 3, 3, 4, 4, 10}; same median = 3, different average -> 2nd List bigger

+ Comparing Mean and median
    + __Mean__: Balance point of the histogram
    + __Median__: Half-way point of data; half the area of histogram is on either side of median
    + If the distribution is symmetric about a value, then that value is both the average and the median.
    + If the histogram is skewed, then the mean is pulled away from the median in the direction of the tail.

+ Discussion Question <br/>
    Which is bigger?
    1. mean
    2. median
    <a href="url">
        <br/><img src="./diagrams/lec1-4.png" alt="text" width="450">
    </a>
    + Answer:  meaian bigger - average is pulled away from the median to the left

+ Demo
    ```python
    nba = Table.read_table('nba2013.csv')
    # Name	Position	Height	Weight	Age in 2013
    # DeQuan Jones	Guard	80	221	23
    # Darius Miller	Guard	80	235	23
    # Trevor Ariza	Guard	80	210	28
    # James Jones	    Guard	80	215	32
    # ...(rows omitted)

    nba.hist('Height', bins=np.arange(65.5, 90.5), ec='w')

    heights = nba.column('Height')
    percentile(50, heights), np.average(heights)
    # (80, 79.06534653465347)
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000100_DTH.mp4" alt="Lec 1.2 Average and Median" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 1.3 Standard Deviation

### Notes

+ Defining Variability <br/>
    Plan A: “biggest value - smallest value”
    + Doesn’t tell us much about the shape of the distribution

    Plan B: 
    + Measure variability around the mean
    + Need to figure out a way to quantify this

+ How Far from the Average?
    + Standard deviation (SD) measures roughly how far the data are from their average
    + __SD = root (5) mean (4) square (3) of deviations (2) from average (1)__ - (#) indicates the calculation step
    + SD has the same units as the data

+ Why Use the SD? - There are two main reasons.
    + The first reason: No matter what the shape of the distribution, the bulk of the data are in the range “average ± a few SDs”
    + The second reason: Relation with bell shaped curves - Coming up in the next lecture.

+ Demo
    ```python
    sd_table = Table().with_columns('Value', values)
    # Value: 2, 3, 3, 9
    
    average = np.average(values)            $# 4.25
    
    deviations = values - average
    sd_table = sd_table.with_column('Deviation', deviations)
    # (Value, Deviation): (2, -2.25), (3, -1.25), (3, -1.25), (9, 4.75)
    
    sum(deviations)                         # 0.0

    sd_table = sd_table.with_column('Squared Deviation', deviations ** 2)
    # (Value, Deviation, Squared Deviation) = (2, -2.25, 5.0625), 
    #   (3, -1.25, 1.5625), (3, -1.25, 1.5625), (9, 4.75, 22.5625)

    # Variance of the data is the average of the squared deviations
    variance = np.average(sd_table.column('Squared Deviation'))     # 7.6875

    # Standard Deviation (SD) is the square root of the variance
    sd = variance ** 0.5            # 2.7726341266023544
    
    np.std(values)                  # 2.7726341266023544
    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000200_DTH.mp4" alt="Lec 1.3 Standard Deviation" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 1.4 Chebyshev's Bounds

### Notes

+ The Mathematician’s Name
    + Chebyshev
    + Chebychev
    + Chebishov
    + Čebyšev
    + Tchebichev
    + Tchebicheff
    + Tschebyscheff
    + Tschebyschew
    + Чебышёв


+ How Big are Most of the Values?
    + No matter what the shape of the distribution, the bulk of the data are in the range “average ± a few SDs”
    + Chebyshev’s Inequality <br/>
        No matter what the shape of the distribution, the proportion of values in the range “average ± z SDs” is at least $1 - 1/z^2$

+ Chebyshev’s Bounds - __No matter what the distribution looks like__

    | Range | Proportion |
    |-------|------------|
    | average ± 2 SDs | at least $1 - 1/4 (75\%)$ |
    | average ± 3 SDs | at least $1 - 1/9 (88.888...\%)$ |
    | average ± 4 SDs | at least $1 - 1/16 (93.75\%)$ |
    | average ± 5 SDs | at least $1 - 1/25 (96\%)$ |

+ Demo
    ```python
    births = Table.read_table('baby.csv')
    # Birth     Gestational     Maternal    Maternal    Maternal            Maternal
    # Weight    Days            Age         Height      Pregnancy Weight    Smoker
    # 120       284             27          62          100                 False
    # 113       282             33          64          135                 False
    # 128       279             28          64          115                 True
    # ... (rows omitted)

    births.hist('Maternal Pregnancy Weight')

    mpw = births.column('Maternal Pregnancy Weight')
    average = np.average(mpw)           # 128.4787052810903
    sd = np.std(mpw)                    # 20.72544970428041

    within_3_SDs = births.where('Maternal Pregnancy Weight', are.between(average - 3*sd, average + 3*sd))

    within_3_SDs.num_rows / births.num_rows     # 0.9863713798977853

    # Chebyshev's bound for the proportion in the range "average plus or minus 3 SDs" is at least
    1 - 1/3**2      # 0.8888888888888888

    births.hist(overlay = False)

    # See if Chebyshev's bounds work for different shapes of distributions
    for k in births.labels:
        values = births.column(k)
        average = np.average(values)
        sd = np.std(values)
        print()
        print(k)
        for z in np.arange(2, 6):
            chosen = births.where(k, are.between(average - z*sd, average + z*sd))
            proportion = chosen.num_rows / births.num_rows
            percent = round(proportion * 100, 2)
            print('Average plus or minus', z, 'SDs:', percent, '%')

    # Birth Weight
    # Average plus or minus 2 SDs: 94.89 %      75.00%
    # Average plus or minus 3 SDs: 99.57 %      88.88%
    # Average plus or minus 4 SDs: 100.0 %      93.75%
    # Average plus or minus 5 SDs: 100.0 %      96.00%
    # 
    # Gestational Days
    # Average plus or minus 2 SDs: 93.78 %
    # Average plus or minus 3 SDs: 98.64 %
    # Average plus or minus 4 SDs: 99.57 %
    # Average plus or minus 5 SDs: 99.83 %
    # 
    # Maternal Age
    # Average plus or minus 2 SDs: 94.89 %
    # Average plus or minus 3 SDs: 99.91 %
    # Average plus or minus 4 SDs: 100.0 %
    # Average plus or minus 5 SDs: 100.0 %
    # 
    # Maternal Height
    # Average plus or minus 2 SDs: 97.19 %
    # Average plus or minus 3 SDs: 99.66 %
    # Average plus or minus 4 SDs: 99.91 %
    # Average plus or minus 5 SDs: 100.0 %
    # 
    # Maternal Pregnancy Weight
    # Average plus or minus 2 SDs: 95.06 %
    # Average plus or minus 3 SDs: 98.64 %
    # Average plus or minus 4 SDs: 99.49 %
    # Average plus or minus 5 SDs: 99.91 %
    # 
    # Maternal Smoker
    # Average plus or minus 2 SDs: 100.0 %
    # Average plus or minus 3 SDs: 100.0 %
    # Average plus or minus 4 SDs: 100.0 %
    # Average plus or minus 5 SDs: 100.0 %


    ```

### Video

<a href="https://edx-video.net/BERD83FD2018-V000400_DTH.mp4" alt="Lec 1.4 Chebyshev's Bounds" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading

This guide assumes that you have watched the videos for Section 1.

This corresponds to textbook section:

[Chapter 14.1: Properties of the Mean ](https://www.inferentialthinking.com/chapters/14/1/Properties_of_the_Mean)

[Chapter 14.2: Variability](https://www.inferentialthinking.com/chapters/14/2/Variability)

In section 1, we studied averages (or means) and what we can say about them with only minimal assumptions about underlying populations. In histograms, the mean acts as the center of gravity or balance point of that histogram. In comparison, the median is the halfway point of the data (i.e. half of the area of the histogram is on either side of the median). We measure variability around the mean and use standard deviations (SDs) to quantify this measurement. Finally, we learned about Chebyshev's Inequality.

Sample means, medians, and standard deviations are extensively used in data science. It's important that you understand these concepts! Try the following practice questions to test your understanding.

### Practice

For the histograms below, is the mean to the left, to the right of the median, or will the mean and the median probably overlap?

<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/7bba8d29a20946e5be64e508fd3481b2/0d318cf40b2f41619d7d47ea519646a3/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%409e9f73070907479fa9f1c4fb55297106">
    <br/><img src="https://prod-edxapp.edx-cdn.org/assets/courseware/v1/84e581b5814e121f07263bfbb36953b2/asset-v1:BerkeleyX+Data8.3x+2T2018+type@asset+block/histogram1_s1.png" alt="Histogram with majority of data points on the lower end of the histogram and therefore very few data points that are higher. The histogram is skewed right." width="300">
</a>

    a. Mean to the right of the median
    b. Mean to the left of the median
    c. Mean and median will probably overlap

    Ans: a

<a href="https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.3x+2T2018/courseware/7bba8d29a20946e5be64e508fd3481b2/0d318cf40b2f41619d7d47ea519646a3/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.3x%2B2T2018%2Btype%40vertical%2Bblock%409e9f73070907479fa9f1c4fb55297106">
    <br/><img src="https://prod-edxapp.edx-cdn.org/assets/courseware/v1/770194acf97fe0d494fb32e3a89024a4/asset-v1:BerkeleyX+Data8.3x+2T2018+type@asset+block/histogram2_s1.png" alt="Histogram with majority of data points on the higher end of the histogram and therefore very few data points that are smaller. The histogram is skewed left." width="300">
</a>

    a. Mean to the right of the median
    b. Mean to the left of the median
    c. Mean and median will probably overlap

    Ans: b

Calculating the standard deviation of a data set can be done in five steps. Given a list of numbers called any_numbers, fill in the remaining steps to calculate the standard deviation for the list any_numbers. 

Your result of step 5 should be the standard deviation of any_numbers. We filled in step 1 and step 2 for you.

Step 1: Calculate the average of any_numbers (i.e. use np.mean(any_numbers)).

Step 2: Calculate the deviations by subtracting the average (step 1) from each number in the list any_numbers (i.e. any_numbers - average)


Step 3:

    Ans: Calculate the square of the deviations

Step 4:

    Ans: Calculate the average of the result(s) of step 3

Step 5:

    Ans: Calculate th positive square root of the result(s) of step 4




