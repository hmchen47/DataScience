# Section 11: Interpreting Confidence (Lec 11.1 - Lec 11.3)

+ [Launch Web Page](https://courses.edx.org/courses/course-v1:BerkeleyX+Data8.2x+1T2018/courseware/95ee24be1f714d51bb48d73712c71aba/c1392ee86ecc4e6ca55908f2d3f9242c/1?activate_block_id=block-v1%3ABerkeleyX%2BData8.2x%2B1T2018%2Btype%40vertical%2Bblock%40ece536653f1143a78cbee4c9858ef8cb)
+ [Web Notebook](https://hub.data8x.berkeley.edu/user/59d217c894d11dbd21d2d37ef6ae9675/notebooks/materials-x18/lec/x18/2/lec11.ipynb#)
+ [Local Notebook](./notebooks/lec11.ipynb)
+ [Local Python](./notebooks/lec11.py)

## Lec 11.1 Applying the Bootstrap

### Notes

+ When to Find a Confidence Interval
    + You wan to guess a parameter of a population.
    + You have a random sample from the population.
    + You want to quantify your uncertainty.
    + A statistic is a reasonable estimate of the parameter.

+ Demo
    ```python
    births = Table.read_table('baby.csv')
    births.show(3)
    # Birth     Gestational     Maternal    Maternal    Maternal            Maternal 
    # Weight    Days            Age         Height      Pregnancy Weight    Smoker
    # 120       284             27          62          100                 False
    # 113       282             33          64          135                 False
    # 128       279             28          64          115                 True

    babies = births.select('Birth Weight', 'Gestational Days')
    # Birth Weight    Gestational Days
    # 120             284
    # 113             282
    # ...

    babies.scatter(1, 0)

    ratios = babies.with_column(
        'Ratio BW/GD', babies.column(0)/babies.column(1)
    )
    # Birth Weight    Gestational Days    Ratio BW/GD
    # 120             284                 0.422535
    # 113             282                 0.400709
    # ...

    ratios.hist('Ratio BW/GD')

    np.median(ratios.column('Ratio BW/GD'))     # 0.42907801418439717

    resampled_medians = []
    for i in np.arange(1000):
        resample = ratios.sample()
        median = np.median(resample.column('Ratio BW/GD'))
        resampled_medians.append(median)

    interval_99 = [percentile(0.5, resampled_medians),
                percentile(99.5, resampled_medians)]
    print(interval_99)      # [0.4243514279485503, 0.43416370106761565]

    Table().with_column('Resampled median', resampled_medians).hist(0)
    plots.plot(interval_99, [0, 0], color='gold', lw=10);
    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003800_DTH.mp4" alt="Lec 11.1 Applying the Bootstrap" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.2 Confidence Interval Pitfalls

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V004000_DTH.mp4" alt="Lec 11.2 Confidence Interval Pitfalls" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Lec 11.3 Confidence Interval Tests

### Notes

+ Demo
    ```python

    ```

### Video

<a href="https://edx-video.net/BERD82FD2018-V003700_DTH.mp4" alt="Lec 11.3 Confidence Interval Tests" target="_blank">
  <img src="http://files.softicons.com/download/system-icons/windows-8-metro-invert-icons-by-dakirby309/png/64x64/Folders%20&%20OS/My%20Videos.png" alt="Video" width="60px"> 
</a>


## Reading and Practice

### Reading


### Practice





