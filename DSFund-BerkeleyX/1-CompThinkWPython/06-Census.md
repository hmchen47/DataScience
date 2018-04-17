# Section 6: Census (Lec 6.1 - Lec 6.4)

## Lec 6.1 Census

### Notes

+ The Decennial Census
    + Every ten years, the Census Bureau counts how many people there in the U.S.
    + In between censuses, the Bureau s=estimates how many people there are each year
    + Article 1, Section 2 of the Constitution:  
        "Representatives and direct Taxes shall be apportioned among the several States ... according to their respective Numbers ..."
+ Public Census Data
    + Analyzing census data can lead to the discovery of interesting features and trends in the population.
    + [Data](http://www2.census.org/programs-surveys)
+ Demo

    ```python
    data = 'http://www2.census.gov/programs-surveys/popest/datasets/2010-2015/national/asrh/nc-est2015-agesex-res.csv'

    partial = full.select('SEX', 'AGE', 4, 9)
    us_pop = partial.relabeled(2, '2010').relabeled(3, '2015')
    us_pop.set_format([2, 3], NumberFormatter)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/b29OrPn5ofw)

## Lec 6.2 Column Arithmetic

### Notes

+ Demo
    ```python
    us_pop.column('2015') - us_pop.column('2010')

    change = us_pop.column('2015') - us_pop.column('2010')
    census = us_pop.with_columns(
        'Change', change,
        'Total Growth', change / us_pop.column('2010')
    )
    census.set_format('Change', NumberFormatter)
    census.set_format('Total Growth', PercentFormatter)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 6.3 Accessing Values

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 6.4 Males and Females

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Reading and Practice for Section 6

### Reading


### Practice

