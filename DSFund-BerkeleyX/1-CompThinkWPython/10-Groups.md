# Section 10a: Groups (Lec 10.1 - Lec 10.3)

## Lec 10.1 One Attribute Group

### Notes

+ Grouping by one Column
    + The `group` method aggregates all rows with the same value a column into a single row in the resulting table.
    + 1st argument: which column to group by
    + 2nd column: (optional) how to combine values
        + `len` - number of grouped values (default)
        + `list` - list of all grouped values
        + `sum` - total of all grouped values
+ Demo
    ```python
    all_cones = Table.read_table('cones.csv')
    cones = all_cones.drop('Color').exclude(5)

    cones.group('Flavor')
    cones.group('Flavor', list)
    cones.group('Flavor', len)
    cones.group('Flavor', min)

    min(cones.where('Flavor', 'chocolate').column('Price'))
    cones.group('Flavor', np.average)

    def data_range(x):
        return max(x) - min(x)
    cones.group('Flavor', data_range)

    nba = Table.read_table('nba_salaries.csv').relabeled(3, 'SALARY')
    teams_and_money = nba.select('TEAM', 'SALARY')
    teams_and_money.group('TEAM', sum).sort(1, descending=True)
    nba.group('TEAM', sum)
    position_and_money = nba.select('POSITION', 'SALARY')
    position_and_money.group('POSITION')
    position_and_money.group('POSITION', np.average)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/n0jAwei6zGY){:target="_blank"}


## Lec 10.2 Cross Classification

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/9NGa2MaDPxw){:target="_blank"}


## Lec 10.3 Example 1

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/8MZW99WJcXs){:target="_blank"}


## Reading and Practice for Section 10a

### Reading


### Practice



# Section 10b: Pivot (Lec 10.4 - Lec 10.6)

## Lec 10.4 Pivot Tables

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


## Lec 10.5 Example 2

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


## Lec 10.6 Comparing Distributions

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](){:target="_blank"}


## Reading and Practice for Section 10b

### Reading


### Practice



