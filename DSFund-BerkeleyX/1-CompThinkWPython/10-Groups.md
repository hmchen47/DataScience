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

+ Grouping by Multiple Columns
    + The `group` method can also aggregate all rows that share the combination of values in multiple columns
    + 1st argument: a list of which columns to group by
    + 2nd argument: (optional) how to combine values
+ Demo
    ```python
    all_cones.group('Flavor')
    all_cones.group(['Flavor', 'Color'])
    all_cones.group(['Flavor', 'Color'], max)

    nba.drop(0).group(['TEAM', 'POSITION'], np.average)
    nba.drop(0, 2).group('POSITION', np.average)

    full_table = Table.read_table('educ_inc.csv')
    ca_2014 = full_table.where('Year', are.equal_to('1/1/14 0:00')).where('Age', are.not_equal_to('00 to 17')).drop(0).sort('Population Count')
    no_ages = ca_2014.drop(0)
    no_ages.group([0, 1, 2], sum)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/9NGa2MaDPxw){:target="_blank"}


## Lec 10.3 Example 1

### Notes

+ Which NBA teams spent the most on their "starters" in 2015-2016?  
  Assume the "starter" for a team & position in the player with the highest salary on that team in that position
  + Ans: 
    ```python
    starter_salaries = nba.drop(0).group(['TEAM', 'POSITION'], max)
    starter_salaries.drop(1).group('TEAM', sum).sort(1, descending=True)
    ```

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



