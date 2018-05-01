# Section 12: Table Examples (Lec 12.1 - Lec 12.4)

## Lec 12.1 Table Method Review

### Notes

+ Important [Table Methods](http://data8.org/datascience/tables.html)
    + `t.select(column, …)` or `t.drop(column, …)`
    + `t.take([row, …])` or `t.exclude([row, …])`
    + `t.sort(column, descending=False, distinct=False)`
    + `t.where(column, are.condition(...))`
    + `t.apply(function, column, …)`
    + `t.group(column)` or `t.group(column, function)`
    + `t.group([column, …])` or `t.group([column, …], function)`
    + `t.pivot(cols, rows)` or `t.pivot(cols, rows, vals, function)`
    + `t.join(column, other_table, other_table_column)`
+ Demo
    ```python
    # table creation
    drinks = Table(['Drink', 'Cafe', 'Price']).with_rows([
        ['Milk Tea', 'Tea One', 4],
        ['Espresso', 'Nefeli',  2],
        ['Coffee',    'Nefeli', 3],
        ['Espresso', "Abe's",   2]
    ])
    discounts = Table().with_columns(
        'Coupon % off', make_array(5, 50, 25),
        'Location', make_array('Tea One', 'Nefeli', 'Tea One')
    )
    ```
+ Reference: [data science module](http://data8.org/datascience/tables.html)

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/tGQfKdCISbA){:target="_blank"}


## Lec 12.2 Discussion Question

### Notes

+ Discussion Question
    + Generate a table with one row per cafe that has the name and discounted price of its cheapest discounted drink
    + drinks

        | Drink | Cafe | Price |
        |-------|------|-------|
        | Milk Tea | Tea One | 4 |
        | Espresso | Nefeli | 2 |
        | Coffee | Nefeli | 3 |
        | Espresso | Abe's | 2 |
    + discounts

        | Coupon | Location |
        |--------|----------|
        | 5% | Tea One |
        | 50% | Nefeli |
        | 25% | Tea One |
    + Cheapest 

        | Cafe | Drink | Discounted Price |
        |------|-------|------------------|
        | Nefeli | Espresso | 1 |
        | Tea One | Milk Tea | 3 |

+ Answer:
    ```python
    a = drinks.join('Cafe', discounts, 'Location')
    a = a.with_column('Discounted Price', a.column(2) * (1 - a.column(3)/100) )
    a = a.drop(2, 3)
    a.sort('Discounted Price').sort('Cafe', distinct=True) # Correct, Espresso is cheaper
    a.group('Cafe', min) # Incorrect, Coffee is first alphabetically
    a.group('Cafe', list) # display the list with  each Cafe
    a.sort('Discounted Price').sort('Cafe', distinct=True)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/79W7XQHnWxo){:target="_blank"}


## Lec 12.3 Midterm Question

### Notes

+ Spring 2016 Midterm, Q2(b)

    Each row of the `trip` table from lecture desceibes a single bicycle rental in the San Francisco area. Durations are integers representing times in seconds,  The first three rows out of 338343 appear below.

    | Start | End | Duration |
    |-------|-----|----------|
    | Ferry Building | SF Caltrain | 765 |
    | San Antonio Shopping Center | Mountain View City Hall | 1036 |
    | Post at Kearny | 2nd at South Part | 307 |

    Write a Python expression below each of the following descriptions that computes its value.  The first one is provided for you.  You _may_ use uo to lines and introduce variables.

    + The average duration of rental
        ```python
        total_duration = sum(trip.column(2))
        total_duration / trip.num_rows
        ```
    + The name of the station where the most rentals ended (assume no ties)
    + The number of stations for which the average duration ending at that station was more than 10 minutes.

+ Answers:
    ```python
    trip = Table.read_table('trip.csv').where('Duration', are.below(1800)).select(3, 6, 1).relabeled(0, 'Start').relabeled(1, 'End')
    trip.show(3)

    # The name of the station where the most rentals ended (assume no ties).
    trip.group('End').sort('count', descending=True).row(0).item(0)

    # The number of stations for which the average duration ending at that station was more than 10 minutes.
    trip.drop('Start').group('End', np.average).where(2, are.above(10*60)).num_rows
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/4ljo9LqtmYI){:target="_blank"}


## Lec 12.4 Advanced Where

### Notes

+ Comparison Operators  
The result of a comparison expresson is a `bool` value
    + Assignment statements: `x = 2`; `y = 3`
    + Comparison expressions: `x > 1`, `x > y`, `y >= 3`, `x == y`, `x != 2`, `2 < x < 5`
    + `t.where(ary_of_bool_vals)` returns a table with only the rows of `t` for which the corresponding `bool` is `True`
+ Demo
    ```python
    # As of Jan 2017, this census file is online here: 
    # http://www2.census.gov/programs-surveys/popest/datasets/2010-2015/national/asrh/nc-est2015-agesex-res.csv

    full_census_table = Table.read_table('nc-est2015-agesex-res.csv')
    partial = full_census_table.select('SEX', 'AGE', 'POPESTIMATE2010', 'POPESTIMATE2015')
    us_pop = partial.relabeled(2, '2010').relabeled(3, '2015')

    us_pop.where('AGE', 70)
    us_pop.where('AGE', 70).where([False, True, True])
    seventy = us_pop.where('AGE', 70)

    # advanced where
    seventy.column('2010') < 2000000
    seventy.where(seventy.column('2010') < 2000000)
    us_pop.column('2015') / us_pop.column('2010') > 1.5
    us_pop.where(us_pop.column('2015') / us_pop.column('2010') > 1.5)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/nUZOdd-w8-s){:target="_blank"}


## Reading and Practice for Section 12

### Reading


### Practice



