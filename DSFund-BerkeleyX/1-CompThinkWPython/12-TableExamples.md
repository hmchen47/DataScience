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

    # table methods
    a = drinks.join('Cafe', discounts, 'Location')
    a = a.with_column('Discounted Price', a.column(2) * (1 - a.column(3)/100) )
    a = a.drop(2, 3)
    a.sort('Discounted Price').sort('Cafe', distinct=True) # Correct, Espresso is cheaper
    a.group('Cafe', min) # Incorrect, Coffee is first alphabetically
    ```
+ Reference: [data science module](http://data8.org/datascience/tables.html)

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/tGQfKdCISbA){:target="_blank"}


## Lec 12.2 Discussion Question

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/79W7XQHnWxo){:target="_blank"}


## Lec 12.3 Midterm Question

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/4ljo9LqtmYI){:target="_blank"}


## Lec 12.4 Advanced Where

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/nUZOdd-w8-s){:target="_blank"}


## Reading and Practice for Section 12

### Reading


### Practice



