# Section 11: Joins (Lec 11.1 - Lec 11.4)

## Lec 11.1 Joins

### Notes

+ Joining Two Tables
    ```python
    tblA.join('col_tblA', tblB, 'col_tblB') # same values in col_1st_tbl & col_2nd_tbl
    ```
    + `tblA`: match rows in this table
    + `col_1st_tbl`: using values in this column
    + `tblB`: with rows in this table
    + `col_2nd_tb;`: using values in the `tblB`
    + Auto sort the matching rows of `col_tblA` and `col_tblB`
    + Result columns generated from both tables
    + Generate all possible rows with matched rows

    ![diagram](./Diagrams/sec11-joins.png)
+ Demo
    ```python
    drinks = Table(['Drink', 'Cafe', 'Price']).with_rows([
        ['Milk Tea', 'Tea One', 4],
        ['Espresso', 'Nefeli',  2],
        ['Latte',    'Nefeli',  3],
        ['Espresso', "Abe's",   2]
    ])

    discounts = Table().with_columns(
        'Coupon % off', make_array(25, 50, 5),
        'Location', make_array('Tea One', 'Nefeli', 'Tea One')
    )

    t = drinks.join('Cafe', discounts, 'Location')

    t.with_column('Discounted', t.column(2) * (1 - t.column(3)/ 100))
    two = drinks.join('Cafe', drinks)
    two.with_column('Total', two.column('Price') + two.column('Price_2'))
    ```

### Videos 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/2s0yP3wp3rI){:target="_blank"}


## Lec 11.2 Bikes

### Notes

+ Demo: Bikes
    ```python
    trips = Table.read_table('trip.csv')
    # some trips are very long and skew the histogram
    # only interested in the trips for commute within 30 mins
    commute = trips.where('Duration', are.below(1800))
    commute.hist('Duration') 

    commute.hist('Duration', bins=60, unit='second')    # 60 bins
    commute.hist('Duration', bins=np.arange(1801), unit='second') 
    # most commute btw [250, 550)

    starts = commute.group('Start Station').sort('count', descending=True)
    commute.pivot('Start Station', 'End Station')
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/-kJEI52bIUM){:target="_blank"}


## Lec 11.3 Shortest Trips

### Notes

+ Demo: Shortest Trips
    ```python
    duration = trips.select(3, 6, 1)
    shortest = duration.group([0, 1], min)
    from_cc = shortest.where(0, are.containing('Civic Center BART')).sort(2)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/KErtBTDpCQo){:target="_blank"}


## Lec 11.4 Maps

### Notes

+ Maps
    + A table containing columns of latitude and longitute values can be used to generate a map of markers
    ```python
    {Marker|Circle}.map_table(table, ...)
    ```
    + `{Marker|Circle}` : map w/ either `Marker` or `Circle`
    + `table`: 
        + Column 0: latitudes
        + Column 1: longitudes
        + Column 2: labels
        + Column 3: colors
        + Column 4: sizes
    + `...`: Applies to all features - color = 'blue', size=200, ...
+ Demo
    ```python
    stations = Table.read_table('station.csv')
    Marker.map_table(stations.select('lat', 'long', 'name'))
    sf = stations.where('landmark', 'San Francisco')
    Circle.map_table(sf.select('lat', 'long', 'name'), color='green', radius=150)

    # generate color table
    colors = stations.group('landmark').with_column(
        'color', make_array('blue', 'red', 'green', 'orange', 'purple'))
    colored = stations.join('landmark', colors).select('lat', 'long', 'name', 'color')
    Marker.map_table(colored)

    station_starts = stations.join('name', starts, 'Start Station') 
    Circle.map_table(station_starts.select('lat', 'long', 'name').with_columns(
        'color', 'blue',
        'area', station_starts.column('count') * 1000
    ))
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/NmvTEc7DjLk){:target="_blank"}

## Reading and Practice for Section 11

### Reading

This guide assumes that you have watched section 11 (video lecture segments Lec 11.1, Lec 11.2, Lec 11.3, Lec 11.4) in Courseware.

This corresponds to textbook sections:

+ [Chapter 8.4: Joining Tables by Columns](https://www.inferentialthinking.com/chapters/08/4/joining-tables-by-columns.html)
+ [Chapter 8.5: Bike Sharing in the Bay Area](https://www.inferentialthinking.com/chapters/08/5/bike-sharing-in-the-bay-area.html)

In section 11, we focused on table joins. The `join` table method is very useful when working with many sets of data. We were also introduced to some mapping capabilities from the `datascience` library.

Here is the description for join.

`tblA.join(colA, tblB, colB)` returns a table with the columns of tblA and tblB, containing rows for all values of a colA and colB that appear in both tables. 

Practice your understanding of join with the following practice problems.

### Practice

Let's revisit our original Data 8X marble store. Data 8X sells small bags of marbles in groups of different amounts. Each bag contains marbles of one color. Each row is a bag of marbles. Our table marbles is as follows:

marbles

| Color | Amount | Price ($) |
|-------|--------|-----------|
| Red | 4 | 1.30 |
| Green | 6 | 1.20 |
| Blue | 12 | 2.00 |
| Red | 7 | 1.75 |
| Green | 9 | 1.40 |
| Green | 2 | 1.00 |

Assume that there was a sale, and we get different amounts of discount based on how many marbles are in the bags we buy from this store. The sales table is as follows:

sales

| Count | Discount |
|-------|----------|
| 2 | 5% |
| 4 | 10% |
| 6 | 20% |
| 7 | 30% |
| 9 | 35% |
| 12 | 40% |

This line of code `marbles.[A]([B], [C], [D])` will evaluate the table for the amount of discount applied to each bundle of marbles. The resulting table from your code should look like this:

marbles.[A]([B], [C], [D])

| Amount | Color | Price ($) | Discount |
| 4 | Red | 1.30 | 10% |
| 6 | Green | 1.20 | 20% |
| 12 | Blue | 2.00 | 40% |
| 7 | Red | 1.75 | 30% |
| 9 | Green | 1.40 | 35% |
| 2 | Green | 1.00 | 5% |

Fill in the following placeholders with the exact code. Here it is again: 
marbles.[A]([B], [C], [D]).

Q1. What should go in placeholder [A]?

    Ans: join

Q2. What should go in placeholder [B]?

    Ans: 'Amount'

Q3. What should go in placeholder [C]?

    Ans: sales

Q4. What should go in placeholder [D]?

    Ans: 'Count'

