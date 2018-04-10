# Section 5a: Strings (Lec 5.1 - Lec 5.4)

## Lec 5.1 Creating Tables

### Notes

+ Ways to create a table

    + Read a table from a spreadsheet: `Table.read_table(filename)`
    + An empty table: `Table()`
    + and ...
+ Array --> Tables

    + Create a table with a single column w/ `data` as an array: `Table().with_column(label, data)`
    + Create a table w/ an array of data for each column: `Table().with_columns(label1, data1, ...)`
+ Table Methods:

    + Create and extending tables: `Table().with_columns` and `Table.read_table
    + Finding the size: `num_rowa` and `num_columns`
    + Referring to columns: labels, relabeling, and indices; column indices start from 0: `labels` and `relabeled`
    + Accessing data in a column: `column` takes a label or index and returns an array
    + Using array methods to work with data in columns: `item`, `sum`, `min`, `max`, and so on
    + Creating new tables containing some of the original columns: `select` and `drop`
+ Examples

    The table `students` has columns `Name`, `ID`, and `Score`. Write one line code that evaluates to:  
    a) A table consisting of only the column labeled `Name`: `students.select('Name')`  
    b) The largest score: `students.column('Score').max()` or `max(students.column('Score'))`

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/Obg7GqjxZ-Q)

## Lec 5.2 Strings

### Notes

+ Text and Strings
    + A Sstring value is a snippet of text of any length
        + 'a'
        + 'We can do it'
        + "there can be 2 sentences. Here's the second!"
    + Strings that contain numbers can be converted to numbers
        + `int('12)`
        + `float('1.2')`
    + Any value can be converted to a string
        + `str(5)`
+ Demo
    ```python
    '2.3' + 4                   # typeError
    '2.3' * 4                   # '2.32.32.32.3' repeats 4 times
    int('23') + float('2.3')
    x = 12; int(str(x) + '0')   # 120
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/NJQr6a-j8b0)

## Lec 5.3 String Exercise

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]

## Lec 5.4 Exercise Answer

### Notes


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)]
