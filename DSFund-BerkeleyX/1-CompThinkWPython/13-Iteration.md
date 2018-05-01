# Section 13: Iteration (Lec 13.1 - Lec 13.7)

## Lec 13.1 Comparison

### Notes

+ Combining Comparisons  
    Boolean operators can be applied to `bool` values
    ```
    a = True        b = False
    not b           a or b              a and not b     # evaluate to True
    a and b         not (a or b)        b and b         # evaluate to False
    ```
+ Aggregating Comparisons  
    Summing an array or list of book values will count the `True` values only.
    ```
    1           + 0         + 1         == 2
    True        + False     + True      == 2
    Sum([1,     0,          1])         == 2
    sum([True,  False,      True])      == 2
    ```
+ Demo
    ```python
    'Dog' > 'Catastrophe' > 'Cat'
    a = np.arange(11, 50)

    def teen(x):
        return 13 <= n <= 19

    ages = Table().with_column('Age', a)
    ages = ages.with_column('Teenager', ages.apply(teen, 'Age'))

    sum([False, True, True, False, True])
    ages.column(1)
    sum(ages.column(1))
    ```


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/5zIr9d0KbLI){:target="_blank"}


## Lec 13.2 Predicates

### Notes

+ a function returns `True` or `False`
+ Demo
    ```python
    still_young = are.between(35, 40)
    ages.where('Age', still_young)
    still_young(38)
    ages.apply(still_young, 'Age')
    sum(ages.apply(still_young, 'Age'))
    ages.where('Age', teen)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/f8L0HBo_QYY){:target="_blank"}


## Lec 13.3 Random Selection

### Notes

+ Random Selection
    + `np.random.choice`
        + Selectes uniformly at random
        + with replacement
        + from an array
        + a specified number of times
    + `np.random.choice(some_ary, sample_size)`
+ Demo
    ```python
    two_groups = make_array('treatment', 'control')
    np.random.choice(two_groups)
    np.random.choice(two_groups, 10)
    np.random.choice(two_groups, 10)

    outcomes = np.random.choice(two_groups, 10)
    outcomes == 'control'       # returns bool array
    sum(outcomes == 'control')
    sum(outcomes == 'treatment')
    np.count_nonzero(outcomes == 'control')     # number of True in array
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/tOczQUu4PBg){:target="_blank"}


## Lec 13.4 Random Selection Discussion

### Notes

+ Discussion Question  
    `d = np.arange(6) + 1`  
    What results from evaluating the following 2 expressions?  
    Are they the same? Do they describe the same process?  
    ```python
    np.random.choice(d, 1000) + np.random.choice(d, 1000)

    2 * np.random.choice(d, 1000)
    ```
+ Answer:  
    Two different processes.

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/gbF9s7xeUKw){:target="_blank"}


## Lec 13.5 Print

### Notes

+ Demo
    ```python
    def double(x):
        print('doubling', x)
        return 2 * x

    def square(x):
        print('squaring', x)
        return x * x

    double(square(double(5)))

    print('one', end=' long ')
    print('line')
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/Hejj5yWdoQo){:target="_blank"}


## Lec 13.6 Control Statements

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/FuTri6BqicM){:target="_blank"}


## Lec 13.7 For Statements

### Notes


### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/hieXCRBU1WE){:target="_blank"}


## Reading and Practice for Section 13

### Reading


### Practice



