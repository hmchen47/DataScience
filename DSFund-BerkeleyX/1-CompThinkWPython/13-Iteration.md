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

+ Control Statements  
    These statements _control_ the sequence of computations that are performed in a program
    + The keyword `if` and `for` begin control statements
    + The purpose of `if` is to define functions that choose different behavior based on their arguments
    + The purpose of `for` is to perform a computation for every element in a list or array
+ Demo
    ```python
    def sign(x):
        print(x, 'is', end=' ')
        if x > 0:
            print('positive')
        if x < 0:
            print('negative')
        if x == 0:
            print('zero')
    sign(3)
    sign(-3)
    sign(0)

    def sign(x):
        print(x, 'is', end=' ')
        if x > 1e-15:
            print('positive')
        elif x < -1e-15:
            print('negative')
        elif x == 0:
            print('zero')
        else:
            print('really close to zero')
    sign(3)
    sign(-3)
    sign(0)
    sign(2**0.5 * 2 **0.5 - 2)
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/FuTri6BqicM){:target="_blank"}


## Lec 13.7 For Statements

### Notes

+ Demo
    ```python
    np.append(make_array(1, 2, 3), 4)
    np.append(make_array(1, 2, 3), 'four')
    np.append(make_array(1, 2, 3), make_array(4, 5, 6))
    np.arange(4)

    for i in np.arange(4):
        print('iteration', i)

    coin = make_array('heads', 'tails')
    np.random.choice(coin)

    sum(np.random.choice(coin, 100) == 'heads')
    num_heads = make_array(sum(np.random.choice(coin, 100) == 'heads'))
    num_heads = np.append(num_heads, sum(np.random.choice(coin, 100) == 'heads'))

    for i in np.arange(10000):
        num_heads = np.append(num_heads, sum(np.random.choice(coin, 100) == 'heads'))

    len(num_heads)
    t = Table().with_column('Heads in 100 coin flips', num_heads)
    t.hist(bins=np.arange(30, 70, 1))

    most = t.group(0).where(0, are.between(40,
    sum(most.column('count')) / t.num_rows * 100
    ```

### Videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/hieXCRBU1WE){:target="_blank"}


## Reading and Practice for Section 13

### Reading

This guide assumes that you have watched section 13 (video lecture segments Lec 13.1, Lec 13.2, Lec 13.3, Lec 13.4, Lec 13.5, Lec 13.6, Lec 13.7) in Courseware.

This corresponds to textbook sections:

+ [Chapter 9.1: Conditional Statements](https://www.inferentialthinking.com/chapters/09/1/conditional-statements.html)
+ [Chapter 9.2: Iteration](https://www.inferentialthinking.com/chapters/09/2/iteration.html)

In section 13, we learned more about control statements and saw the first demonstration of using randomness and for loops together. Both of these concepts will be used widely in Data 8.2X Foundations of Data Science: Inferential Thinking by Resampling.

Practice your understanding of for loops with this walkthrough question.

### Practice

Trace each of the iterations in the following for loop.

```python
for count in make_array(1,2,3,7):
    if count % 2 == 0:
        print(“Even!”)
    else:
	print(“Odd!”)
```
Q1. How many times (iterations) does the body of the for loop run?

    Ans: 4

Q2. At Iteration 1...  
    What is count equal to?

    Ans: 1

Q3. What is the output printed? If nothing, leave blank.

    Ans: Odd!

Q4. At Iteration 2...  
    What is count equal to?

    Ans: 2

Q5. What is the output printed? If nothing, leave blank.

    Ans: Even!

Q6. At Iteration 3...  
    What is count equal to?

    Ans: 3

Q7. What is the output printed? If nothing, leave blank.

    Ans: Odd!

Q8. At Iteration 4...  
    What is count equal to?

    Ans: 7

Q9. What is the output printed? If nothing, leave blank.

    Ans: Odd!
