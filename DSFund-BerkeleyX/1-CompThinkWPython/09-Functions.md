# Section 9a: Comparing Histograms (Lec 9.1 - Lec 9.2)

## Lec 9.1 Comparing Histograms

### Notes

+ Overlaid graph: for visually comparing two populatons
+ Demo
    ```python
    height = Table.read_table('galton.csv').select(1, 2, 7).relabeled(2, 'child')
    height.hist('father', unit='inch')          # histogram for father
    height.hist('child', unit='inch')           # histogram for mother
    height.hist(unit='inch', bins=np.arange(55, 80, 2)) # overlaid histograms for 3 cols
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/0ImGO0BG630){:target="_blank"}


## Lec 9.2 Comparing Histograms Discussion

### Notes

+ Discussion Questions  
    The histogram describes a __year__ of daily temperatures  
    Answer these questions , if possible:
    + What portion of days had a high temp in the range $60-70$?
    + What portion had a low of 45 or more?
    + How many days had a difference of more than 20 degrees btw their high & low temperatures.

    ![diagram](./Diagrams/sec09-sec09-temp.png)

+ Answers:
    + $(70 - 60) \times 4.8\% = 48\%$
    + $1 - (0.1\% \times (35 - 10)) - (.7\% \times (40-35)) - (2\% \times (45 - 40)) = 1 - 14\% = 86\%$
    + N/A

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/Ag2929CN3MA){:target="_blank"}


# Section 9b: Functions (Lec 9.3 - Lec 9.6)

## Lec 9.3 Defining Functions

### Notes

+ `def` Statements  
    User-defined functions give names to blocks of code
    ```python
    def spread(values):
        return max(values) - min(values)
    ```
    + Name: `spread`
    + Arguments: `values`
    + Body: `max(values) - min(values)`
    + Return expression: `return max(values) - min(values)`
+ Demo
    ```python
    def double(x):
        """ Double x """
        return 2*x

    double(3)
    double(-4)
    y = 5;  double(y/4)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/DEEsmyz3oRo){:target="_blank"}


## Lec 9.4 Defining Functions Discussion

### Notes

+ Discussion Question
    ```python
    def f(s):
        return np.round(s / sum(s) * 100, 2)
    ```
    + What does this function do?
    + What kind of input does it take?
    + What output will it give?
    + What's a reasonable name?

+ Answer:
    + Get an array of percentage of values, `s`, with 2 decimal precision
    + `s` is an array of the numerical values
    + an array of of float values
    + `f` = percent

+ Demo
    ```python
    counts = make_array(1, 2, 3)
    total = sum(counts)
    np.round((counts/total)*100, 2)

    percents(counts)
    percents(make_array(1, 1, 1, 1))

    def percents(counts, decimal_places=2): # 2 as default value for decimal_places
        """Convert the counts to percents out of the total."""  # string box
        total = sum(counts)
        return np.round((counts/total)*100, decimal_places)

    parts = make_array(2, 1, 4)
    print("Rounded to 1 decimal place:", percents(parts, 1), "or", percents(parts, decimal_places=1))
    print("Rounded to the default number of decimal places:", percents(parts))

    help(percents)  # display what describe within string box
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/4dat6zBtddM){:target="_blank"}


## Lec 9.5 Apply

### Notes

+ Apply with one arguments
    ```python
    `t.apply(func_name, 'col_label')
    ```
    + The `apply` method creates an array by calling a function on every element in input column(s)
    + 1st argument: function to apply
    + 2nd argument: the input column(s)
+ Apply with multiple arguments
    ```python
    `t.apply(two_arg_func, 'col_label_for_1st_arg', 'col_label_for_2nd_arg)
    ```
    + `apply` called with only a function applies it to each row
+ Demo
    ```python
    def cut_off_at_a_billion(x):
    """The smaller of x and 1,000,000,000"""
    return min(x, 1e9)

    cut_off_at_a_billion(12)
    cut_off_at_a_billion(123456)
    cut_off_at_a_billion(1234567890)

    top = Table.read_table('top_movies_2017.csv').where('Studio', 'Fox')
    cut_off = top.apply(cut_off_at_a_billion, 'Gross (Adjusted)')   # example fpr apply
    top.with_column('Adjusted but cut', cut_off)
    ```

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/A9lKV2QBTXs){:target="_blank"}


## Lec 9.6 Example Prediction

### Notes

+ Sir Francis Galton
    + 1822-1911 (knighted in 1909)
    + A pioneer in making predictions
    + Particular (an troublesome) interest in heredity
    + VCharles Darwin's half-cousin
+ Demo
    ```python
    height = Table.read_table('galton.csv').select(1, 2, 7).relabeled(2, 'child')
    height
    height.scatter(2)

    height = height.with_column(
        'parent average', (height.column('mother') + height.column('father')) / 2
    )

    height.scatter('parent average', 'child')
    _ = plots.plot([67.5, 67.5], [50, 85], color='red', lw=2) # vertical line
    _ = plots.plot([68.5, 68.5], [50, 85], color='red', lw=2) # vertical line

    close_to_68 = height.where('parent average', are.between(67.5, 68.5))
    close_to_68.column('child').mean()

    def predict_child(pa):
        close_points = height.where('parent average', are.between(pa - 0.5, pa + 0.5))
        return close_points.column('child').mean()

    predict_child(68)
    predict_child(62)

    # Apply predict_child to all the midparent heights
    height.with_column(
        'prediction', height.apply(predict_child, 'parent average')
    ).select(2, 3, 4).scatter('parent average')
    # result shows a linear association
    ```


### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://youtu.be/eLtLrb_Mfnk){:target="_blank"}


## Reading and Practice for Section 9

### Reading

This guide assumes that you have watched section 9b (video lecture segments Lec 9.3, Lec 9.4, Lec 9.5, Lec 9.6) in Courseware.

This corresponds to textbook sections:

+ [Chapter 8: Applying a Function to a Column](https://www.inferentialthinking.com/chapters/08/1/applying-a-function-to-a-column.html)
+ [Chapter 8.1: Functions and Tables](https://www.inferentialthinking.com/chapters/08/functions-and-tables.html)

In section 9, we continued our discussion of histograms by comparing histograms and their distributions. This is just an extension of the histogram fundaments we learned in section 8. We also learned another important Python programming concept called functions. Functions are powerful pieces of code that give a name to a computational process that may be applied multiple times. Pay attention to the specific structure and format for creating a function. Many students forget little things like using def, the semicolon, or the indentation when creating a new function.

We also saw how a function can be used in a table using the tbl.apply(...) method.

`tbl.apply(function, column)` returns an array where a function is applied to each item in a column.

Try and complete the practice questions below.

### Practice

Here is a function `mystery_function` that takes an input `mystery_input` and returns a mysterious output. Your job is to figure out what `mystery_function` does and answer the following questions below. 

```python
def mystery_function(mystery_input):
      return mystery_input.where("mystery", are.equal_to(42))
```

Q1. What data type for `mystery_input` would not cause an error if passed in as an argument into the function mystery_function? Select the answer that would not cause an error.

    a. An int 42
    b. A string "mystery"
    c. An array called mystery
    d. A table with a column "mystery"

    Ans: d

Q2. What is the output data type for mystery_function?

    a. An int 42
    b. A string "mystery"
    c. An array called mystery
    d. A table with a column "mystery"

    Ans: d

