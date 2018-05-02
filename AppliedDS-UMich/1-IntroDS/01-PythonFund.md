# Python Fundamentals

## Week 1 Lectures Jupyter Notebook

To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource.

[Notebook](https://hub.coursera-notebooks.org/user/qceqpnyfwlofzjpttttssh/notebooks/Week%201.ipynb)

## Python Functions

+ Demo
    ```python
    def add_numbers(x, y, z=None, flag=False):
        if (flag):
            print('Flag is true!')
        if (z==None):
            return x + y
        else:
            return x + y + z
        
    print(add_numbers(1, 2, flag=True))

    # Assign function add_numbers to variable a.
    def add_numbers(x,y):
        return x+y

    a = add_numbers
    a(1,2)
    ```

+ Quiz:  
    + This function should add the two values if the value of the "kind" parameter is "add" or is not passed in, otherwise it should subtract the second value from the first.   
        Can you fix the function so that it works?
        ```python
        def do_math(?, ?, ?):
        if (kind=='add'):
            return a+b
        else:
            return a-b

        do_math(1, 2)
        ```
    + Answer
        ```python
        def do_math(a, b, kind='add'):
        if (kind=='add'):
            return a+b
        else:
            return a-b

        do_math(1, 2)
        ```


[Video](https://d3c33hcgiwev3.cloudfront.net/64XjTZU-EeanawoaUJkV-g.processed/full/540p/index.mp4?Expires=1525392000&Signature=TpEYM4VlELIwFgRoFeh0ax07BCXlfWDXBBTKkYtMO~ZHcQZDZsU0~qExC0rJ6r3xpSR7EZUCEsCaqBA4hsMVk2zS~zeAKL6mTOJ-nWINtaf4U0tJBmJTkounkPnCRV1h83FJ775dpVJloEDobw3LP9cIcJiPoWmhMnIlfeIE4Kw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Python Types and Sequences

## Python More on Strings

## Python Demonstration: Reading and Writing CSV files

## Python Dates and Times

## Advanced Python Objects, map()

## Advanced Python Lambda and List Comprehensions

## Advanced Python Demonstration: The Numerical Python Library (NumPy)

## Quiz


