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

+ Demo
    ```python
    # Use type to return the object's type.
    type('This is a string')
    type(None)
    type(1)
    type(1.0)
    type(add_number)

    # Tuples are an immutable data structure (cannot be altered).
    x = (1, 'a', 2, 'b')
    type(x)

    # Lists are a mutable data structure.
    x = [1, 'a', 2, 'b']
    type(x)

    # Use append to append an object to a list.
    x.append(3.3)

    # loop through each item in the list
    for item in x:
        print(item)

    # using the indexing operator
    i=0
    while( i != len(x) ):
        print(x[i])
        i = i + 1

    # Use + to concatenate lists
    [1,2] + [3,4]

    # Use * to repeat lists.
    [1]*3

    # Use the in operator to check if something is inside a list.
    1 in [1, 2, 3]

    # Use bracket notation to slice a string.
    x = 'This is a string'
    print(x[0]) #first character
    print(x[0:1]) #first character, but we have explicitly set the end character
    print(x[0:2]) #first two characters

    # return the last element of the string
    x[-1]

    # return the slice starting from the 4th element from the end and stopping before the 2nd element from the end
    x[-4:-2]

    # a slice from the beginning of the string and stopping before the 3rd element
    x[:3]

    # a slice starting from the 4th element of the string and going all the way to the end
    x[3:]

    # String manipulations

    firstname = 'Christopher'
    lastname = 'Brooks'

    print(firstname + ' ' + lastname)
    print(firstname*3)
    print('Chris' in firstname)

    # split returns a list of all the words in a string, or a list split on a specific character.
    firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
    lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
    print(firstname)
    print(lastname)

    # convert objects to strings
    'Chris' + str(2)

    # Dictionaries associate keys with values.
    x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
    x['Christopher Brooks'] # Retrieve a value by using the indexing operator

    # Iterate over all of the keys
    for name in x:
        print(x[name])

    # Iterate over all of the values
    for email in x.values():
        print(email)

    # Iterate over all of the items in the list
    for name, email in x.items():
        print(name)
        print(email)

    # unpack a sequence into different variables. Make sure the number of values you are unpacking matches the number of variables being assigned.
    x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
    fname, lname, email = x



    ```

[video](https://d3c33hcgiwev3.cloudfront.net/AuqMXJVAEeaUSArAHh3eJg.processed/full/540p/index.mp4?Expires=1525392000&Signature=Eh8804MzDRIM3qQelGNhx-M2s0iS4ukXXzNqC3C03n0TuUYcqJT452XfZDsv-lq0VI-2bo9vrARcBYRaWEp~YUEjYHvSk9pdjt~dKceIyAR8DwQvJwRkQWLvmxYq9BrlGKLDN9NPuDQEfQHKVAnu9o10NmaVKbmwehpnp7Szp6Q_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Python More on Strings

## Python Demonstration: Reading and Writing CSV files

## Python Dates and Times

## Advanced Python Objects, map()

## Advanced Python Lambda and List Comprehensions

## Advanced Python Demonstration: The Numerical Python Library (NumPy)

## Quiz


