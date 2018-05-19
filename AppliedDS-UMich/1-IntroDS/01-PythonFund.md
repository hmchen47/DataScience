# Python Fundamentals

## Week 1 Lectures Jupyter Notebook

To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource.

[Web Notebook](https://hub.coursera-notebooks.org/user/qceqpnyfwlofzjpttttssh/notebooks/Week%201.ipynb)

[Local Notebook](./notebooks/Week+1.ipynb)

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


[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/64XjTZU-EeanawoaUJkV-g.processed/full/540p/index.mp4?Expires=1525392000&Signature=TpEYM4VlELIwFgRoFeh0ax07BCXlfWDXBBTKkYtMO~ZHcQZDZsU0~qExC0rJ6r3xpSR7EZUCEsCaqBA4hsMVk2zS~zeAKL6mTOJ-nWINtaf4U0tJBmJTkounkPnCRV1h83FJ775dpVJloEDobw3LP9cIcJiPoWmhMnIlfeIE4Kw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

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

+ Quiz  
    + What would be an appropriate slice to get the name "Christopher" from the string "Dr. Christopher Brooks"?

        ```python
        x = 'Dr. Christopher Brooks'

        print(x[???])
        ```
    + Answer:
        ```python
        x = 'Dr. Christopher Brooks'

        print(x[4:15])
        ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/AuqMXJVAEeaUSArAHh3eJg.processed/full/540p/index.mp4?Expires=1525392000&Signature=Eh8804MzDRIM3qQelGNhx-M2s0iS4ukXXzNqC3C03n0TuUYcqJT452XfZDsv-lq0VI-2bo9vrARcBYRaWEp~YUEjYHvSk9pdjt~dKceIyAR8DwQvJwRkQWLvmxYq9BrlGKLDN9NPuDQEfQHKVAnu9o10NmaVKbmwehpnp7Szp6Q_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Python More on Strings

+ Demo
    ```python
    print('Chris' + 2)          # TypeError
    print('Chris' + str(2))     # Chris2

    # string formatting
    sales_record = {        # Dictionary
    'price': 3.24,
    'num_items': 4,
    'person': 'Chris'}

    sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'
    print(sales_statement.format(sales_record['person'],
                                sales_record['num_items'],
                                sales_record['price'],
                                sales_record['num_items']*sales_record['price']))
    ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/EMUmMJVAEeanawoaUJkV-g.processed/full/540p/index.mp4?Expires=1525478400&Signature=AMzoI9XdNEehc39avztsbfY1mbO3WX6oQNAz4JV5owPFhs~BVMplI4vUQ2B9AwD5G4N9qbysuw-WX7iSOsUNaQ5v1WXhlA7D9WA7eWhSI~vM0SBtf5bI3dFuHPKLA5lORK6fKrsyhVYbjd9sVXtJFceWm0Hkr2QSHb8RLQNKmiM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Python Demonstration: Reading and Writing CSV files

+ Demo  
    datafile `mpg.csv`, which contains fuel economy data for 234 cars.

    | parameter | description |
    |-----------|--------------|
    | mpg | miles per gallon   |
    | class | car classification   |
    | cty | city mpg   |
    | cyl | # of cylinders   |
    | displ | engine displacement in liters   |
    | drv | f = front-wheel drive, r = rear wheel drive, 4 = 4wd   |
    | fl | fuel (e = ethanol E85, d = diesel, r = regular, p = premium, c = CNG)   |
    | hwy | highway mpg   |
    | manufacturer | automobile manufacturer   |
    | model | model of car   |
    | trans | type of transmission   |
    | year | model year |

    ```python
    import csv

    %precision 2    # floating precision for printing

    with open('filename.csv') as csvfile:
        mpg = list(csv.DictReader(csvfile))
        # read data and convert to nested dictionary

    mpg[:3]     # The first three dictionaries in list.

    len(mpg)    # number of records

    # keys gives the column names of the csv
    mpg[0].keys()

    # find the average cty fuel economy across all cars
    sum(float(d['cty']) for d in mpg) / len(mpg)
    sum(float(d['hwy']) for d in mpg) / len(mpg)

    # Use set to return the unique values for the number of cylinders the cars in our dataset have
    cylinders = set(d['cyl'] for d in mpg)

    # grouping the cars by number of cylinder, and finding the average cty mpg for each group
    CtyMpgByCyl = []

    for c in cylinders: # iterate over all the cylinder levels
        summpg = 0
        cyltypecount = 0
        for d in mpg:                       # iterate over all dictionaries
            if d['cyl'] == c:               # if the cylinder level type matches,
                summpg += float(d['cty'])   # add the cty mpg
                cyltypecount += 1           # increment the count
        CtyMpgByCyl.append((c, summpg / cyltypecount)) # append the tuple ('cylinder', 'avg mpg')

    CtyMpgByCyl.sort(key=lambda x: x[0])
    CtyMpgByCyl

    # Use set to return the unique values for the class types in the dataset.
    vehicleclass = set(d['class'] for d in mpg) # what are the class types

    # find the average hwy mpg for each class of vehicle in the dataset
    HwyMpgByClass = []

    for t in vehicleclass: # iterate over all the vehicle classes
        summpg = 0
        vclasscount = 0
        for d in mpg:                       # iterate over all dictionaries
            if d['class'] == t:             # if the cylinder amount type matches,
                summpg += float(d['hwy'])   # add the hwy mpg
                vclasscount += 1            # increment the count
        HwyMpgByClass.append((t, summpg / vclasscount)) # append the tuple ('class', 'avg mpg')

    HwyMpgByClass.sort(key=lambda x: x[1])
    HwyMpgByClass
    ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/Js4rn5VAEeaUSArAHh3eJg.processed/full/540p/index.mp4?Expires=1525478400&Signature=JlgG6ke5Md8DFKGIow8rpDn0c02YwO0KchbWMtAPJXGdCYgsIQfIXkAPWvJe68hvpFkfFhvCdQKS44FmVyYgtXu04MBF4iHxzgAKDdG~xAILDZn3I4G2o9DGJ2XWv7PTi9vADQNcryRQ7Z~jRevNRpt8XNPBzu0FOsvQD~2SyVY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Python Dates and Times

+ Demo
    ```python
    import datetime as dt
    import time as tm

    # time returns the current time in seconds since the Epoch. (January 1st, 1970)
    tm.time()

    # Convert the timestamp to datetime
    dtnow = dt.datetime.fromtimestamp(tm.time())

    # datetime attributes
    dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime

    # timedelta is a duration expressing the difference between two dates
    delta = dt.timedelta(days = 100) # create a timedelta of 100 days

    # date.today returns the current local date
    today = dt.date.today()
    today - delta # the date 100 days ago
    today > today-delta # compare dates
    ```

[![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/G2bb3pVAEeaUSArAHh3eJg.processed/full/540p/index.mp4?Expires=1525478400&Signature=ibsctGYJRKl72W23GG-KVZ3~d-GGGaA903s8NJt1glTihsK~ykkqxecdDLMAU7G7ipaRVNxG9m0WwxUQRzOQa1lYCJW-T8TFWwst4itPuuk9L636zaaUL7KuropXt5r1kVZAb2mfpf~vkBGYaZJ-NVxGWISlld06v8F~DMwk3KA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Advanced Python Objects, map()

+ Class Object
    + Mostly use camel name
    + No private or protect members
    + No explicit constructor for creating objects, can be achieved by `__init__(self, ...)`

+ `map()` function
    + one of the basic functional programming
    + functional programming: 
        + a programming paradigm in which programmers explicitly declare all parameters which can change through execution of a given function
        + referred to being side effect free, because there is a software contract that describes what can actually change by calling a function
        + chaining operations together
    + return a `map` object which does not actually execute the function until access it

+ Demo
    ```python
    # class example
    class Person:
        department = 'School of Information' #a class variable

        def set_name(self, new_name): #a method
            self.name = new_name
        def set_location(self, new_location):
            self.location = new_location

    person = Person()
    person.set_name('Christopher Brooks')
    person.set_location('Ann Arbor, MI, USA')
    print('{} live in {} and works in the department {}'.format(person.name, 
        person.location, person.department))

    # mapping the min function between two lists
    store1 = [10.00, 11.00, 12.34, 2.34]
    store2 = [9.00, 11.10, 12.34, 2.01]
    cheapest = map(min, store1, store2)

    # iterate through the map object to see the values
    for item in cheapest:
        print(item)
    ```

+ Quiz  
    + Here is a list of faculty teaching this MOOC. Can you write a function and apply it using `map()` to get a list of all faculty titles and last names (e.g. `['Dr. Brooks', 'Dr. Collins-Thompson', …]`) ?
        ```python
        people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 
                'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

        def split_title_and_name(person):
            return #Your answer here

        list(map(#Your answer here))
        ```
    + Answer:
        ```python
        people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 
                'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

        def split_title_and_name(person):
            title = person.split()[0]
            lastname = person.split()[-1]
            return '{} {}'.format(title, lastname)

        list(map(split_title_and_name, people))
        ```


## Advanced Python Lambda and List Comprehensions

+ `lambda` function
    + Anonymous function, a function w/o name
    + simple and short, usually in one line
    + useful for data cleaning task

+ Demo
    ```python
    my_function = lambda a, b, c | a + b
    my_function(1, 2, 3)

    # Iteration
    my_list = []
    for number in range(0, 1000):
        if number % 2 == 0:
            my_list.append(number)
    my_list

    # list comprehension
    my_list = [number for number in range(0,1000) if number % 2 == 0]
    ```


+ Quiz
    + Convert this function into a lambda:
        ```python
        people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 
                'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

        def split_title_and_name(person):
            return person.split()[0] + ' ' + person.split()[-1]

        #option 1
        for person in people:
            print(split_title_and_name(person) == (lambda person:???))

        # option 2
        # list(map(split_title_and_name, people)) == list(map(???))
        ```
    + Answer:
        ```python
        people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 
                'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

        def split_title_and_name(person):
            return person.split()[0] + ' ' + person.split()[-1]

        #option 1
        for person in people:
            print(split_title_and_name(person) == (lambda x: x.split()[0] + ' ' 
                + x.split()[-1])(person))

        #option 2
        list(map(split_title_and_name, people)) == list(map(lambda person: person.split()[0] 
            + ' ' + person.split()[-1], people))
        ```
    + List comprehensiom
        ```python
        def times_tables():
            lst = []
            for i in range(10):
                for j in range (10):
                    lst.append(i*j)
            return lst

        times_tables() == [???]
        ```
    + Answer:
        ```python
        times_tables() == [i * j for i in range(10) for j in rang(10)]
        ```
    + Here’s a harder question which brings a few things together.

        Many organizations have user ids which are constrained in some way. Imagine you work at an internet service provider and the user ids are all two letters followed by two numbers (e.g. aa49). Your task at such an organization might be to hold a record on the billing activity for each possible user.

        Write an initialization line as a single list comprehension which creates a list of all possible user ids. Assume the letters are all lower case.
        ```python
        lowercase = 'abcdefghijklmnopqrstuvwxyz'
        digits = '0123456789'

        answer = [...]
        correct_answer == answer
        ```
    + Answer:
        ```python
        answer = [a+b+c+d for a in lowercase for b in lowercase for c in digits for d in digits]
        ```

    [![Video Icon](https://www.nslegalaid.ca/wp-content/uploads/2017/05/Video. =120x)](https://d3c33hcgiwev3.cloudfront.net/ZRyxx4kQEeaKKwpaECzIKQ.processed/full/540p/index.mp4?Expires=1525478400&Signature=MUAfuzVgQJAxTPsY0Ey5qf3ZS4yRStcsrujIdbXe9OxxZgW5wL~sy~RYukvA8MJ-W-uD1FH13kfI-EtWGqGhwkiLpqp8rnAgOQwbk7nJ3OC03efeDZqotcmNaCL8aszQbwwtMgbzX8DWRbSBw7V~FvcLYxXimggauMMJmQlXu7Q_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)


## Advanced Python Demonstration: The Numerical Python Library (NumPy)

+ `np.array` func:
    + Syntax: `array(object, subok=False, ndmin=0)`
    + Create an array
    + `object`: array_like
    + `subok`: bool, optional  
        + True: sub-classes will be passed-through
        + False: returned array forced to be a base-class array (default).
    + `ndmin`: int, optional
        Specifies the minimum number of dimensions that the resulting array should have.
+ `np.arange` func:
    + Syntax: `arange([start,] stop[, step,])`
    + Return evenly spaced values within a given interval; values are generated within the half-open interval $[start, stop)$
+ `np.reshape` func:
    + Syntax: `reshape(a, newshape, order='C')`
    + Gives a new shape to an array without changing its data
    + `a`: array_like
    + `newshape`: int or tuple of ints  
        The new shape should be compatible with the original shape.
    + `order`: {'C', 'F', 'A'}, optional
        + 'C': read / write the elements using C-like index order
        + 'F': read / write the elements using Fortran-like index order
        + 'A': read / write the elements in Fortran-like index order if `a` is Fortran *contiguous* in memory, C-like order otherwise.
+ `np.linspace` func:
    + Syntax: `linspace(start, stop, num=50, endpoint=True)`
    + Return evenly spaced numbers over a specified interval
    + `start`, `stop`: scalar
    + `num`: int; eturn evenly spaced numbers over a specified interval
    + `endpoint`: bool
        + True: `stop` is the last sample.
        + False: `stop` not included
+ `np.resize` func:
    + Syntax: `resize(a, new_shape)`
    + Return a new array with the specified shape; reshaped_array : ndarray
    + `a`: array_like
    + new_shape : int or tuple of int; Shape of resized array
+ `np.ones` & `np.zeros` func:
    + Syntax: `ones(shape, order='C')` &  `zeros(shape, order='C')`
    + Return a new array of given shape and type, filled with ones
    + `shape`: int or sequence of ints  
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    + `order`: {'C', 'F'}, optional
+ `np.eye` func:
    + Syntax: `eye(N, M=None, k=0, order='C')`
    + Return a 2-D array with ones on the diagonal and zeros elsewhere
    + `N`, `M`: ints, Number of rows and columns in the output
    + `k`: int; Index of the diagonal: 
        + `0` (the default): the main diagonal
        + a positive value refers to an upper diagonal
        + a negative value to a lower diagonal.
+ `np.diag` func:
    + Syntax: `diag(v, k=0)`
    + Extract a diagonal or construct a diagonal array
    + `v`: array_like  
        + `v` 2-D array: return a copy of its `k`-th diagonal.
        + `v` 1-D array: return a 2-D array with `v` on the `k`-th diagonal.
+ `np.vstack` func:
    + Syntax: `vstack(tup)`
    + Stack arrays in sequence vertically (row wise)
    + `tup`: sequence of ndarrays
+ `np.dot` func:
    + Syntax: `dot(a, b, out=None)`
    + Dot product of two arrays. Specifically,
        + If both `a` and `b` are 1-D arrays, it is inner product of vectors (without complex conjugation).
        + If both `a` and `b` are 2-D arrays, it is matrix multiplication, but using :func:`matmul` or ``a @ b`` is preferred.
        + If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply` and using `numpy.multiply(a, b)` or `a * b` is preferred.
        + If `a` is an N-D array and `b` is a 1-D array, it is a sum product over the last axis of `a` and `b`.
        + If `a` is an N-D array and `b` is an M-D array (where $M>=2$), it is a sum product over the last axis of `a` and the second-to-last axis of `b`:: `dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`
    + `a`, `b`: array_like
    + `out`: Output argument
    
+ Demo
    ```python
    import numpy as np

    # create numpy array
    mylist = [1, 2, 3]
    x = np.array(mylist)

    y = np.array([4, 5, 6])
    m = np.array([[7, 8, 9], [10, 11, 12]])

    # shape method to find the dimensions of the array. (rows, columns)
    m.shape

    # arange returns evenly spaced values within a given interval.
    n = np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30

    # reshape returns an array with the same data with a new shape.
    n = n.reshape(3, 5) # reshape array to be 3x5

    # linspace returns evenly spaced numbers over a specified interval.
    o = np.linspace(0, 4, 9) # return 9 evenly spaced values from 0 to 4

    # resize changes the shape and size of array in-place.
    o.resize(3, 3)

    # ones returns a new array of given shape and type, filled with ones.
    np.ones((3, 2))

    # zeros returns a new array of given shape and type, filled with zeros.
    np.zeros((2, 3))

    # eye returns a 2-D array with ones on the diagonal and zeros elsewhere.
    np.eye(3)

    # diag extracts a diagonal or constructs a diagonal array.
    np.diag(y)

    # Create an array using repeating list (or see np.tile)
    np.array([1, 2, 3] * 3)

    # Repeat elements of an array using repeat.
    np.repeat([1, 2, 3], 3)

    # Combining Arrays
    p = np.ones([2, 3], int)

    # Use vstack to stack arrays in sequence vertically (row wise)
    np.vstack([p, 2*p])

    # Use hstack to stack arrays in sequence horizontally (column wise).
    np.hstack([p, 2*p])

    # Operations
    print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
    print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]
    print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
    print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
    print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]

    # Dot Product:
    x.dot(y) # dot product  1*4 + 2*5 + 3*6

    z = np.array([y, y**2])
    print(len(z)) # number of rows of array

    # Transposing permutes the dimensions of the array.
    z = np.array([y, y**2])

    # shape of array z is (2,3) before transposing.
    z.shape

    # Use .T to get the transpose.
    z.T

    # The number of rows has swapped with the number of columns.
    z.T.shape

    # Use .dtype to see the data type of the elements in the array.
    z.dtype

    # Use .astype to cast to a specific type.
    z = z.astype('f')
    z.dtype

    # Math Functions
    a = np.array([-4, -2, 1, 3, 5])
    a.sum()
    a.max()
    a.min()
    a.mean()
    a.std()

    # argmax and argmin return the index of the maximum and minimum values in the array.
    a.argmax()
    a.argmin()

    # Indexing / Slicing
    s = np.arange(13)**2

    # Use bracket notation to get the value at a specific index. Remember that indexing starts at 0.
    s[0], s[4], s[-1]

    # Leaving start or stop empty will default to the beginning/end of the array
    s[1:5]

    # Use negatives to count from the back.
    s[-4:]

    # A second | can be used to indicate step-size. array[start:stop:stepsize]
    # Here we are starting 5th element from the end, and counting backwards by 2 until the beginning of the array is reached.
    s[-5::-2]

    # multidimensional array
    r = np.arange(36)
    r.resize((6, 6))

    # Use bracket notation to slice: array[row, column]
    r[2, 2]

    # And use | to select a range of rows or columns
    r[3, 3:6]

    # Here we are selecting all the rows up to (and not including) row 2, and all the columns up to (and not including) the last column.
    r[:2, :-1]

    # This is a slice of the last row, and only every other element.
    r[-1, ::2]

    # We can also perform conditional indexing. Here we are selecting values from the array that are greater than 30. (Also see np.where)
    r[r > 30]

    # Here we are assigning all values in the array that are greater than 30 to the value of 30.
    r[r > 30] = 30

    # copying data, use r.copy to create a copy that will not affect the original array
    r_copy = r.copy()

    # Iterating Over Arrays
    test = np.random.randint(0, 10, (4,3))

    # Iterate by row:
    for row in test:
        print(row)

    # Iterate by index:
    for i in range(len(test)):
        print(test[i])

    # Iterate by row and index:
    for i, row in enumerate(test):
        print('row', i, 'is', row)

    # Use zip to iterate over multiple iterables.
    test2 = test**2

    for i, j in zip(test, test2):
        print(i,'+',j,'=',i+j)

    ```

+ Quiz
    + Q1
        ```python
        old = np.array([[1, 1, 1],
                        [1, 1, 1]])

        new = old
        new[0, :2] = 0

        print(old)
        ```
    + Answer
        ```python
        [[0 0 1]
         [1 1 1]]
        ```
    + Q2
        ```python
        old = np.array([[1, 1, 1],
                        [1, 1, 1]])

        new = old.copy()
        new[:, 0] = 0

        print(old)
        ```
    + Answer: 
        ```python
        [[1 1 1]
         [1 1 1]]
        ```

## Quiz

1. Python is an example of an

    a. Interpreted language  
    b. Declarative language  
    c. Operating system language  
    d. Data science language  
    e. Low level language

    Ans: a


2. Data Science is a

    a. Branch of statistics  
    b. Branch of computer science  
    c. Branch of artificial intelligence  
    d. Interdisciplinary, made up of all of the above

    Ans: d


3. Data visualization is not a part of data science.

    Ans: False


4. Which bracketing style does Python use for tuples?

    a. { }  
    b. ( )  
    c. [ ]

    Ans: b


5. In Python, strings are considered Mutable, and can be changed.

    Ans: False


6. What is the result of the following code: `['a', 'b', 'c'] + [1, 2, 3]`

    a. `['a', 'b', 'c', 1, 2, 3]`  
    b. TypeError: Cannot convert list(int) to list(str)  
    c. `['a1', 'b2', 'c3']`  
    d. `[['a', 'b', 'c'], [1, 2, 3]]`

    Ans: a


7. String slicing is

    a. A way to make string mutable in python  
    b. A way to reduce the size on disk of strings in python  
    c. A way to make a substring of a string in python

    Ans: c


8. When you create a lambda, what type is returned? E.g. type(lambda x: x+1) returns

    a. <class 'function'>
    b. <class 'type'>
    c. <class 'int'>
    d. <class 'lambda'>

    Ans: a


9. The epoch refers to  
    a. January 1, year 0  
    b. January 1, year 1970  
    c. January 1, year 1980  
    d. January 1, year 2000

    Ans: b


10. This code, `[x**2 for x in range(10)]` , is an example of a

    a. List comprehension  
    b. Sequence comprehension  
    c. Tuple comprehension  
    d. List multiplication

    Ans: a

11. Given a 6x6 NumPy array `r`, which of the following options would slice the shaded elements?

    ![diagram](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/ld68zYqoEeaCXQ5dBCgoUw_5c8cf678f7cb43d00e2b9e30c434b0c6_Slice1.png?expiry=1525478400000&hmac=Ny6OoEmefuNRaWfc071uK1MKJmPJCWRiZDSSAYdk3AY)

    a. `r[::7]`  
    b. `r[0:6,::-7]`  
    c. `r[:,::7]`  
    d. `r.reshape(36)[::7]`

    Ans: d, b(x)  
    You could also use `np.diag(r)`. 


12. Given a 6x6 NumPy array r, which of the following options would slice the shaded elements?

    ![diagram](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/INF_bIqrEealuRI3K47d-Q_0931f86653a0cee389ee5b03480acb7d_Slice3.png?expiry=1525478400000&hmac=0XUXfAIg2066W8gKiXzEhiBNwNKAmltP85aD_TbixXU)

    a. `r[2::2,2::2]`  
    b. `r[[2,3],[2,3]]`  
    c. `r[2:4,2:4]`  
    d. `r[::2,::2]`  

    Ans: c


