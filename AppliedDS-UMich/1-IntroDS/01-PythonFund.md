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

[video](https://d3c33hcgiwev3.cloudfront.net/AuqMXJVAEeaUSArAHh3eJg.processed/full/540p/index.mp4?Expires=1525392000&Signature=Eh8804MzDRIM3qQelGNhx-M2s0iS4ukXXzNqC3C03n0TuUYcqJT452XfZDsv-lq0VI-2bo9vrARcBYRaWEp~YUEjYHvSk9pdjt~dKceIyAR8DwQvJwRkQWLvmxYq9BrlGKLDN9NPuDQEfQHKVAnu9o10NmaVKbmwehpnp7Szp6Q_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

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

[video](https://d3c33hcgiwev3.cloudfront.net/EMUmMJVAEeanawoaUJkV-g.processed/full/540p/index.mp4?Expires=1525478400&Signature=AMzoI9XdNEehc39avztsbfY1mbO3WX6oQNAz4JV5owPFhs~BVMplI4vUQ2B9AwD5G4N9qbysuw-WX7iSOsUNaQ5v1WXhlA7D9WA7eWhSI~vM0SBtf5bI3dFuHPKLA5lORK6fKrsyhVYbjd9sVXtJFceWm0Hkr2QSHb8RLQNKmiM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Python Demonstration: Reading and Writing CSV files

+ Demo  
    datafile `mpg.csv`, which contains fuel economy data for 234 cars.

    mpg : miles per gallon  
    class : car classification  
    cty : city mpg  
    cyl : # of cylinders  
    displ : engine displacement in liters  
    drv : f = front-wheel drive, r = rear wheel drive, 4 = 4wd  
    fl : fuel (e = ethanol E85, d = diesel, r = regular, p = premium, c = CNG)  
    hwy : highway mpg  
    manufacturer : automobile manufacturer  
    model : model of car  
    trans : type of transmission  
    year : model year

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

[video](https://d3c33hcgiwev3.cloudfront.net/Js4rn5VAEeaUSArAHh3eJg.processed/full/540p/index.mp4?Expires=1525478400&Signature=JlgG6ke5Md8DFKGIow8rpDn0c02YwO0KchbWMtAPJXGdCYgsIQfIXkAPWvJe68hvpFkfFhvCdQKS44FmVyYgtXu04MBF4iHxzgAKDdG~xAILDZn3I4G2o9DGJ2XWv7PTi9vADQNcryRQ7Z~jRevNRpt8XNPBzu0FOsvQD~2SyVY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

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

[video](https://d3c33hcgiwev3.cloudfront.net/G2bb3pVAEeaUSArAHh3eJg.processed/full/540p/index.mp4?Expires=1525478400&Signature=ibsctGYJRKl72W23GG-KVZ3~d-GGGaA903s8NJt1glTihsK~ykkqxecdDLMAU7G7ipaRVNxG9m0WwxUQRzOQa1lYCJW-T8TFWwst4itPuuk9L636zaaUL7KuropXt5r1kVZAb2mfpf~vkBGYaZJ-NVxGWISlld06v8F~DMwk3KA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

## Advanced Python Objects, map()

+ Class Object
    + Mostly use camel name
    + No private or protect members
    + No explicit constructor for creating objects, can be achieved by `__init__(self, ...)`

+ `map()` fi=unction
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
    print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))

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
        people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

        def split_title_and_name(person):
            return #Your answer here

        list(map(#Your answer here))
        ```
    + Answer:
        ```python
        people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

        def split_title_and_name(person):
            title = person.split()[0]
            lastname = person.split()[-1]
            return '{} {}'.format(title, lastname)

        list(map(split_title_and_name, people))
        ```


## Advanced Python Lambda and List Comprehensions

## Advanced Python Demonstration: The Numerical Python Library (NumPy)

## Quiz


