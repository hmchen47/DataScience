Modules
=======

# The CSV Module
In previous missions, we learned how to work with CSV files by:
1. Opening a file
2. Reading the contents of that file into a string
3. Splitting the string on the newline character
4. Splitting each line on the comma character

We can work with CSV files more easily through the csv module. This module has a reader() function that takes a file object as its argument, and returns an object that represents our data. We'll cover objects in more depth in the next mission, but for now, we'll simply convert this object to a list and use that result.

To read data from a file called "my_data.csv", we first import the csv module:

```python
import csv
```
Next, we open the file:
```python
f = open("my_data.csv")
```
Then, we call the module's reader() function:
```python
csvreader = csv.reader(f)
```
Finally, we convert the result to a list:
```python
my_data = list(csvreader)
```

# Introducing NumPy
Using NumPy, we can much more efficiently analyze data than we can using lists. NumPy is a Python module that is used to create and manipulate multidimensional arrays.

An array is a collection of values. Arrays have one or more dimensions. An array dimension is the number of indices it takes to extract a value. In a list, we specify a single index, so it is one-dimensional:


first_row =  [1986, "Western Pacific", "Viet Nam", "Wine", 0]
print(first_row[0])
The code above will print out 1986. A list is similar to a NumPy one-dimensional array, or vector, because we only need a single index to get a value.

world_alcohol.csv, on the other hand, looks like this:

            Column Index
            0       1                   2               3       4   
Row Index    
    0       1986    'Western Pacific'   'Viet Nam'      'Wine'  0
    1       1986    'Americas'          'Uruguay'       'Other' 0.5
    2       1985    'Africa'            'Cte d'Ivoire'  'Wine'  1.62

To extract a single value, we need to specify a row index then a column index. 1, 2 results in Uruguay. 2, 3 results in Wine.

This is a two-dimensional array, also known as a _matrix_. Data sets are usually distributed in the form of _matrices_, and NumPy makes it extremely easy to read in and work with matrices.

## Instructions
+ Use the csv module to read world_alcohol.csv into the variable world_alcohol.
    - You can use the csv.reader function to accomplish this.
    - world_alcohol should be a list of lists.
+ Extract the first column of world_alcohol, and assign it to the variable years.
+ Use list slicing to remove the first item in years (this is a header).
+ Find the sum of all the items in years. Assign the result to total.
    - Remember to convert each item to a float before adding them together.
+ Divide total by the length of years to get the average. Assign the result to avg_year.

```python
import csv

world_alcohol = list(csv.reader(open("world_alcohol.csv", 'r')))

years = [year[0] for year in world_alcohol][1:]
years = [int(year) for year in years]

total = sum(years)
avg_year = total / len(years)
```

# Using NumPy
To get started with NumPy, we first need to import it using import numpy. We can then read in datasets using the __numpy.genfromtxt()__ function.

Since world_alcohol.csv is a csv file, rows are separated by line breaks, and columns are separated by commas, like this:

```python
Year,WHO region,Country,Beverage Types,Display Value
1986,Western Pacific,Viet Nam,Wine,0
1986,Americas,Uruguay,Other,0.5
1985,Africa,Cte d'Ivoire,Wine,1.62
```

In files like this, the comma is called the delimiter, because it indicates where each field ends and a new one begins. Other delimiters, such as tabs, are occasionally used, but commas are the most common.

To use , we need to pass a keyword argument called delimiter that indicates what character is the delimiter:

```python
import numpy
nfl = numpy.genfromtxt("nfl.csv", delimiter=",")
```

The above code would read in the nfl.csv file into a NumPy array. NumPy arrays are represented using the numpy.ndarray class. We'll refer to ndarray objects as NumPy arrays in our material.

## Instructions
Import NumPy.

Use the genfromtxt() function to read world_alcohol.csv into the world_alcohol variable.
Use the type() and print() functions to display the type for world_alcohol.

```python
import numpy as np

world_alcohol = np.genfromtxt("world_alcohol.csv", delimiter=",")

print(type(world_alcohol))
```

# Creating Arrays
We can directly construct arrays from lists using the __numpy.array()__ function.

The __numpy.array()__ function can take a list or list of lists as input. When we input a list, we get a one-dimensional array as a result:

```python
vector = numpy.array([5, 10, 15, 20])
```

When we input a list of lists, we get a matrix as a result:
```python
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
```

## Instructions
+ Create a vector from the list [10, 20, 30].
    - Assign the result to the variable vector.
+ Create a matrix from the list of lists [[5, 10, 15], [20, 25, 30], [35, 40, 45]].
    - Assign the result to the variable matrix.

```python
import numpy as np

vector = np.array([10, 20, 30])
matrix = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
```

# Array Shape
Vectors have a certain number of elements. The vector below has 5 elements:

1986    'Western Pacific' 'Viet Nam' 'Wine' 0

Matrices have a certain number of rows, and a certain number of columns. The matrix below has 5 columns and 3 rows:

Column Index
        0       1                   2               3       4   
Row Index    
0       1986    'Western Pacific'   'Viet Nam'      'Wine'  0
1       1986    'Americas'          'Uruguay'       'Other' 0.5
2       1985    'Africa'            'Cte d'Ivoire'  'Wine'  1.62

It's often useful to know how many elements an array contains. We can use the __ndarray.shape__ property to figure out how many elements are in the array.

For vectors, the shape property contains a tuple with 1 element. A tuple is a kind of list where the elements can't be changed.

```python
vector = numpy.array([1, 2, 3, 4])
print(vector.shape)
```

The code above would result in the tuple (4,). This tuple indicates that the array __vector__ has one dimension, with length 4, which matches our intuition that vector has 4 elements.

For matrices, the shape property contains a tuple with 2 elements.

```python
matrix = numpy.array([[5, 10, 15], [20, 25, 30]])
print(matrix.shape)
```

The above code will result in the tuple (2,3) indicating that __matrix__ has 2 rows and 3 columns.

## Instructions
+ Assign the shape of vector to vector_shape.
+ Assign the shape of matrix to matrix_shape.

```python
vector = numpy.array([10, 20, 30])
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])

vector_shape = vector.shape
matrix_shape = matrix.shape
```

# Data Types
Each value in a NumPy array has to have the same data type. NumPy data types are similar to Python data types, but have slight differences. You can find a full list of NumPy data types here, but here are some of the common ones:
+ bool -- Boolean. Can be True or False.
+ int -- Integer values. An example is 10. Can be int16, int32, or int64. The suffix 16, 32, or 64 indicates how long the number can be, in bytes (we'll dive more into bytes later on). The larger the suffix, the larger the integers can be.
+ float -- Floating point values. An example is 10.6. Can be float16, float32, or float64. The suffix 16, 32, or 64 indicates how many numbers after the decimal point the number can have. The larger the suffix, the more precise the float can be.
+ string -- String values. Can be string or unicode. We'll get more into what unicode is later on, but the difference between string and unicode is how the characters are stored by the computer.

NumPy will automatically figure out an appropriate data type when reading in data or converting lists to arrays. You can check the data type of a NumPy array using the __dtype__ property.

```python
numbers = numpy.array([1, 2, 3, 4])
numbers.dtype
```

Because numbers only contains integers, its data type is int64.

# Inspecting The Data
Here's how the first few rows of world_alcohol look:


array([[             nan,              nan,              nan,              nan,              nan],
       [  1.98600000e+03,              nan,              nan,              nan,   0.00000000e+00],
       [  1.98600000e+03,              nan,              nan,              nan,   5.00000000e-01]])
There are a few concepts we haven't been introduced to yet that we'll get into one by one:
+ Many items in world_alcohol are nan.
+ The entire first row is nan.
+ Some of the numbers are written like 1.98600000e+03.

The data type of world_alcohol is float. Because all of the values in a NumPy array have to have the same data type, NumPy attempted to convert all of the columns to floats when they were read in. The __numpy.genfromtxt()__ function will attempt to guess the correct data type of the array it creates.

In this case, the WHO Region, Country, and Beverage Types columns are actually strings, and couldn't be converted to floats. When NumPy can't convert a value to a numeric data type like float or integer, it uses a special __nan__ value that stands for _Not a Number_. NumPy assigns an __na__ value, which stands for _Not Available_, when the value doesn't exist. __nan__ and __na__ values are types of missing data. We'll dive more into how to deal with missing data in later missions.

The whole first row of world_alcohol.csv is a header row that contains the names of each column. This is not actually part of the data, and consists entirely of strings. Since the strings couldn't be converted to floats properly, NumPy uses __nan__ values to represent them.

If you haven't seen __scientific notation__ before, you might not recognize numbers like 1.98600000e+03. Scientific notation is a way to condense how very large or very precise numbers are displayed. We can represent 100 in scientific notation as 1e+02. The e+02 indicates that we should multiply what comes before it by 10 ^ 2(10 to the power 2, or 10 squared). This results in 1 * 100, or 100. Thus, 1.98600000e+03 is actually 1.986 * 10 ^ 3, or 1986. 1000000000000000 can be written as 1e+15.

In this case, 1.98600000e+03 is actually longer than 1986, but NumPy displays numeric values in scientific notation by default to account for larger or more precise numbers.

# Reading In The Data Properly
Our data wasn't read in properly, which resulted in NumPy trying to convert strings to floats, and nan values. We can fix this by specifying in the __numpy.genfromtxt()__ function that we want to read in all the data as strings. While we're at it, we can also specify that we want to skip the header row of world_alcohol.csv.

We can do this by:
+ Specifying the keyword argument __dtype__ when reading in world_alcohol.csv, and setting it to "U75". This specifies that we want to read in each value as a 75 byte unicode data type. We'll dive more into unicode and bytes later on, but for now, it's enough to know that this will read in our data properly.
+ Specifying the keyword argument __skip_header__, and setting it to 1. This will skip the first row of world_alcohol.csv when reading in the data.

## Instructions
+ Use the numpy.genfromtxt() function to read in world_alcohol.csv:
    - Set the dtype parameter to "U75".
    - Set the skip_header parameter to 1.
    - Set the delimiter parameter to ,.
+ Assign the result to world_alcohol.
+ Use the print() function to display world_alcohol.

```python
world_alcohol = numpy.genfromtxt("world_alcohol.csv", dtype="U75", skip_header=1, delimiter=",")

print(world_alcohol)
```

# Indexing Arrays
We can index NumPy arrays very similarly to how we index regular Python lists. Here's how we would index a NumPy vector:
```python
vector = numpy.array([5, 10, 15, 20])
print(vector[0])
```

The above code would print the first element of vector, or 5.

Indexing matrices is similar to indexing lists of lists. Here's a refresher on indexing lists of lists:
```python
list_of_lists = [
        [5, 10, 15],
        [20, 25, 30]
       ]
```

The first item in list_of_lists is [5, 10, 15]. If we wanted to access the element 15, we could do this:
```python
first_item = list_of_lists[0]
first_item[2]
```
We could also condense the notation like this:
```python
list_of_lists[0][2]
```

We can index matrices in a similar way, but we place both indices inside square brackets. The first index specifies which row the data comes from, and the second index specifies which column the data comes from:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30]
])
print(matrix[1,2])  # no space between numbers and ,
```

In the above code, we pass two indices into the square brackets when we index matrix. This will result in the value in the second row and the third column, or 30.

# Slicing Arrays
We can slice vectors very similarly to how we slice lists:

```python
vector = numpy.array([5, 10, 15, 20])
print(vector[0:3])
```

The above code will print out 5, 10, 15. Like lists, vector slicing is from the first index up to but not including the second index.

Matrix slicing is a bit more complex, and has four forms:
+ When we want to select one entire dimension, and a single element from the other.
+ When we want to select one entire dimension, and a slice of the other.
+ When you want to select a slice of one dimension, and a single element from the other.
+ When we want to slice both dimensions.

We'll dive into the first form in this screen. When we want to select one whole dimension, and an element from the other, we can do this:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix[:,1])
```
This will select all of the rows, but only the column with index 1. So we'll end up with 10, 25, 40, which is the whole second column.

The colon by itself : specifies that the entirety of a single dimension should be selected. Think of the colon as selecting from the first element in a dimension up to and including the last element.

## Instructions
Assign the whole third column from world_alcohol to the variable countries.
Assign the whole fifth column from world_alcohol to the variable alcohol_consumption.
```python
countries = world_alcohol[:,2]
alcohol_consumption = world_alcohol[:,4]
```

# Slicing One Dimension
When we want to select one whole dimension, and a slice of the other, we can do this:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix[:,0:2])
```
This will select all the rows, and columns with index 0, and index 1. We'll end up with:

```python
[
    [5, 10],
    [20, 25],
    [35, 40]
]
```

We can select rows by specifying a colon in the columns area:
```python
print(matrix[1:3,:])
```

The code above will select rows 1 and 2, and all of the columns. We'll end up with this:
```python
[
    [20, 25, 30],
    [35, 40, 45]
]
```

We can also select a single value alongside an entire dimension:
```python
print(matrix[1:3,1])
```

The code above will print rows 1 and 2 and column 1:
```python
[
     [25, 40]
]
```

## Instructions
Assign all the rows and the first 2 columns of world_alcohol to first_two_columns.
Assign the first 10 rows and the first column of world_alcohol to first_ten_years.
Assign the first 10 rows and all of the columns of world_alcohol to first_ten_rows.
```python
first_two_columns = world_alcohol[:,:2]
first_ten_years = world_alcohol[:10,0]
first_ten_rows = world_alcohol[:10,:]
```

# Slicing Arrays
When we want to slice both dimensions, we can do this:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix[1:3,0:2])
```

This will select rows with index 1 and 2, and columns with index 0 and 1:
```python
[
    [20, 25],
    [35, 40]
]
```

## Instructions
Assign the first 20 rows of the columns at index 1 and 2 of world_alcohol to first_twenty_regions.
```python
first_twenty_regions = world_alcohol[:20,1:3]
```

# Array Comparisons
One of the most powerful aspects of the NumPy module is the ability to make comparisons across an entire array. These comparisons result in Boolean values.

Here's an example of how we can do this with a vector:
```python
vector = numpy.array([5, 10, 15, 20])
vector == 10
```

If you'll recall from an earlier mission, the double equals sign (==) compares two values. When used with NumPy, it will compare the second value to each element in the vector. If the value are equal, the Python interpreter returns True; otherwise, it returns False. It stores the Boolean results in a new vector.

For example, the code above will generate the vector [False, True, False, False], since only the second element in vector equals 10.

Here's an example with a matrix:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
matrix == 25
```

The final statement will compare 25 to every element in matrix. The result will be a matrix where elements are True or False:

```python
[
    [False, False, False],
    [False, True,  False],
    [False, False, False]
]
```

In the result, True only appears in a single position - where the corresponding element in matrix is 25.

## Instructions
The variable world_alcohol already contains the data set we're working with.
+ Extract the third column in world_alcohol, and compare it to the string Canada. Assign the result to countries_canada.
+ Extract the first column in world_alcohol, and compare it to the string 1984. Assign the result to years_1984.
```python
countries_canada = (world_alcohol[:,2] == "Canada")
years_1984 = (world_alcohol[:,0] == '1984')
```

# Selecting Elements
We mentioned that comparisons are very powerful, but it may not have been obvious why on the last screen. Comparisons give us the power to select elements in arrays using Boolean vectors. This allows us to conditionally select certain elements in vectors, or certain rows in matrices.

Here's an example of how we would do this with a vector:
```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten = (vector == 10)
​
print(vector[equal_to_ten])
```

The code above:
+ Creates vector.
+ Compares vector to the value 10, which generates a Boolean vector [False, True, False, False]. It assigns the result to equal_to_ten.
+ Uses equal_to_ten to only select elements in vector where equal_to_ten is True. This results in the vector [10].

We can use the same principle to select rows in matrices:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
second_column_25 = (matrix[:,1] == 25)
print(matrix[second_column_25, :])
```

The code above:

Creates matrix.
Uses second_column_25 to select any rows in matrix where second_column_25 is True.
We end up with this matrix:
```python
[
    [20, 25, 30]
]
```
We selected a single row from matrix, which was returned in a new matrix.

## Instructions
Compare the third column of world_alcohol to the string Algeria.
Assign the result to country_is_algeria.
Select only the rows in world_alcohol where country_is_algeria is True.
Assign the result to country_algeria.
```python
country_is_algeria = (world_alcohol[:,2] == "Algeria")
country_algeria = world_alcohol[country_is_algeria == True,:]
```

# Comparisons With Multiple Conditions
On the last screen, we made comparisons based on a single condition. We can also perform comparisons with multiple conditions by specifying each one separately, then joining them with an ampersand (&). When constructing a comparison with multiple conditions, it's critical to put each one in parentheses.

Here's an example of how we would do this with a vector:
```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_and_five = (vector == 10) & (vector == 5)
```

In the above statement, we have two conditions, (vector == 10) and (vector == 5). We use the ampersand (&) to indicate that both conditions must be True for the final result to be True. The statement returns [False, False, False, False], because none of the elements can be 10 and 5 at the same time. Here's a diagram of the comparison logic:

Vector
                        5   10  15  20
                               |
                ---------------------------------
                |                               |
            Condition 1                     Condition 2
            Vector == 10                    Vector == 5

False | True | False | False   and   True | False | False | False
            |                                    |
            |          Combine Conditions        |
            --------------------------------------
                               |
                 False | False | False | False

We can also use the pipe symbol (|) to specify that either one condition or the other should be True:
```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
```

The code above will result in [True, True, False, False].

## Instructions
Perform a comparison with multiple conditions, and join the conditions with &.
 - Compare the first column of world_alcohol to the string 1986.
 - Compare the third column of world_alcohol to the string Algeria.
 - Enclose each condition in parentheses, and join the conditions with &.
 - Assign the result to is_algeria_and_1986.
Use is_algeria_and_1986 to select rows from world_alcohol.
Assign the rows that is_algeria_and_1986 selects to rows_with_algeria_and_1986.
```python
is_algeria_and_1986 = (world_alcohol[:,0] == '1986') & (world_alcohol[:,2] == "Algeria")
rows_with_algeria_and_1986 = world_alcohol[is_algeria_and_1986,:]
```

# Replacing Values
We can also use comparisons to replace values in an array, based on certain conditions. Here's an example of how we would do this for a vector:
```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
vector[equal_to_ten_or_five] = 50
print(vector)
```

This code will complete the following steps:
+ Create an array vector.
+ Compare vector to 10 and 5, and generate a vector that's True where vector is equal to either value.
+ Select only the elements in vector where equal_to_ten_or_five is True.
+ Replace the selected values with the value 50.

The result will be [50, 50, 15, 20].

Here's a diagram showing what takes place at each step in the process:

vector                      5       10      15      20
equal_to_five_or_ten        True    True    False   False
Replacement                 50      50      50      50

                            If equal_to_five_or_ten is True,
Replace                     pick the replacement value.
                            Otherwise, pick the value from
                            vector.

Final vector                50      50      15      20

We can perform the same replacement on a matrix. To do this, we'll need to use indexing to select a column or row first:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
second_column_25 = matrix[:,1] == 25
matrix[second_column_25, 1] = 10
```

The code above will result in:
```python
[
    [5, 10, 15],
    [20, 10, 30],
    [35, 40, 45]
]
```

## Instructions
Replace all instances of the string 1986 in the first column of world_alcohol with the string 2014.
Replace all instances of the string Wine in the fourth column of world_alcohol with the string Grog.
```python
world_alcohol[(world_alcohol[:,0] == '1986'), 0] = '2014'
world_alcohol[(world_alcohol[:,3] == 'Wine'), 3] = 'Grog'
```

# Replacing Empty Strings
We'll soon be working with the Display Value column, which shows how much alcohol the average citizen of a country drinks. However, because world_alcohol currently has a unicode datatype, all of the values in the column are strings. To add these values together or perform any other mathematical operations on them, we'll have to convert the data in the column to floats.

Before we can do this, we need to address the empty string values ('') that appear where there was no original data for the given country and year. If we try to convert the data in the column to floats without removing these values first, we'll get a ValueError. Thankfully, we can remove these items using the replacement technique we learned on the last screen.

## Instructions
Compare all the items in the fifth column of world_alcohol with an empty string ''. Assign the result to is_value_empty.
Select all the values in the fifth column of world_alcohol where is_value_empty is True, and replace them with the string 0.
```python
is_value_empty = (world_alcohol[:,4] == '')
world_alcohol[is_value_empty,4] = 0
```

# Converting Data Types
We can convert the data type of an array with the astype() method. Here's an example of how this works:
```python
vector = numpy.array(["1", "2", "3"])
vector = vector.astype(float)
```

The code above will convert all of the values in vector to floats: [1.0, 2.0, 3.0].

We'll do something similar with the fifth column of world_alcohol, which contains information on how much alcohol the average citizen of a country drank in a given year. To determine which country drinks the most, we'll have to convert the values in this column to float values. That's because we can't add or perform calculations on these values while they're strings.

## Instructions
Extract the fifth column from world_alcohol, and assign it to the variable alcohol_consumption.
Use the astype() method to convert alcohol_consumption to the float data type.
```python
alcohol_consumption = world_alcohol[:,4]
alcohol_consumption = alcohol_consumption.astype(float)
```

# Computing With NumPy
Now that alcohol_consumption consists of numeric values, we can perform computations on it. NumPy has a few built-in methods that operate on arrays. You can view all of them in the documentation. For now, here are a few important ones:
+ sum() -- Computes the sum of all the elements in a vector, or the sum along a dimension in a matrix
+ mean() -- Computes the average of all the elements in a vector, or the average along a dimension in a matrix
+ max() -- Identifies the maximum value among all the elements in a vector, or the maximum along a dimension in a matrix

Here's an example of how we'd use one of these methods on a vector:
```python
vector = numpy.array([5, 10, 15, 20])
vector.sum()
```

This would add together all of the elements in vector, and result in 50.

With a matrix, we have to specify an additional keyword argument, axis. The axis dictates which dimension we perform the operation on. 1 means that we want to perform the operation on each row, and 0 means on each column. The example below performs an operation across each row:
```python
matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
matrix.sum(axis=1)
```

This would compute the sum for each row, resulting in [30, 75, 120].

## Instructions
Use the sum() method to calculate the sum of the values in alcohol_consumption. Assign the result to total_alcohol.
Use the mean() method to calculate the average of the values in alcohol_consumption. Assign the result to average_alcohol.
```python
total_alcohol = alcohol_consumption.sum()
average_alcohol = alcohol_consumption.mean()
```

# Total Annual Alcohol Consumption
Each country is associated with several rows for different types of beverages:

[['1986', 'Americas', 'Canada', 'Other', ''],
 ['1986', 'Americas', 'Canada', 'Spirits', '3.11'],
 ['1986', 'Americas', 'Canada', 'Beer', '4.87'],
 ['1986', 'Americas', 'Canada', 'Wine', '1.33']]

To find the total amount the average person in Canada drank in 1986, for example, we'd have to add up all 4 of the rows shown above, then repeat this process for each country.

## Instructions
Create a matrix called canada_1986 that only contains the rows in world_alcohol where the first column is the string 1986 and the third column is the string Canada.
Extract the fifth column of canada_1986, replace any empty strings ('') with the string 0, and convert the column to the float data type. Assign the result to canada_alcohol.
Compute the sum of canada_alcohol. Assign the result to total_canadian_drinking.
```python
canada_1986 = world_alcohol[(world_alcohol[:,0] == '1986') & (world_alcohol[:,2] == 'Canada'),:]
canada_1986[(canada_1986[:,4] == ''),4] = '0'
canada_alcohol = canada_1986[:,4].astype(float)
total_canadian_drinking = canada_alcohol.sum()
```

# Calculating Consumption For Each Country
Now that we know how to calculate the average consumption of all types of alcohol for a single country and year, we can scale up the process and make the same calculation for all countries in a given year. Here's a rough process:
+ Create an empty dictionary called totals.
+ Select only the rows in world_alcohol that match a given year. Assign the result to year.
+ Loop through a list of countries. For each country:
    - Select only the rows from year that match the given country.
    - Assign the result to country_consumption.
    - Extract the fifth column from country_consumption.
    - Replace any empty string values in the column with the string 0.
    - Convert the column to the float data type.
    - Find the sum of the column.
    - Add the sum to the totals dictionary, with the country name as the key.
+ After the code executes, you'll have a dictionary containing all of the country names as keys, with the associated alcohol consumption totals as the values.

## Instructions
+ We've assigned the list of all countries to the variable countries.
+ Find the total consumption for each country in countries for the year 1989.
    - Refer to the steps outlined above for help.
+ When you're finished, totals should contain all of the country names as keys, with the corresponding alcohol consumption totals for 1989 as values.
```python
totals = {}
year = world_alcohol[(world_alcohol[:,0] == '1989'),:]
for country in countries:
    country_consumption = year[(year[:,2] == country),:]
    country_consumption[country_consumption[:,4] == '',4] = '0'
    country_consumption = country_consumption[:,4].astype(float)
    totals[country] = sum(country_consumption)
```

# Finding The Country That Drinks The Most
Now that we've computed total alcohol consumption for each country in 1989, we can loop through the totals dictionary to find the country with the highest value.

The process we've outlined below will help you find the key with the highest value in a dictionary:
+ Create a variable called highest_value that will keep track of the highest value. Set its value to 0.
+ Create a variable called highest_key that will keep track of the key associated with the highest value. Set its value to None.
+ Loop through each key in the dictionary.
+ If the value associated with the key is greater than highest_value, assign the value to highest_value, and assign the key to highest_key.
+ After the code runs, highest_key will be the key associated with the highest value in the dictionary.

## Instructions
+ Find the country with the highest total alcohol consumption.
+ To do this, you'll need to find the key associated with the highest value in the totals dictionary.
+ Follow the process outlined above to find the highest value in totals.
+ When you're finished, highest_value will contain the highest average alcohol consumption, and highest_key will contain the country that had the highest per capital alcohol consumption in 1989.
```python
highest_value = 0
highest_key = None
for country, consumption in totals.items():
    if consumption > highest_value:
        highest_value = consumption
        highest_key = country
```

# NumPy Strengths And Weaknesses
You should now have a good foundation in NumPy, and in handling issues with your data. NumPy is much easier to work with than lists of lists, because:
+ It's easy to perform computations on data.
+ Data indexing and slicing is faster and easier.
+ We can convert data types quickly.

Overall, NumPy makes working with data in Python much more efficient. It's widely used for this reason, especially for machine learning.

You may have noticed some limitations with NumPy as you worked through the past two missions, though. For example:
+ All of the items in an array must have the same data type. For many datasets, this can make arrays cumbersome to work with.
+ Columns and rows must be referred to by number, which gets confusing when you go back and forth from column name to column number.

In the next few missions, we'll learn about the Pandas library, one of the most popular data analysis libraries. Pandas builds on NumPy, but does a better job addressing the limitations of NumPy.
