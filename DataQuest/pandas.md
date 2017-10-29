Pandas
======

1. Introduction To Pandas
Pandas is a library that unifies the most common workflows that data analysts and data scientists previously relied on many different libraries for. Pandas has quickly became an important tool in a data professional's toolbelt and is the most popular library for working with tabular data in Python. Tabular data is any data that can be represented as rows and columns. The CSV files we've worked with in previous missions are all examples of tabular data.

To represent tabular data, pandas uses a custom data structure called a __dataframe__. A dataframe is a highly efficient, 2-dimensional data structure that provides a suite of methods and attributes to quickly explore, analyze, and visualize data. The dataframe is similar to the NumPy 2D array but adds support for many features that help you work with tabular data.

One of the biggest advantages that pandas has over NumPy is the ability to store mixed data types in rows and columns. Many tabular datasets contain a range of data types and pandas dataframes handle mixed data types effortlessly while NumPy doesn't. Pandas dataframes can also handle missing values gracefully using a custom object, __NaN__, to represent those values. A common complaint with NumPy is its lack of an object to represent missing values and people end up having to find and replace these values manually. In addition, pandas dataframes contain axis labels for both rows and columns and enable you to refer to elements in the dataframe more intuitively. Since many tabular datasets contain column titles, this means that dataframes preserve the _metadata_ from the file around the data.

2. Introduction To The Data
In this mission, you'll learn the basics of pandas while exploring a dataset from the United States Department of Agriculture (USDA). This dataset contains nutritional information on the most common foods Americans consume. Each column in the dataset shows a different attribute of the foods and each row describes a different food item.

Here are some of the columns in the dataset:
+ NDB_No - unique id of the food.
+ Shrt_Desc - name of the food.
+ Water_(g) - water content in grams.
+ Energ_Kcal - energy measured in kilo-calories.
+ Protein_(g) - protein measured in grams.
+ Cholestrl_(mg) - cholesterol in milligrams.

Here's a preview of the first few rows and columns in the dataset:

NDB_No	Shrt_Desc	Water_(g)	Energy_Kcal	Protein_(g)	Lipid_Tot_(g)	Ash_(g)	Carbohydrt_(g)	Fiber_TD_(g)	Sugar_Tot_(g)
1001	BUTTER WITH SALT	15.87	717	0.85	81.11	2.11	0.06	0.0	0.06
1002	BUTTER WHIPPED WITH SALT	15.87	717	0.85	81.11	2.11	0.06	0.0	0.06
1003	BUTTER OIL ANHYDROUS	0.24	876	0.28	99.48	0.00	0.00	0.0	0.0
1004	CHEESE BLUE	42.41	353	21.40	28.74	5.11	2.34	0.0	0.50
1005	CHEESE BRICK	41.11	371	23.24	29.68	3.18	2.79	0.0	0.51

3. Read In A CSV File
To use the Pandas library, we need to import it into the environment using the import keyword:

```python
import pandas
```

We can then refer to the module using pandas and use dot notation to call its methods. To read a CSV file into a dataframe, we use the pandas.read_csv() function and pass in the file name as a string:
```python
# To read in the file `crime_rates.csv` into a dataframe named crime_rates.
crime_rates = pandas.read_csv("crime_rates.csv")
```

You can read more about the parameters the read_csv() method takes to customize how a file is read in on the [documentation page](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html).

## Instructions
+ Import the pandas library.
+ Use the pandas.read_csv() function to read the file "food_info.csv" into a dataframe named food_info.
+ Use the type() and print() functions to display the type of food_info to confirm that it's a dataframe object.
```python
import pandas as pd
food_info = pd.read_csv("food_info.csv")
```

4. Exploring The DataFrame
Now that we've read the dataset into a dataframe, we can start using the dataframe methods to explore the data. To select the first 5 rows of a dataframe, use the dataframe method head(). When you call the __head()__ method, pandas will return a new dataframe containing just the first 5 rows:
```python
first_rows = food_info.head()
```

If you peek at the documentation, you'll notice that you can pass in an integer (n) into the head() method to display the first n rows instead of the first 5:
```python
# First 3 rows.
print(food_info.head(3))
```

Because this dataframe contains many columns and rows, pandas uses ellipsis (...) to hide the columns and rows in the middle. Only the first few and the last few columns and rows are displayed to conserve space.

To access the full list of column names, use the columns attribute:
```python
column_names = food_info.columns
```

Lastly, you can use the shape attribute to understand the dimensions of the dataframe. The shape attribute returns a tuple of integers representing the number of rows followed by the number of columns:
```python
# Returns the tuple (8618,36) and assigns to `dimensions`.
dimensions = food_info.shape
# The number of rows, 8618.
num_rows = dimensions[0]
# The number of columns, 36.
num_cols = dimensions[1]
```

## Instructions
Select the first 20 rows from food_info and assign to the variable first_twenty.
```python
print(food_info.head(3))
dimensions = food_info.shape
print(dimensions)               # (8618, 36)
num_rows = dimensions[0]
print(num_rows)                 # 8618
num_cols = dimensions[1]
print(num_cols)                 # 36
first_twenty = food_info.head(20)
print(first_twenty)
```
