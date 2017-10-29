Files
=====

# Opening Files
To open a file in Python, we use the open() function. This function accepts two different arguments (inputs) in the parentheses, always in the following order:
+ the name of the file (as a _string_)
+ the mode of working with the file (as a _string_)

```python
a = open("story.txt", "r")
```

# Reading In Files
File objects have a __read()__ method that returns a string representation of the text in a file. Unlike the __append()__ method from the previous mission, the __read()__ method returns a value instead of modifying the object that calls the method. In the following code, we use the __read()__ function to read the contents of "test.txt" into a File object, and assign that object to g:

```python
f = open("test.txt", "r")
g = f.read()
```

# Splitting
In Python, we can use the split() method to turn a string object into a list of strings, like so:
["Albuquerque,749", "Anaheim,371", "Anchorage,828"]

The __split()__ method takes a string input corresponding to the delimiter, or separator. This delimiter determines how the string is split into elements in a list. For example, the delimiter for the crime rate data we just looked at is _\n_. Many other files use commas to separate elements:

```python
sample = "john,plastic,joe"
split_list = sample.split(",")
# split_list is a list of _strings_: ["john", "plastic", "joe"]

# We can split a string into a list.
sample = "john,plastic,joe"
split_list = sample.split(",")
print(split_list)

# Here's another example.
string_two = "How much wood\ncan a woodchuck chuck\nif a woodchuck\ncould chuck wood?"
split_string_two = string_two.split('\n')
print(split_string_two)

# Code from previous cells
f = open('crime_rates.csv', 'r')
data = f.read()
rows = data.split('\n')
print(rows[:5])
"""
data ==>
'Albuquerque,749\nAnaheim,371\nAnchorage,828\nArlington,503\nAtlanta,1379\nAurora,425\nAustin,408\nBakersfield,542\nBaltimore,1405\nBoston,835\nBuffalo,1288\nCharlotte-Mecklenburg,647\nCincinnati,974\nCleveland,1383\nColorado Springs,455\nCorpus Christi,658\nDallas,675\nDenver,615\nDetroit,2122\nEl Paso,423\nFort Wayne,362\nFort Worth,587\nFresno,543\nGreensboro,563\nHenderson,168\nHouston,992\nIndianapolis,1185\nJacksonville,617\nJersey City,734\nKansas City,1263\nLas Vegas,784\nLexington,352\nLincoln,397\nLong Beach,575\nLos Angeles,481\nLouisville Metro,598\nMemphis,1750\nMesa,399\nMiami,1172\nMilwaukee,1294\nMinneapolis,992\nMobile,522\nNashville,1216\nNew Orleans,815\nNew York,639\nNewark,1154\nOakland,1993\nOklahoma City,919\nOmaha,594\nPhiladelphia,1160\nPhoenix,636\nPittsburgh,752\nPlano,130\nPortland,517\nRaleigh,423\nRiverside,443\nSacramento,738\nSan Antonio,503\nSan Diego,413\nSan Francisco,704\nSan Jose,363\nSanta Ana,401\nSeattle,597\nSt. Louis,1776\nSt. Paul,722\nStockton,1548\nTampa,616\nToledo,1171\nTucson,724\nTulsa,990\nVirginia Beach,169\nWashington,1177\nWichita,742'
"""
```
