# Pandas APIs

## Common used Pandas code snippets

+ [Basic data information](..Notes/a19-Pandas4DS.md)
  + read in CSV data set: `pd.read_csv("csv_file")`
  + read in an Excel file: `pd.read_excel("excel_file")`
  + write dataframe directly to CSV (w/o indices): `df.to_csv("data.csv", sep=",", index=False)`
  + basic dataset feature info: `df.info`
  + basic dataset statistics: `df.describe()`
  + print dataframe in table: `print(tabulate(tbl, headers=headers))`
    + tbl = list of list, 
    + installation: `pip3 install tabulate`
  + list of column names: `df.columns`

+ [Basic data handling](..Notes/a19-Pandas4DS.md)
  + drop missing data: `df.dropna(axis=0, how='any')`
  + replacing missing data: `to_replace=None, value=None)`
  + check for NaNs: `pd.isnull(object)`
    + numeric array: NaN
    + object array: NaN/None
  + drop a feature: `df.drop('feature_name', axis=1)` w/ `axis=0` for rows, `axis=1` for columns
  + convert object type to float (string to numeric): `pd.to_numeric(df['feature_name'], error="coerce")`
  + convert data frame to Numpy array: `df.as_matrix()`
  + get first $n$ rows of a data frame: `df.head(n)`
  + get data by feature name: `df.loc['feature_name']`

+ [Operating on data frame](..Notes/a19-Pandas4DS.md)
  + apply a function to a data frame: 
  
    ```python
    df['height'].apply(lambda height: 2 * height)

    def multiply(x):
      return 2 * x

    df.['height'].apply(multiply)
    ```

  + renaming a column

    ```python
    # rename 3rd col as 'size
    df.rename(column={df.columns[2]: 'size'}, inplace=true)
    ```

  + get the unique entries of a column: `df['name'].unique()`
  + accessing sub-data frames:

    ```python
    # grab a selection of the columns, 'name' and 'size'
    new_df = df[['name', 'size']]`
    ```

  + summary info about data

    ```python
    # sum of values in a data frame
    df.sum()

    # lowest value of a data frame
    df.min()

    # highest value of a data frame
    df.max()

    # index of the lowest value
    df.idxmin()

    # index of the highest value
    df.idxmax()

    # statistical summary of the data frame
    df.describe()

    # average value of a data frame
    df.mean()

    # median value of a data frame
    df.median()

    # correlation btw columns
    df.corr()

    # median value of a specific column
    df['size'].median()
    ```

  + sorting data: `df_sort_values(ascending=False)`
  + Boolean indexing

    ```python
    # filter data column 'size' w/ value = 5
    df[df['size'] == 5]
    ```

  + selecting values:

    ```python
    # select nth row of a column `size`
    df.loc([n-1], ['size'])
    ```

## DataFrame Attributes

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y5vn7xyd">Attributes and Underlying data</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.index.html#pandas.DataFrame.index" title="pandas.DataFrame.index"><code>DataFrame.index</code></a></td>
      <td>The index (row labels) of the DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html#pandas.DataFrame.columns" title="pandas.DataFrame.columns"><code>DataFrame.columns</code></a></td>
      <td>The column labels of the DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html#pandas.DataFrame.dtypes" title="pandas.DataFrame.dtypes"><code>DataFrame.dtypes</code></a></td>
      <td>Return the dtypes in the DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes" title="pandas.DataFrame.select_dtypes"><code>DataFrame.select_dtypes</code></a>(self[,&nbsp;include,&nbsp;exclude])</td>
      <td>Return a subset of the DataFrame’s columns based on the column dtypes.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html#pandas.DataFrame.values" title="pandas.DataFrame.values"><code>DataFrame.values</code></a></td>
      <td>Return a Numpy representation of the DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.axes.html#pandas.DataFrame.axes" title="pandas.DataFrame.axes"><code>DataFrame.axes</code></a></td>
      <td>Return a list representing the axes of the DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ndim.html#pandas.DataFrame.ndim" title="pandas.DataFrame.ndim"><code>DataFrame.ndim</code></a></td>
      <td>Return an int representing the number of axes / array dimensions.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.size.html#pandas.DataFrame.size" title="pandas.DataFrame.size"><code>DataFrame.size</code></a></td>
      <td>Return an int representing the number of elements in this object.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html#pandas.DataFrame.shape" title="pandas.DataFrame.shape"><code>DataFrame.shape</code></a></td>
      <td>Return a tuple representing the dimensionality of the DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.memory_usage.html#pandas.DataFrame.memory_usage" title="pandas.DataFrame.memory_usage"><code>DataFrame.memory_usage</code></a>(self[,&nbsp;index,&nbsp;deep])</td>
      <td>Return the memory usage of each column in bytes.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.empty.html#pandas.DataFrame.empty" title="pandas.DataFrame.empty"><code>DataFrame.empty</code></a></td>
      <td>Indicator whether DataFrame is empty.</td>
    </tr>
    </tbody>
  </table>


## Dataframe Indexing & Iteration

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y6788rm2">Indexing and Iteration</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html#pandas.DataFrame.head" title="pandas.DataFrame.head"><code>DataFrame.head</code></a>(self,&nbsp;n)</td>
      <td>Return the first <cite>n</cite> rows.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at" title="pandas.DataFrame.at"><code>DataFrame.at</code></a></td>
      <td>Access a single value for a row/column label pair.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iat.html#pandas.DataFrame.iat" title="pandas.DataFrame.iat"><code>DataFrame.iat</code></a></td>
      <td>Access a single value for a row/column pair by integer position.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html#pandas.DataFrame.loc" title="pandas.DataFrame.loc"><code>DataFrame.loc</code></a></td>
      <td>Access a group of rows and columns by label(s) or a boolean array.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc" title="pandas.DataFrame.iloc"><code>DataFrame.iloc</code></a></td>
      <td>Purely integer-location based indexing for selection by position.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.insert.html#pandas.DataFrame.insert" title="pandas.DataFrame.insert"><code>DataFrame.insert</code></a>(self,&nbsp;loc,&nbsp;column,&nbsp;value[,&nbsp;…])</td>
      <td>Insert column into DataFrame at specified location.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.__iter__.html#pandas.DataFrame.__iter__" title="pandas.DataFrame.__iter__"><code>DataFrame.__iter__</code></a>(self)</td>
      <td>Iterate over info axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.items.html#pandas.DataFrame.items" title="pandas.DataFrame.items"><code>DataFrame.items</code></a>(self)</td>
      <td>Iterate over (column name, Series) pairs.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iteritems.html#pandas.DataFrame.iteritems" title="pandas.DataFrame.iteritems"><code>DataFrame.iteritems</code></a>(self)</td>
      <td>Iterate over (column name, Series) pairs.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.keys.html#pandas.DataFrame.keys" title="pandas.DataFrame.keys"><code>DataFrame.keys</code></a>(self)</td>
      <td>Get the ‘info axis’ (see Indexing for more).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows" title="pandas.DataFrame.iterrows"><code>DataFrame.iterrows</code></a>(self)</td>
      <td>Iterate over DataFrame rows as (index, Series) pairs.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.itertuples.html#pandas.DataFrame.itertuples" title="pandas.DataFrame.itertuples"><code>DataFrame.itertuples</code></a>(self[,&nbsp;index,&nbsp;name])</td>
      <td>Iterate over DataFrame rows as namedtuples.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.lookup.html#pandas.DataFrame.lookup" title="pandas.DataFrame.lookup"><code>DataFrame.lookup</code></a>(self,&nbsp;row_labels,&nbsp;col_labels)</td>
      <td>Label-based “fancy indexing” function for DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pop.html#pandas.DataFrame.pop" title="pandas.DataFrame.pop"><code>DataFrame.pop</code></a>(self,&nbsp;item)</td>
      <td>Return item and drop from frame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html#pandas.DataFrame.tail" title="pandas.DataFrame.tail"><code>DataFrame.tail</code></a>(self,&nbsp;n)</td>
      <td>Return the last <cite>n</cite> rows.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.xs.html#pandas.DataFrame.xs" title="pandas.DataFrame.xs"><code>DataFrame.xs</code></a>(self,&nbsp;key[,&nbsp;axis,&nbsp;level])</td>
      <td>Return cross-section from the Series/DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.get.html#pandas.DataFrame.get" title="pandas.DataFrame.get"><code>DataFrame.get</code></a>(self,&nbsp;key[,&nbsp;default])</td>
      <td>Get item from object for given key (ex: DataFrame column).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isin.html#pandas.DataFrame.isin" title="pandas.DataFrame.isin"><code>DataFrame.isin</code></a>(self,&nbsp;values)</td>
      <td>Whether each element in the DataFrame is contained in values.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.where.html#pandas.DataFrame.where" title="pandas.DataFrame.where"><code>DataFrame.where</code></a>(self,&nbsp;cond[,&nbsp;other,&nbsp;…])</td>
      <td>Replace values where the condition is False.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mask.html#pandas.DataFrame.mask" title="pandas.DataFrame.mask"><code>DataFrame.mask</code></a>(self,&nbsp;cond[,&nbsp;other,&nbsp;inplace,&nbsp;…])</td>
      <td>Replace values where the condition is True.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html#pandas.DataFrame.query" title="pandas.DataFrame.query"><code>DataFrame.query</code></a>(self,&nbsp;expr[,&nbsp;inplace])</td>
      <td>Query the columns of a DataFrame with a boolean expression.</td>
    </tr>
    </tbody>
  </table>


## DataFrame Binary Operators

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yxqwpewf">Binary Operator Functions</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.add.html#pandas.DataFrame.add" title="pandas.DataFrame.add"><code>DataFrame.add</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Addition of dataframe and other, element-wise (binary operator <cite>add</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sub.html#pandas.DataFrame.sub" title="pandas.DataFrame.sub"><code>DataFrame.sub</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Subtraction of dataframe and other, element-wise (binary operator <cite>sub</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mul.html#pandas.DataFrame.mul" title="pandas.DataFrame.mul"><code>DataFrame.mul</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Multiplication of dataframe and other, element-wise (binary operator <cite>mul</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.div.html#pandas.DataFrame.div" title="pandas.DataFrame.div"><code>DataFrame.div</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Floating division of dataframe and other, element-wise (binary operator <cite>truediv</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.truediv.html#pandas.DataFrame.truediv" title="pandas.DataFrame.truediv"><code>DataFrame.truediv</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;…])</td>
      <td>Get Floating division of dataframe and other, element-wise (binary operator <cite>truediv</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.floordiv.html#pandas.DataFrame.floordiv" title="pandas.DataFrame.floordiv"><code>DataFrame.floordiv</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;…])</td>
      <td>Get Integer division of dataframe and other, element-wise (binary operator <cite>floordiv</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mod.html#pandas.DataFrame.mod" title="pandas.DataFrame.mod"><code>DataFrame.mod</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Modulo of dataframe and other, element-wise (binary operator <cite>mod</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pow.html#pandas.DataFrame.pow" title="pandas.DataFrame.pow"><code>DataFrame.pow</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Exponential power of dataframe and other, element-wise (binary operator <cite>pow</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dot.html#pandas.DataFrame.dot" title="pandas.DataFrame.dot"><code>DataFrame.dot</code></a>(self,&nbsp;other)</td>
      <td>Compute the matrix multiplication between the DataFrame and other.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.radd.html#pandas.DataFrame.radd" title="pandas.DataFrame.radd"><code>DataFrame.radd</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Addition of dataframe and other, element-wise (binary operator <cite>radd</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rsub.html#pandas.DataFrame.rsub" title="pandas.DataFrame.rsub"><code>DataFrame.rsub</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Subtraction of dataframe and other, element-wise (binary operator <cite>rsub</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rmul.html#pandas.DataFrame.rmul" title="pandas.DataFrame.rmul"><code>DataFrame.rmul</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Multiplication of dataframe and other, element-wise (binary operator <cite>rmul</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rdiv.html#pandas.DataFrame.rdiv" title="pandas.DataFrame.rdiv"><code>DataFrame.rdiv</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Floating division of dataframe and other, element-wise (binary operator <cite>rtruediv</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rtruediv.html#pandas.DataFrame.rtruediv" title="pandas.DataFrame.rtruediv"><code>DataFrame.rtruediv</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;…])</td>
      <td>Get Floating division of dataframe and other, element-wise (binary operator <cite>rtruediv</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rfloordiv.html#pandas.DataFrame.rfloordiv" title="pandas.DataFrame.rfloordiv"><code>DataFrame.rfloordiv</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;…])</td>
      <td>Get Integer division of dataframe and other, element-wise (binary operator <cite>rfloordiv</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rmod.html#pandas.DataFrame.rmod" title="pandas.DataFrame.rmod"><code>DataFrame.rmod</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Modulo of dataframe and other, element-wise (binary operator <cite>rmod</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rpow.html#pandas.DataFrame.rpow" title="pandas.DataFrame.rpow"><code>DataFrame.rpow</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Get Exponential power of dataframe and other, element-wise (binary operator <cite>rpow</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.lt.html#pandas.DataFrame.lt" title="pandas.DataFrame.lt"><code>DataFrame.lt</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level])</td>
      <td>Get Less than of dataframe and other, element-wise (binary operator <cite>lt</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.gt.html#pandas.DataFrame.gt" title="pandas.DataFrame.gt"><code>DataFrame.gt</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level])</td>
      <td>Get Greater than of dataframe and other, element-wise (binary operator <cite>gt</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.le.html#pandas.DataFrame.le" title="pandas.DataFrame.le"><code>DataFrame.le</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level])</td>
      <td>Get Less than or equal to of dataframe and other, element-wise (binary operator <cite>le</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ge.html#pandas.DataFrame.ge" title="pandas.DataFrame.ge"><code>DataFrame.ge</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level])</td>
      <td>Get Greater than or equal to of dataframe and other, element-wise (binary operator <cite>ge</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ne.html#pandas.DataFrame.ne" title="pandas.DataFrame.ne"><code>DataFrame.ne</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level])</td>
      <td>Get Not equal to of dataframe and other, element-wise (binary operator <cite>ne</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.eq.html#pandas.DataFrame.eq" title="pandas.DataFrame.eq"><code>DataFrame.eq</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;level])</td>
      <td>Get Equal to of dataframe and other, element-wise (binary operator <cite>eq</cite>).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.combine.html#pandas.DataFrame.combine" title="pandas.DataFrame.combine"><code>DataFrame.combine</code></a>(self,&nbsp;other,&nbsp;func[,&nbsp;…])</td>
      <td>Perform column-wise combine with another DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.combine_first.html#pandas.DataFrame.combine_first" title="pandas.DataFrame.combine_first"><code>DataFrame.combine_first</code></a>(self,&nbsp;other)</td>
      <td>Update null elements with value in the same location in <cite>other</cite>.</td>
    </tr>
    </tbody>
  </table>


## DataFrame Fucntions

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y424dm6t">Function application, GroupBy & window</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply" title="pandas.DataFrame.apply"><code>DataFrame.apply</code></a>(self,&nbsp;func[,&nbsp;axis,&nbsp;raw,&nbsp;…])</td>
      <td>Apply a function along an axis of the DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.applymap.html#pandas.DataFrame.applymap" title="pandas.DataFrame.applymap"><code>DataFrame.applymap</code></a>(self,&nbsp;func)</td>
      <td>Apply a function to a Dataframe elementwise.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pipe.html#pandas.DataFrame.pipe" title="pandas.DataFrame.pipe"><code>DataFrame.pipe</code></a>(self,&nbsp;func,&nbsp;*args,&nbsp;**kwargs)</td>
      <td>Apply func(self, *args, **kwargs).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html#pandas.DataFrame.agg" title="pandas.DataFrame.agg"><code>DataFrame.agg</code></a>(self,&nbsp;func[,&nbsp;axis])</td>
      <td>Aggregate using one or more operations over the specified axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.aggregate.html#pandas.DataFrame.aggregate" title="pandas.DataFrame.aggregate"><code>DataFrame.aggregate</code></a>(self,&nbsp;func[,&nbsp;axis])</td>
      <td>Aggregate using one or more operations over the specified axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transform.html#pandas.DataFrame.transform" title="pandas.DataFrame.transform"><code>DataFrame.transform</code></a>(self,&nbsp;func[,&nbsp;axis])</td>
      <td>Call <code>func</code> on self producing a DataFrame with transformed values.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby" title="pandas.DataFrame.groupby"><code>DataFrame.groupby</code></a>(self[,&nbsp;by,&nbsp;axis,&nbsp;level])</td>
      <td>Group DataFrame using a mapper or by a Series of columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html#pandas.DataFrame.rolling" title="pandas.DataFrame.rolling"><code>DataFrame.rolling</code></a>(self,&nbsp;window[,&nbsp;…])</td>
      <td>Provide rolling window calculations.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.expanding.html#pandas.DataFrame.expanding" title="pandas.DataFrame.expanding"><code>DataFrame.expanding</code></a>(self[,&nbsp;min_periods,&nbsp;…])</td>
      <td>Provide expanding transformations.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html#pandas.DataFrame.ewm" title="pandas.DataFrame.ewm"><code>DataFrame.ewm</code></a>(self[,&nbsp;com,&nbsp;span,&nbsp;halflife,&nbsp;…])</td>
      <td>Provide exponential weighted functions.</td>
    </tr>
    </tbody>
  </table>


## DataFrame Statistics

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y23vpvaa">Computations / descriptive stats</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.abs.html#pandas.DataFrame.abs" title="pandas.DataFrame.abs"><code>DataFrame.abs</code></a>(self)</td>
      <td>Return a Series/DataFrame with absolute numeric value of each element.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.all.html#pandas.DataFrame.all" title="pandas.DataFrame.all"><code>DataFrame.all</code></a>(self[,&nbsp;axis,&nbsp;bool_only,&nbsp;…])</td>
      <td>Return whether all elements are True, potentially over an axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.any.html#pandas.DataFrame.any" title="pandas.DataFrame.any"><code>DataFrame.any</code></a>(self[,&nbsp;axis,&nbsp;bool_only,&nbsp;…])</td>
      <td>Return whether any element is True, potentially over an axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.clip.html#pandas.DataFrame.clip" title="pandas.DataFrame.clip"><code>DataFrame.clip</code></a>(self[,&nbsp;lower,&nbsp;upper,&nbsp;axis])</td>
      <td>Trim values at input threshold(s).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html#pandas.DataFrame.corr" title="pandas.DataFrame.corr"><code>DataFrame.corr</code></a>(self[,&nbsp;method,&nbsp;min_periods])</td>
      <td>Compute pairwise correlation of columns, excluding NA/null values.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corrwith.html#pandas.DataFrame.corrwith" title="pandas.DataFrame.corrwith"><code>DataFrame.corrwith</code></a>(self,&nbsp;other[,&nbsp;axis,&nbsp;…])</td>
      <td>Compute pairwise correlation.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.count.html#pandas.DataFrame.count" title="pandas.DataFrame.count"><code>DataFrame.count</code></a>(self[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Count non-NA cells for each column or row.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cov.html#pandas.DataFrame.cov" title="pandas.DataFrame.cov"><code>DataFrame.cov</code></a>(self[,&nbsp;min_periods])</td>
      <td>Compute pairwise covariance of columns, excluding NA/null values.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cummax.html#pandas.DataFrame.cummax" title="pandas.DataFrame.cummax"><code>DataFrame.cummax</code></a>(self[,&nbsp;axis,&nbsp;skipna])</td>
      <td>Return cumulative maximum over a DataFrame or Series axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cummin.html#pandas.DataFrame.cummin" title="pandas.DataFrame.cummin"><code>DataFrame.cummin</code></a>(self[,&nbsp;axis,&nbsp;skipna])</td>
      <td>Return cumulative minimum over a DataFrame or Series axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cumprod.html#pandas.DataFrame.cumprod" title="pandas.DataFrame.cumprod"><code>DataFrame.cumprod</code></a>(self[,&nbsp;axis,&nbsp;skipna])</td>
      <td>Return cumulative product over a DataFrame or Series axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cumsum.html#pandas.DataFrame.cumsum" title="pandas.DataFrame.cumsum"><code>DataFrame.cumsum</code></a>(self[,&nbsp;axis,&nbsp;skipna])</td>
      <td>Return cumulative sum over a DataFrame or Series axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html#pandas.DataFrame.describe" title="pandas.DataFrame.describe"><code>DataFrame.describe</code></a>(self[,&nbsp;percentiles,&nbsp;…])</td>
      <td>Generate descriptive statistics.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.diff.html#pandas.DataFrame.diff" title="pandas.DataFrame.diff"><code>DataFrame.diff</code></a>(self[,&nbsp;periods,&nbsp;axis])</td>
      <td>First discrete difference of element.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.eval.html#pandas.DataFrame.eval" title="pandas.DataFrame.eval"><code>DataFrame.eval</code></a>(self,&nbsp;expr[,&nbsp;inplace])</td>
      <td>Evaluate a string describing operations on DataFrame columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.kurt.html#pandas.DataFrame.kurt" title="pandas.DataFrame.kurt"><code>DataFrame.kurt</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return unbiased kurtosis over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.kurtosis.html#pandas.DataFrame.kurtosis" title="pandas.DataFrame.kurtosis"><code>DataFrame.kurtosis</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;…])</td>
      <td>Return unbiased kurtosis over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mad.html#pandas.DataFrame.mad" title="pandas.DataFrame.mad"><code>DataFrame.mad</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level])</td>
      <td>Return the mean absolute deviation of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.max.html#pandas.DataFrame.max" title="pandas.DataFrame.max"><code>DataFrame.max</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return the maximum of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html#pandas.DataFrame.mean" title="pandas.DataFrame.mean"><code>DataFrame.mean</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return the mean of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.median.html#pandas.DataFrame.median" title="pandas.DataFrame.median"><code>DataFrame.median</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;…])</td>
      <td>Return the median of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.min.html#pandas.DataFrame.min" title="pandas.DataFrame.min"><code>DataFrame.min</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return the minimum of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mode.html#pandas.DataFrame.mode" title="pandas.DataFrame.mode"><code>DataFrame.mode</code></a>(self[,&nbsp;axis,&nbsp;numeric_only,&nbsp;…])</td>
      <td>Get the mode(s) of each element along the selected axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html#pandas.DataFrame.pct_change" title="pandas.DataFrame.pct_change"><code>DataFrame.pct_change</code></a>(self[,&nbsp;periods,&nbsp;…])</td>
      <td>Percentage change between the current and a prior element.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.prod.html#pandas.DataFrame.prod" title="pandas.DataFrame.prod"><code>DataFrame.prod</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return the product of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.product.html#pandas.DataFrame.product" title="pandas.DataFrame.product"><code>DataFrame.product</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;…])</td>
      <td>Return the product of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html#pandas.DataFrame.quantile" title="pandas.DataFrame.quantile"><code>DataFrame.quantile</code></a>(self[,&nbsp;q,&nbsp;axis,&nbsp;…])</td>
      <td>Return values at the given quantile over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html#pandas.DataFrame.rank" title="pandas.DataFrame.rank"><code>DataFrame.rank</code></a>(self[,&nbsp;axis])</td>
      <td>Compute numerical data ranks (1 through n) along axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.round.html#pandas.DataFrame.round" title="pandas.DataFrame.round"><code>DataFrame.round</code></a>(self[,&nbsp;decimals])</td>
      <td>Round a DataFrame to a variable number of decimal places.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sem.html#pandas.DataFrame.sem" title="pandas.DataFrame.sem"><code>DataFrame.sem</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return unbiased standard error of the mean over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.skew.html#pandas.DataFrame.skew" title="pandas.DataFrame.skew"><code>DataFrame.skew</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return unbiased skew over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html#pandas.DataFrame.sum" title="pandas.DataFrame.sum"><code>DataFrame.sum</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return the sum of the values for the requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.std.html#pandas.DataFrame.std" title="pandas.DataFrame.std"><code>DataFrame.std</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return sample standard deviation over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.var.html#pandas.DataFrame.var" title="pandas.DataFrame.var"><code>DataFrame.var</code></a>(self[,&nbsp;axis,&nbsp;skipna,&nbsp;level,&nbsp;…])</td>
      <td>Return unbiased variance over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html#pandas.DataFrame.nunique" title="pandas.DataFrame.nunique"><code>DataFrame.nunique</code></a>(self[,&nbsp;axis,&nbsp;dropna])</td>
      <td>Count distinct observations over requested axis.</td>
    </tr>
    </tbody>
  </table>


## DataFrame Indexing & Manipulation

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Reindexing / selection / label manipulation</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.add_prefix.html#pandas.DataFrame.add_prefix" title="pandas.DataFrame.add_prefix"><code>DataFrame.add_prefix</code></a>(self,&nbsp;prefix)</td>
      <td>Prefix labels with string <cite>prefix</cite>.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.add_suffix.html#pandas.DataFrame.add_suffix" title="pandas.DataFrame.add_suffix"><code>DataFrame.add_suffix</code></a>(self,&nbsp;suffix)</td>
      <td>Suffix labels with string <cite>suffix</cite>.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.align.html#pandas.DataFrame.align" title="pandas.DataFrame.align"><code>DataFrame.align</code></a>(self,&nbsp;other[,&nbsp;join,&nbsp;axis,&nbsp;…])</td>
      <td>Align two objects on their axes with the specified join method.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at_time.html#pandas.DataFrame.at_time" title="pandas.DataFrame.at_time"><code>DataFrame.at_time</code></a>(self,&nbsp;time,&nbsp;asof[,&nbsp;axis])</td>
      <td>Select values at particular time of day (e.g.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.between_time.html#pandas.DataFrame.between_time" title="pandas.DataFrame.between_time"><code>DataFrame.between_time</code></a>(self,&nbsp;start_time,&nbsp;…)</td>
      <td>Select values between particular times of the day (e.g., 9:00-9:30 AM).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop" title="pandas.DataFrame.drop"><code>DataFrame.drop</code></a>(self[,&nbsp;labels,&nbsp;axis,&nbsp;index,&nbsp;…])</td>
      <td>Drop specified labels from rows or columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html#pandas.DataFrame.drop_duplicates" title="pandas.DataFrame.drop_duplicates"><code>DataFrame.drop_duplicates</code></a>(self,&nbsp;subset,&nbsp;…)</td>
      <td>Return DataFrame with duplicate rows removed.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html#pandas.DataFrame.duplicated" title="pandas.DataFrame.duplicated"><code>DataFrame.duplicated</code></a>(self,&nbsp;subset,&nbsp;…)</td>
      <td>Return boolean Series denoting duplicate rows.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.equals.html#pandas.DataFrame.equals" title="pandas.DataFrame.equals"><code>DataFrame.equals</code></a>(self,&nbsp;other)</td>
      <td>Test whether two objects contain the same elements.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html#pandas.DataFrame.filter" title="pandas.DataFrame.filter"><code>DataFrame.filter</code></a>(self[,&nbsp;items,&nbsp;axis])</td>
      <td>Subset the dataframe rows or columns according to the specified index labels.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.first.html#pandas.DataFrame.first" title="pandas.DataFrame.first"><code>DataFrame.first</code></a>(self,&nbsp;offset)</td>
      <td>Method to subset initial periods of time series data based on a date offset.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html#pandas.DataFrame.head" title="pandas.DataFrame.head"><code>DataFrame.head</code></a>(self,&nbsp;n)</td>
      <td>Return the first <cite>n</cite> rows.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.idxmax.html#pandas.DataFrame.idxmax" title="pandas.DataFrame.idxmax"><code>DataFrame.idxmax</code></a>(self[,&nbsp;axis,&nbsp;skipna])</td>
      <td>Return index of first occurrence of maximum over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.idxmin.html#pandas.DataFrame.idxmin" title="pandas.DataFrame.idxmin"><code>DataFrame.idxmin</code></a>(self[,&nbsp;axis,&nbsp;skipna])</td>
      <td>Return index of first occurrence of minimum over requested axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.last.html#pandas.DataFrame.last" title="pandas.DataFrame.last"><code>DataFrame.last</code></a>(self,&nbsp;offset)</td>
      <td>Method to subset final periods of time series data based on a date offset.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html#pandas.DataFrame.reindex" title="pandas.DataFrame.reindex"><code>DataFrame.reindex</code></a>(self[,&nbsp;labels,&nbsp;index,&nbsp;…])</td>
      <td>Conform DataFrame to new index with optional filling logic.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex_like.html#pandas.DataFrame.reindex_like" title="pandas.DataFrame.reindex_like"><code>DataFrame.reindex_like</code></a>(self,&nbsp;other,&nbsp;method,&nbsp;…)</td>
      <td>Return an object with matching indices as other object.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html#pandas.DataFrame.rename" title="pandas.DataFrame.rename"><code>DataFrame.rename</code></a>(self[,&nbsp;mapper,&nbsp;index,&nbsp;…])</td>
      <td>Alter axes labels.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename_axis.html#pandas.DataFrame.rename_axis" title="pandas.DataFrame.rename_axis"><code>DataFrame.rename_axis</code></a>(self[,&nbsp;mapper,&nbsp;index,&nbsp;…])</td>
      <td>Set the name of the axis for the index or columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html#pandas.DataFrame.reset_index" title="pandas.DataFrame.reset_index"><code>DataFrame.reset_index</code></a>(self,&nbsp;level,&nbsp;…)</td>
      <td>Reset the index, or a level of it.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html#pandas.DataFrame.sample" title="pandas.DataFrame.sample"><code>DataFrame.sample</code></a>(self[,&nbsp;n,&nbsp;frac,&nbsp;replace,&nbsp;…])</td>
      <td>Return a random sample of items from an axis of object.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_axis.html#pandas.DataFrame.set_axis" title="pandas.DataFrame.set_axis"><code>DataFrame.set_axis</code></a>(self,&nbsp;labels[,&nbsp;axis,&nbsp;inplace])</td>
      <td>Assign desired index to given axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html#pandas.DataFrame.set_index" title="pandas.DataFrame.set_index"><code>DataFrame.set_index</code></a>(self,&nbsp;keys[,&nbsp;drop,&nbsp;…])</td>
      <td>Set the DataFrame index using existing columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html#pandas.DataFrame.tail" title="pandas.DataFrame.tail"><code>DataFrame.tail</code></a>(self,&nbsp;n)</td>
      <td>Return the last <cite>n</cite> rows.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.take.html#pandas.DataFrame.take" title="pandas.DataFrame.take"><code>DataFrame.take</code></a>(self,&nbsp;indices[,&nbsp;axis])</td>
      <td>Return the elements in the given <em>positional</em> indices along an axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.truncate.html#pandas.DataFrame.truncate" title="pandas.DataFrame.truncate"><code>DataFrame.truncate</code></a>(self[,&nbsp;before,&nbsp;after,&nbsp;axis])</td>
      <td>Truncate a Series or DataFrame before and after some index value.</td>
    </tr>
    </tbody>
  </table>


## DataFrame Missing Data

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y9my23yz">Missing Data Handling</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.backfill.html#pandas.DataFrame.backfill" title="pandas.DataFrame.backfill"><code>DataFrame.backfill</code></a>([axis,&nbsp;inplace,&nbsp;limit,&nbsp;…])</td>
    <td>Synonym for <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna" title="pandas.DataFrame.fillna"><code>DataFrame.fillna()</code></a> with <code>method='bfill'</code>.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.bfill.html#pandas.DataFrame.bfill" title="pandas.DataFrame.bfill"><code>DataFrame.bfill</code></a>([axis,&nbsp;inplace,&nbsp;limit,&nbsp;downcast])</td>
    <td>Synonym for <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna" title="pandas.DataFrame.fillna"><code>DataFrame.fillna()</code></a> with <code>method='bfill'</code>.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna" title="pandas.DataFrame.dropna"><code>DataFrame.dropna</code></a>([axis,&nbsp;how,&nbsp;thresh,&nbsp;…])</td>
    <td>Remove missing values.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html#pandas.DataFrame.ffill" title="pandas.DataFrame.ffill"><code>DataFrame.ffill</code></a>([axis,&nbsp;inplace,&nbsp;limit,&nbsp;downcast])</td>
    <td>Synonym for <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna" title="pandas.DataFrame.fillna"><code>DataFrame.fillna()</code></a> with <code>method='ffill'</code>.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna" title="pandas.DataFrame.fillna"><code>DataFrame.fillna</code></a>([value,&nbsp;method,&nbsp;axis,&nbsp;…])</td>
    <td>Fill NA/NaN values using the specified method.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate" title="pandas.DataFrame.interpolate"><code>DataFrame.interpolate</code></a>([method,&nbsp;axis,&nbsp;limit,&nbsp;…])</td>
    <td>Please note that only <code>method='linear'</code> is supported for DataFrame/Series with a MultiIndex.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html#pandas.DataFrame.isna" title="pandas.DataFrame.isna"><code>DataFrame.isna</code></a>()</td>
    <td>Detect missing values.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html#pandas.DataFrame.isnull" title="pandas.DataFrame.isnull"><code>DataFrame.isnull</code></a>()</td>
    <td>Detect missing values.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.notna.html#pandas.DataFrame.notna" title="pandas.DataFrame.notna"><code>DataFrame.notna</code></a>()</td>
    <td>Detect existing (non-missing) values.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.notnull.html#pandas.DataFrame.notnull" title="pandas.DataFrame.notnull"><code>DataFrame.notnull</code></a>()</td>
    <td>Detect existing (non-missing) values.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pad.html#pandas.DataFrame.pad" title="pandas.DataFrame.pad"><code>DataFrame.pad</code></a>([axis,&nbsp;inplace,&nbsp;limit,&nbsp;downcast])</td>
    <td>Synonym for <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna" title="pandas.DataFrame.fillna"><code>DataFrame.fillna()</code></a> with <code>method='ffill'</code>.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html#pandas.DataFrame.replace" title="pandas.DataFrame.replace"><code>DataFrame.replace</code></a>([to_replace,&nbsp;value,&nbsp;…])</td>
    <td>Replace values given in <cite>to_replace</cite> with <cite>value</cite>.</td>
    </tr>
    </tbody>
  </table>


## Pandas Dataframe Reshaping, Sorting, & Transposing

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y65kz8nw">Reshaping, sorting, transposing</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.droplevel.html#pandas.DataFrame.droplevel" title="pandas.DataFrame.droplevel"><code>DataFrame.droplevel</code></a>(self,&nbsp;level[,&nbsp;axis])</td>
      <td>Return DataFrame with requested index / column level(s) removed.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html#pandas.DataFrame.pivot" title="pandas.DataFrame.pivot"><code>DataFrame.pivot</code></a>(self[,&nbsp;index,&nbsp;columns,&nbsp;values])</td>
      <td>Return reshaped DataFrame organized by given index / column values.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html#pandas.DataFrame.pivot_table" title="pandas.DataFrame.pivot_table"><code>DataFrame.pivot_table</code></a>(self[,&nbsp;values,&nbsp;index,&nbsp;…])</td>
      <td>Create a spreadsheet-style pivot table as a DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reorder_levels.html#pandas.DataFrame.reorder_levels" title="pandas.DataFrame.reorder_levels"><code>DataFrame.reorder_levels</code></a>(self,&nbsp;order[,&nbsp;axis])</td>
      <td>Rearrange index levels using input order.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html#pandas.DataFrame.sort_values" title="pandas.DataFrame.sort_values"><code>DataFrame.sort_values</code></a>(self,&nbsp;by[,&nbsp;axis,&nbsp;…])</td>
      <td>Sort by the values along either axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_index.html#pandas.DataFrame.sort_index" title="pandas.DataFrame.sort_index"><code>DataFrame.sort_index</code></a>(self[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
      <td>Sort object by labels (along an axis).</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html#pandas.DataFrame.nlargest" title="pandas.DataFrame.nlargest"><code>DataFrame.nlargest</code></a>(self,&nbsp;n,&nbsp;columns[,&nbsp;keep])</td>
      <td>Return the first <cite>n</cite> rows ordered by <cite>columns</cite> in descending order.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nsmallest.html#pandas.DataFrame.nsmallest" title="pandas.DataFrame.nsmallest"><code>DataFrame.nsmallest</code></a>(self,&nbsp;n,&nbsp;columns[,&nbsp;keep])</td>
      <td>Return the first <cite>n</cite> rows ordered by <cite>columns</cite> in ascending order.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.swaplevel.html#pandas.DataFrame.swaplevel" title="pandas.DataFrame.swaplevel"><code>DataFrame.swaplevel</code></a>(self[,&nbsp;i,&nbsp;j,&nbsp;axis])</td>
      <td>Swap levels i and j in a MultiIndex on a particular axis.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html#pandas.DataFrame.stack" title="pandas.DataFrame.stack"><code>DataFrame.stack</code></a>(self[,&nbsp;level,&nbsp;dropna])</td>
      <td>Stack the prescribed level(s) from columns to index.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html#pandas.DataFrame.unstack" title="pandas.DataFrame.unstack"><code>DataFrame.unstack</code></a>(self[,&nbsp;level,&nbsp;fill_value])</td>
      <td>Pivot a level of the (necessarily hierarchical) index labels.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.swapaxes.html#pandas.DataFrame.swapaxes" title="pandas.DataFrame.swapaxes"><code>DataFrame.swapaxes</code></a>(self,&nbsp;axis1,&nbsp;axis2[,&nbsp;copy])</td>
      <td>Interchange axes and swap values axes appropriately.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html#pandas.DataFrame.melt" title="pandas.DataFrame.melt"><code>DataFrame.melt</code></a>(self[,&nbsp;id_vars,&nbsp;value_vars,&nbsp;…])</td>
      <td>Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html#pandas.DataFrame.explode" title="pandas.DataFrame.explode"><code>DataFrame.explode</code></a>(self,&nbsp;column,&nbsp;Tuple])</td>
      <td>Transform each element of a list-like to a row, replicating index values.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.squeeze.html#pandas.DataFrame.squeeze" title="pandas.DataFrame.squeeze"><code>DataFrame.squeeze</code></a>(self[,&nbsp;axis])</td>
      <td>Squeeze 1 dimensional axis objects into scalars.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_xarray.html#pandas.DataFrame.to_xarray" title="pandas.DataFrame.to_xarray"><code>DataFrame.to_xarray</code></a>(self)</td>
      <td>Return an xarray object from the pandas object.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.T.html#pandas.DataFrame.T" title="pandas.DataFrame.T"><code>DataFrame.T</code></a></td>
      <td>Transpose index and columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transpose.html#pandas.DataFrame.transpose" title="pandas.DataFrame.transpose"><code>DataFrame.transpose</code></a>(self,&nbsp;*args,&nbsp;copy)</td>
      <td>Transpose index and columns.</td>
    </tr>
    </tbody>
  </table>

## DataFrames Manupulation 

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y3bcu5e5">Combining / joining / merging</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.append.html#pandas.DataFrame.append" title="pandas.DataFrame.append"><code>DataFrame.append</code></a>(self,&nbsp;other[,&nbsp;…])</td>
      <td>Append rows of <cite>other</cite> to the end of caller, returning a new object.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html#pandas.DataFrame.assign" title="pandas.DataFrame.assign"><code>DataFrame.assign</code></a>(self,&nbsp;**kwargs)</td>
      <td>Assign new columns to a DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html#pandas.DataFrame.join" title="pandas.DataFrame.join"><code>DataFrame.join</code></a>(self,&nbsp;other[,&nbsp;on,&nbsp;how,&nbsp;…])</td>
      <td>Join columns of another DataFrame.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge" title="pandas.DataFrame.merge"><code>DataFrame.merge</code></a>(self,&nbsp;right[,&nbsp;how,&nbsp;on,&nbsp;…])</td>
      <td>Merge DataFrame or named Series objects with a database-style join.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.update.html#pandas.DataFrame.update" title="pandas.DataFrame.update"><code>DataFrame.update</code></a>(self,&nbsp;other[,&nbsp;join,&nbsp;…])</td>
      <td>Modify in place using non-NA values from another DataFrame.</td>
    </tr>
    </tbody>
  </table>



## DataFrame Time Series

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y7zntevu">Time Series-related</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asfreq.html#pandas.DataFrame.asfreq" title="pandas.DataFrame.asfreq"><code>DataFrame.asfreq</code></a>(freq[,&nbsp;method,&nbsp;how,&nbsp;…])</td>
    <td>Convert TimeSeries to specified frequency.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asof.html#pandas.DataFrame.asof" title="pandas.DataFrame.asof"><code>DataFrame.asof</code></a>(where[,&nbsp;subset])</td>
    <td>Return the last row(s) without any NaNs before <cite>where</cite>.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html#pandas.DataFrame.shift" title="pandas.DataFrame.shift"><code>DataFrame.shift</code></a>([periods,&nbsp;freq,&nbsp;axis,&nbsp;…])</td>
    <td>Shift index by desired number of periods with an optional time <cite>freq</cite>.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.slice_shift.html#pandas.DataFrame.slice_shift" title="pandas.DataFrame.slice_shift"><code>DataFrame.slice_shift</code></a>([periods,&nbsp;axis])</td>
    <td>Equivalent to <cite>shift</cite> without copying data.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tshift.html#pandas.DataFrame.tshift" title="pandas.DataFrame.tshift"><code>DataFrame.tshift</code></a>([periods,&nbsp;freq,&nbsp;axis])</td>
    <td>(DEPRECATED) Shift the time index, using the index’s frequency if available.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.first_valid_index.html#pandas.DataFrame.first_valid_index" title="pandas.DataFrame.first_valid_index"><code>DataFrame.first_valid_index</code></a>()</td>
    <td>Return index for first non-NA/null value.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.last_valid_index.html#pandas.DataFrame.last_valid_index" title="pandas.DataFrame.last_valid_index"><code>DataFrame.last_valid_index</code></a>()</td>
    <td>Return index for last non-NA/null value.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html#pandas.DataFrame.resample" title="pandas.DataFrame.resample"><code>DataFrame.resample</code></a>(rule[,&nbsp;axis,&nbsp;closed,&nbsp;…])</td>
    <td>Resample time-series data.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_period.html#pandas.DataFrame.to_period" title="pandas.DataFrame.to_period"><code>DataFrame.to_period</code></a>([freq,&nbsp;axis,&nbsp;copy])</td>
    <td>Convert DataFrame from DatetimeIndex to PeriodIndex.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_timestamp.html#pandas.DataFrame.to_timestamp" title="pandas.DataFrame.to_timestamp"><code>DataFrame.to_timestamp</code></a>([freq,&nbsp;how,&nbsp;axis,&nbsp;copy])</td>
    <td>Cast to DatetimeIndex of timestamps, at <em>beginning</em> of period.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tz_convert.html#pandas.DataFrame.tz_convert" title="pandas.DataFrame.tz_convert"><code>DataFrame.tz_convert</code></a>(tz[,&nbsp;axis,&nbsp;level,&nbsp;copy])</td>
    <td>Convert tz-aware axis to target time zone.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tz_localize.html#pandas.DataFrame.tz_localize" title="pandas.DataFrame.tz_localize"><code>DataFrame.tz_localize</code></a>(tz[,&nbsp;axis,&nbsp;level,&nbsp;…])</td>
    <td>Localize tz-naive index of a Series or DataFrame to target time zone.</td>
    </tr>
    </tbody>
    </table>


## DataFrame Plotting

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y4hhnfme">Plotting</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot" title="pandas.DataFrame.plot"><code>DataFrame.plot</code></a>([x,&nbsp;y,&nbsp;kind,&nbsp;ax,&nbsp;….])</td>
      <td>DataFrame plotting accessor and method</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.area.html#pandas.DataFrame.plot.area" title="pandas.DataFrame.plot.area"><code>DataFrame.plot.area</code></a>(self[,&nbsp;x,&nbsp;y])</td>
      <td>Draw a stacked area plot.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html#pandas.DataFrame.plot.bar" title="pandas.DataFrame.plot.bar"><code>DataFrame.plot.bar</code></a>(self[,&nbsp;x,&nbsp;y])</td>
      <td>Vertical bar plot.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.barh.html#pandas.DataFrame.plot.barh" title="pandas.DataFrame.plot.barh"><code>DataFrame.plot.barh</code></a>(self[,&nbsp;x,&nbsp;y])</td>
      <td>Make a horizontal bar plot.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.box.html#pandas.DataFrame.plot.box" title="pandas.DataFrame.plot.box"><code>DataFrame.plot.box</code></a>(self[,&nbsp;by])</td>
      <td>Make a box plot of the DataFrame columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.density.html#pandas.DataFrame.plot.density" title="pandas.DataFrame.plot.density"><code>DataFrame.plot.density</code></a>(self[,&nbsp;bw_method,&nbsp;ind])</td>
      <td>Generate Kernel Density Estimate plot using Gaussian kernels.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.hexbin.html#pandas.DataFrame.plot.hexbin" title="pandas.DataFrame.plot.hexbin"><code>DataFrame.plot.hexbin</code></a>(self,&nbsp;x,&nbsp;y[,&nbsp;C,&nbsp;…])</td>
      <td>Generate a hexagonal binning plot.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.hist.html#pandas.DataFrame.plot.hist" title="pandas.DataFrame.plot.hist"><code>DataFrame.plot.hist</code></a>(self[,&nbsp;by,&nbsp;bins])</td>
      <td>Draw one histogram of the DataFrame’s columns.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.kde.html#pandas.DataFrame.plot.kde" title="pandas.DataFrame.plot.kde"><code>DataFrame.plot.kde</code></a>(self[,&nbsp;bw_method,&nbsp;ind])</td>
      <td>Generate Kernel Density Estimate plot using Gaussian kernels.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.line.html#pandas.DataFrame.plot.line" title="pandas.DataFrame.plot.line"><code>DataFrame.plot.line</code></a>(self[,&nbsp;x,&nbsp;y])</td>
      <td>Plot Series or DataFrame as lines.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.pie.html#pandas.DataFrame.plot.pie" title="pandas.DataFrame.plot.pie"><code>DataFrame.plot.pie</code></a>(self,&nbsp;**kwargs)</td>
      <td>Generate a pie plot.</td>
    </tr>
    <tr>
      <td><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html#pandas.DataFrame.plot.scatter" title="pandas.DataFrame.plot.scatter"><code>DataFrame.plot.scatter</code></a>(self,&nbsp;x,&nbsp;y[,&nbsp;s,&nbsp;c])</td>
      <td>Create a scatter plot with varying marker point size and color.</td>
    </tr>
    </tbody>
  </table>



## TimeStamp Class

+ `pandas.Timestamp` class ([Ref](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html))
  + Syntax: `class pandas.Timestamp(ts_input=<object object>, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, *, fold=None)`
  + Docstring:
    + Pandas replacement for python `datetime.datetime` object.
    + Timestamp is the pandas equivalent of python’s Datetime and is interchangeable with it in most cases. It’s the type used for the entries that make up a `DatetimeIndex`, and other timeseries oriented data structures in pandas.
  + Parameters
    + `ts_input`: datetime-like, str, int, float<br/>
      Value to be converted to Timestamp.
    + `freq`: str, DateOffset<br/>
      Offset which Timestamp will have.
    + `tz`: str, pytz.timezone, dateutil.tz.tzfile or None<br/>
      Time zone for time which Timestamp will have.
    + `unit`: str<br/>
      Unit used for conversion if ts_input is of type int or float. The valid values are ‘D’, ‘h’, ‘m’, ‘s’, ‘ms’, ‘us’, and ‘ns’. For example, ‘s’ means seconds and ‘ms’ means milliseconds.
    + `year`, `month`, `day`: int
    + `hour`, `minute`, `second`, `microsecond`: int, optional, default 0
    + `nanosecond`: int, optional, default 0
    + `tzinfo`: datetime.tzinfo, optional, default None
    + `fold`: {0, 1}, default None, keyword-only <br/>
      Due to daylight saving time, one wall clock time can occur twice when shifting from summer to winter time; fold describes whether the datetime-like corresponds to the first (0) or the second time (1) the wall clock hits the ambiguous time


## TimeStamp Attributes

  <table ><table style="font-family: Arial,Helvetica,Sans-Serif; margin: 0 auto; width: 55vw;" cellspacing="0" cellpadding="5" border="1">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y4fjaptb">Attributes of <code>pandas.TimeStamp</code></a></caption>
  <thead>
  <tr style="font-size: 1.2em; vertical-align:middle"">
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Property</th>
    <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
  </tr>
  </thead>
  <tbody>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.asm8.html#pandas.Timestamp.asm8" title="pandas.Timestamp.asm8"><code>asm8</code></a></td>
  <td>Return numpy datetime64 format in nanoseconds.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.dayofweek.html#pandas.Timestamp.dayofweek" title="pandas.Timestamp.dayofweek"><code>dayofweek</code></a></td>
  <td>Return day of the week.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.dayofyear.html#pandas.Timestamp.dayofyear" title="pandas.Timestamp.dayofyear"><code>dayofyear</code></a></td>
  <td>Return the day of the year.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.days_in_month.html#pandas.Timestamp.days_in_month" title="pandas.Timestamp.days_in_month"><code>days_in_month</code></a></td>
  <td>Return the number of days in the month.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.daysinmonth.html#pandas.Timestamp.daysinmonth" title="pandas.Timestamp.daysinmonth"><code>daysinmonth</code></a></td>
  <td>Return the number of days in the month.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.freqstr.html#pandas.Timestamp.freqstr" title="pandas.Timestamp.freqstr"><code>freqstr</code></a></td>
  <td>Return the total number of days in the month.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.is_leap_year.html#pandas.Timestamp.is_leap_year" title="pandas.Timestamp.is_leap_year"><code>is_leap_year</code></a></td>
  <td>Return True if year is a leap year.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.is_month_end.html#pandas.Timestamp.is_month_end" title="pandas.Timestamp.is_month_end"><code>is_month_end</code></a></td>
  <td>Return True if date is last day of month.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.is_month_start.html#pandas.Timestamp.is_month_start" title="pandas.Timestamp.is_month_start"><code>is_month_start</code></a></td>
  <td>Return True if date is first day of month.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.is_quarter_end.html#pandas.Timestamp.is_quarter_end" title="pandas.Timestamp.is_quarter_end"><code>is_quarter_end</code></a></td>
  <td>Return True if date is last day of the quarter.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.is_quarter_start.html#pandas.Timestamp.is_quarter_start" title="pandas.Timestamp.is_quarter_start"><code>is_quarter_start</code></a></td>
  <td>Return True if date is first day of the quarter.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.is_year_end.html#pandas.Timestamp.is_year_end" title="pandas.Timestamp.is_year_end"><code>is_year_end</code></a></td>
  <td>Return True if date is last day of the year.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.is_year_start.html#pandas.Timestamp.is_year_start" title="pandas.Timestamp.is_year_start"><code>is_year_start</code></a></td>
  <td>Return True if date is first day of the year.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.quarter.html#pandas.Timestamp.quarter" title="pandas.Timestamp.quarter"><code>quarter</code></a></td>
  <td>Return the quarter of the year.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.tz.html#pandas.Timestamp.tz" title="pandas.Timestamp.tz"><code>tz</code></a></td>
  <td>Alias for tzinfo.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.week.html#pandas.Timestamp.week" title="pandas.Timestamp.week"><code>week</code></a></td>
  <td>Return the week number of the year.</td>
  </tr>
  <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.weekofyear.html#pandas.Timestamp.weekofyear" title="pandas.Timestamp.weekofyear"><code>weekofyear</code></a></td>
  <td>Return the week number of the year.</td>
  </tr>
  <tr><td colspan=2><code>year</code>, <code>month</code>, <code>day</code>, <code>hour</code>, <code>minute</code>, <code>second</code>, <code>microsecond</code>, <code>nanosecond</code>, <code>fold</code>, <code>freq</code>, <code>tzinfo</code>, <code>value</code></td></tr>
  </tbody>
  </table>


## TimeStamp Methods

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y4fjaptb">TimeStamp Methods</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.astimezone.html#pandas.Timestamp.astimezone" title="pandas.Timestamp.astimezone"><code>astimezone</code></a>(tz)</td>
    <td>Convert tz-aware Timestamp to another time zone.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.ceil.html#pandas.Timestamp.ceil" title="pandas.Timestamp.ceil"><code>ceil</code></a>(freq[,&nbsp;ambiguous,&nbsp;nonexistent])</td>
    <td>return a new Timestamp ceiled to this resolution.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.combine.html#pandas.Timestamp.combine" title="pandas.Timestamp.combine"><code>combine</code></a>(date,&nbsp;time)</td>
    <td>date, time -&gt; datetime with same date and time fields.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.ctime.html#pandas.Timestamp.ctime" title="pandas.Timestamp.ctime"><code>ctime</code></a></td>
    <td>Return ctime() style string.</td>
    </tr>
    <tr ><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.date.html#pandas.Timestamp.date" title="pandas.Timestamp.date"><code>date</code></a></td>
    <td>Return date object with same year, month and day.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.day_name.html#pandas.Timestamp.day_name" title="pandas.Timestamp.day_name"><code>day_name</code></a></td>
    <td>Return the day name of the Timestamp with specified locale.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.dst.html#pandas.Timestamp.dst" title="pandas.Timestamp.dst"><code>dst</code></a></td>
    <td>Return self.tzinfo.dst(self).</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.floor.html#pandas.Timestamp.floor" title="pandas.Timestamp.floor"><code>floor</code></a>(freq[,&nbsp;ambiguous,&nbsp;nonexistent])</td>
    <td>return a new Timestamp floored to this resolution.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.fromisocalendar.html#pandas.Timestamp.fromisocalendar" title="pandas.Timestamp.fromisocalendar"><code>fromisocalendar</code></a></td>
    <td>int, int, int -&gt; Construct a date from the ISO year, week number and weekday.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.fromisoformat.html#pandas.Timestamp.fromisoformat" title="pandas.Timestamp.fromisoformat"><code>fromisoformat</code></a></td>
    <td>string -&gt; datetime from datetime.isoformat() output</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.fromordinal.html#pandas.Timestamp.fromordinal" title="pandas.Timestamp.fromordinal"><code>fromordinal</code></a>(ordinal[,&nbsp;freq,&nbsp;tz])</td>
    <td>Passed an ordinal, translate and convert to a ts.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.fromtimestamp.html#pandas.Timestamp.fromtimestamp" title="pandas.Timestamp.fromtimestamp"><code>fromtimestamp</code></a>(ts)</td>
    <td>timestamp[, tz] -&gt; tz’s local time from POSIX timestamp.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.isocalendar.html#pandas.Timestamp.isocalendar" title="pandas.Timestamp.isocalendar"><code>isocalendar</code></a></td>
    <td>Return a 3-tuple containing ISO year, week number, and weekday.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.isoformat.html#pandas.Timestamp.isoformat" title="pandas.Timestamp.isoformat"><code>isoformat</code></a></td>
    <td>[sep] -&gt; string in ISO 8601 format, YYYY-MM-DDT[HH[:MM[:SS[.mmm[uuu]]]]][+HH:MM].</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.isoweekday.html#pandas.Timestamp.isoweekday" title="pandas.Timestamp.isoweekday"><code>isoweekday</code></a></td>
    <td>Return the day of the week represented by the date.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.month_name.html#pandas.Timestamp.month_name" title="pandas.Timestamp.month_name"><code>month_name</code></a></td>
    <td>Return the month name of the Timestamp with specified locale.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.normalize.html#pandas.Timestamp.normalize" title="pandas.Timestamp.normalize"><code>normalize</code></a></td>
    <td>Normalize Timestamp to midnight, preserving tz information.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.now.html#pandas.Timestamp.now" title="pandas.Timestamp.now"><code>now</code></a>([tz])</td>
    <td>Return new Timestamp object representing current time local to tz.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.replace.html#pandas.Timestamp.replace" title="pandas.Timestamp.replace"><code>replace</code></a>([year,&nbsp;month,&nbsp;day,&nbsp;hour,&nbsp;minute,&nbsp;…])</td>
    <td>implements datetime.replace, handles nanoseconds.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.round.html#pandas.Timestamp.round" title="pandas.Timestamp.round"><code>round</code></a>(freq[,&nbsp;ambiguous,&nbsp;nonexistent])</td>
    <td>Round the Timestamp to the specified resolution.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.strftime.html#pandas.Timestamp.strftime" title="pandas.Timestamp.strftime"><code>strftime</code></a></td>
    <td>format -&gt; strftime() style string.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.strptime.html#pandas.Timestamp.strptime" title="pandas.Timestamp.strptime"><code>strptime</code></a>(string,&nbsp;format)</td>
    <td>Function is not implemented.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.time.html#pandas.Timestamp.time" title="pandas.Timestamp.time"><code>time</code></a></td>
    <td>Return time object with same time but with tzinfo=None.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.timestamp.html#pandas.Timestamp.timestamp" title="pandas.Timestamp.timestamp"><code>timestamp</code></a></td>
    <td>Return POSIX timestamp as float.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.timetuple.html#pandas.Timestamp.timetuple" title="pandas.Timestamp.timetuple"><code>timetuple</code></a></td>
    <td>Return time tuple, compatible with time.localtime().</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.timetz.html#pandas.Timestamp.timetz" title="pandas.Timestamp.timetz"><code>timetz</code></a></td>
    <td>Return time object with same time and tzinfo.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.to_datetime64.html#pandas.Timestamp.to_datetime64" title="pandas.Timestamp.to_datetime64"><code>to_datetime64</code></a></td>
    <td>Return a numpy.datetime64 object with ‘ns’ precision.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.to_julian_date.html#pandas.Timestamp.to_julian_date" title="pandas.Timestamp.to_julian_date"><code>to_julian_date</code></a>()</td>
    <td>Convert TimeStamp to a Julian Date.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.to_numpy.html#pandas.Timestamp.to_numpy" title="pandas.Timestamp.to_numpy"><code>to_numpy</code></a></td>
    <td>Convert the Timestamp to a NumPy datetime64.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.to_period.html#pandas.Timestamp.to_period" title="pandas.Timestamp.to_period"><code>to_period</code></a></td>
    <td>Return an period of which this timestamp is an observation.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.to_pydatetime.html#pandas.Timestamp.to_pydatetime" title="pandas.Timestamp.to_pydatetime"><code>to_pydatetime</code></a></td>
    <td>Convert a Timestamp object to a native Python datetime object.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.today.html#pandas.Timestamp.today" title="pandas.Timestamp.today"><code>today</code></a>(cls[,&nbsp;tz])</td>
    <td>Return the current time in the local timezone.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.toordinal.html#pandas.Timestamp.toordinal" title="pandas.Timestamp.toordinal"><code>toordinal</code></a></td>
    <td>Return proleptic Gregorian ordinal.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.tz_convert.html#pandas.Timestamp.tz_convert" title="pandas.Timestamp.tz_convert"><code>tz_convert</code></a>(tz)</td>
    <td>Convert tz-aware Timestamp to another time zone.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.tz_localize.html#pandas.Timestamp.tz_localize" title="pandas.Timestamp.tz_localize"><code>tz_localize</code></a>(tz[,&nbsp;ambiguous,&nbsp;nonexistent])</td>
    <td>Convert naive Timestamp to local time zone, or remove timezone from tz-aware Timestamp.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.tzname.html#pandas.Timestamp.tzname" title="pandas.Timestamp.tzname"><code>tzname</code></a></td>
    <td>Return self.tzinfo.tzname(self).</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.utcfromtimestamp.html#pandas.Timestamp.utcfromtimestamp" title="pandas.Timestamp.utcfromtimestamp"><code>utcfromtimestamp</code></a>(ts)</td>
    <td>Construct a naive UTC datetime from a POSIX timestamp.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.utcnow.html#pandas.Timestamp.utcnow" title="pandas.Timestamp.utcnow"><code>utcnow</code></a>()</td>
    <td>Return a new Timestamp representing UTC day and time.</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.utcoffset.html#pandas.Timestamp.utcoffset" title="pandas.Timestamp.utcoffset"><code>utcoffset</code></a></td>
    <td>Return self.tzinfo.utcoffset(self).</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.utctimetuple.html#pandas.Timestamp.utctimetuple" title="pandas.Timestamp.utctimetuple"><code>utctimetuple</code></a></td>
    <td>Return UTC time tuple, compatible with time.localtime().</td>
    </tr>
    <tr ><td><a  href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.weekday.html#pandas.Timestamp.weekday" title="pandas.Timestamp.weekday"><code>weekday</code></a></td>
    <td>Return the day of the week represented by the date.</td>
    </tr>
    </tbody>
    </table>


## Timedelta Class

+ `pandas.Timedelta` class
  + Syntax: `class pandas.Timedelta(value=<object object>, unit=None, **kwargs)`
  + Docstring:
    + Represents a duration, the difference between two dates or times.
    + Timedelta is the pandas equivalent of python’s datetime.timedelta and is interchangeable with it in most cases.
  + Parameters
    + `value`: Timedelta, timedelta, np.timedelta64, str, or int
    + `unit`: str, default ‘ns’
      + Denote the unit of the input, if input is an integer.
      + Possible values:
        + ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
        + ‘days’ or ‘day’
        + ‘hours’, ‘hour’, ‘hr’, or ‘h’
        + ‘minutes’, ‘minute’, ‘min’, or ‘m’
        + ‘seconds’, ‘second’, or ‘sec’
        + ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
        + ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
        + ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.
    + `**kwargs`<br/>
      Available kwargs: {days, seconds, microseconds, milliseconds, minutes, hours, weeks}. Values for construction in compat with datetime.timedelta. Numpy ints and floats will be coerced to python ints and floats.


## Timedelta Attributes

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Timedelta Attributes</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.asm8.html#pandas.Timedelta.asm8" title="pandas.Timedelta.asm8"><code>asm8</code></a></td>
    <td>Return a numpy timedelta64 array scalar view.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.components.html#pandas.Timedelta.components" title="pandas.Timedelta.components"><code>components</code></a></td>
    <td>Return a components namedtuple-like.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.days.html#pandas.Timedelta.days" title="pandas.Timedelta.days"><code>days</code></a></td>
    <td>Number of days.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.delta.html#pandas.Timedelta.delta" title="pandas.Timedelta.delta"><code>delta</code></a></td>
    <td>Return the timedelta in nanoseconds (ns), for internal compatibility.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.microseconds.html#pandas.Timedelta.microseconds" title="pandas.Timedelta.microseconds"><code>microseconds</code></a></td>
    <td>Number of microseconds (&gt;= 0 and less than 1 second).</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.nanoseconds.html#pandas.Timedelta.nanoseconds" title="pandas.Timedelta.nanoseconds"><code>nanoseconds</code></a></td>
    <td>Return the number of nanoseconds (n), where 0 &lt;= n &lt; 1 microsecond.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.resolution_string.html#pandas.Timedelta.resolution_string" title="pandas.Timedelta.resolution_string"><code>resolution_string</code></a></td>
    <td>Return a string representing the lowest timedelta resolution.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.seconds.html#pandas.Timedelta.seconds" title="pandas.Timedelta.seconds"><code>seconds</code></a></td>
    <td>Number of seconds (&gt;= 0 and less than 1 day).</td>
    </tr>
    <tr><td colspan=2><code>freq</code>, <code>is_populated</code>, <code>value</code></td></tr>
    </tbody>
  </table>


## Timedelta Methods

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Template</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.ceil.html#pandas.Timedelta.ceil" title="pandas.Timedelta.ceil"><code>ceil</code></a>(freq)</td>
    <td>Return a new Timedelta ceiled to this resolution.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.floor.html#pandas.Timedelta.floor" title="pandas.Timedelta.floor"><code>floor</code></a>(freq)</td>
    <td>Return a new Timedelta floored to this resolution.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.isoformat.html#pandas.Timedelta.isoformat" title="pandas.Timedelta.isoformat"><code>isoformat</code></a></td>
    <td>Format Timedelta as ISO 8601 Duration like <code>P[n]Y[n]M[n]DT[n]H[n]M[n]S</code>, where the <code>[n]</code> s are replaced by the values.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.round.html#pandas.Timedelta.round" title="pandas.Timedelta.round"><code>round</code></a>(freq)</td>
    <td>Round the Timedelta to the specified resolution.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.to_numpy.html#pandas.Timedelta.to_numpy" title="pandas.Timedelta.to_numpy"><code>to_numpy</code></a></td>
    <td>Convert the Timedelta to a NumPy timedelta64.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.to_pytimedelta.html#pandas.Timedelta.to_pytimedelta" title="pandas.Timedelta.to_pytimedelta"><code>to_pytimedelta</code></a></td>
    <td>Convert a pandas Timedelta object into a python timedelta object.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.to_timedelta64.html#pandas.Timedelta.to_timedelta64" title="pandas.Timedelta.to_timedelta64"><code>to_timedelta64</code></a></td>
    <td>Return a numpy.timedelta64 object with ‘ns’ precision.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.total_seconds.html#pandas.Timedelta.total_seconds" title="pandas.Timedelta.total_seconds"><code>total_seconds</code></a></td>
    <td>Total seconds in the duration.</td>
    </tr>
    <tr><td><a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.view.html#pandas.Timedelta.view" title="pandas.Timedelta.view"><code>view</code></a></td>
    <td>Array view compatibility.</td>
    </tr>
    </tbody>
  </table>




##  Template

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Template</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
  </table>
