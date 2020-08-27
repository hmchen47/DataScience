# Pandas - Dataframe

## Pandas DataFrame

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


  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yxqwpewf">Binary Opperator Functions</a></caption>
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
      <td>Call <code class="docutils literal notranslate"><span class="pre">func</code> on self producing a DataFrame with transformed values.</td>
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


  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Reshaping, sorting, transposing</a></caption>
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

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="url">Template</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
  </table>

