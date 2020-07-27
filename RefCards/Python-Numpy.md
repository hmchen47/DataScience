# Numpy - Array, Vector, Matrix, Linear Algebra, and Other Useful Functions


## Array Basics

+ `ndarray` attributes
  + `T`: ndarray
      Transpose of the array.
  + `data`: buffer<br/>
      The array's elements, in memory.
  + `dtype`: dtype object<br/>
      Describes the format of the elements in the array.
  + `flags`: dict<br/>
      Dictionary containing information related to memory use, e.g., 'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
  + `flat`: numpy.flatiter object<br/>
      Flattened version of the array as an iterator.  The iterator allows assignments, e.g., `x.flat = 3` (See `ndarray.flat` for assignment examples; TODO).
  + `imag`: ndarray<br/>
      Imaginary part of the array.
  + `real`: ndarray<br/>
      Real part of the array.
  + `size`: int<br/>
      Number of elements in the array.
  + `itemsize`: int<br/>
      The memory use of each array element in bytes.
  + `nbytes`: int<br/>
      The total number of bytes required to store the array data,
      i.e., `itemsize * size`.
  + `ndim`: int<br/>
      The array's number of dimensions.
  + `shape`: tuple of ints<br/>
      Shape of the array.
  + `strides`: tuple of ints<br/>
    + The step-size required to move from one element to the next in memory. For example, a contiguous `(3, 4)` array of type
      `int16` in C-order has strides `(8, 2)`.  This implies that to move from element to element in memory requires jumps of 2 bytes.
    + To move from row-to-row, one needs to jump 8 bytes at a time
      (`2 * 4`).
  + `ctypes`: ctypes object<br/>
      Class containing properties of the array needed for interaction
      with ctypes.
  + `base`: ndarray<br/>
      If the array is a view into another array, that array is its `base`
      (unless that array is also a view).  The `base` array is where the
      array data is actually stored.


## Array Creating routines

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y2ecowxs">Template</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.empty.html#numpy.empty" title="numpy.empty"><code>empty</code></a>(shape[,&nbsp;dtype,&nbsp;order])</td>
      <td>Return a new array of given shape and type, without initializing entries.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html#numpy.empty_like" title="numpy.empty_like"><code>empty_like</code></a>(prototype[,&nbsp;dtype,&nbsp;order,&nbsp;subok,&nbsp;…])</td>
      <td>Return a new array with the same shape and type as a given array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.eye.html#numpy.eye" title="numpy.eye"><code>eye</code></a>(N[,&nbsp;M,&nbsp;k,&nbsp;dtype,&nbsp;order])</td>
      <td>Return a 2-D array with ones on the diagonal and zeros elsewhere.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.identity.html#numpy.identity" title="numpy.identity"><code>identity</code></a>(n[,&nbsp;dtype])</td>
      <td>Return the identity array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ones.html#numpy.ones" title="numpy.ones"><code>ones</code></a>(shape[,&nbsp;dtype,&nbsp;order])</td>
      <td>Return a new array of given shape and type, filled with ones.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html#numpy.ones_like" title="numpy.ones_like"><code>ones_like</code></a>(a[,&nbsp;dtype,&nbsp;order,&nbsp;subok,&nbsp;shape])</td>
      <td>Return an array of ones with the same shape and type as a given array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.zeros.html#numpy.zeros" title="numpy.zeros"><code>zeros</code></a>(shape[,&nbsp;dtype,&nbsp;order])</td>
      <td>Return a new array of given shape and type, filled with zeros.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html#numpy.zeros_like" title="numpy.zeros_like"><code>zeros_like</code></a>(a[,&nbsp;dtype,&nbsp;order,&nbsp;subok,&nbsp;shape])</td>
      <td>Return an array of zeros with the same shape and type as a given array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.full.html#numpy.full" title="numpy.full"><code>full</code></a>(shape,&nbsp;fill_value[,&nbsp;dtype,&nbsp;order])</td>
      <td>Return a new array of given shape and type, filled with <em class="xref py py-obj">fill_value</em>.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy.full_like" title="numpy.full_like"><code>full_like</code></a>(a,&nbsp;fill_value[,&nbsp;dtype,&nbsp;order,&nbsp;…])</td>
      <td>Return a full array with the same shape and type as a given array.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y2c2nlc8">From existing data</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array" title="numpy.array"><code>array</code></a>(object[,&nbsp;dtype,&nbsp;copy,&nbsp;order,&nbsp;subok,&nbsp;ndmin])</td>
      <td>Create an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray" title="numpy.asarray"><code>asarray</code></a>(a[,&nbsp;dtype,&nbsp;order])</td>
      <td>Convert the input to an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html#numpy.asanyarray" title="numpy.asanyarray"><code>asanyarray</code></a>(a[,&nbsp;dtype,&nbsp;order])</td>
      <td>Convert the input to an ndarray, but pass ndarray subclasses through.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray" title="numpy.ascontiguousarray"><code>ascontiguousarray</code></a>(a[,&nbsp;dtype])</td>
      <td>Return a contiguous array (ndim &gt;= 1) in memory (C order).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html#numpy.asmatrix" title="numpy.asmatrix"><code>asmatrix</code></a>(data[,&nbsp;dtype])</td>
      <td>Interpret the input as a matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.copy.html#numpy.copy" title="numpy.copy"><code>copy</code></a>(a[,&nbsp;order,&nbsp;subok])</td>
      <td>Return an array copy of the given object.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html#numpy.frombuffer" title="numpy.frombuffer"><code>frombuffer</code></a>(buffer[,&nbsp;dtype,&nbsp;count,&nbsp;offset])</td>
      <td>Interpret a buffer as a 1-dimensional array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html#numpy.fromfile" title="numpy.fromfile"><code>fromfile</code></a>(file[,&nbsp;dtype,&nbsp;count,&nbsp;sep,&nbsp;offset])</td>
      <td>Construct an array from data in a text or binary file.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html#numpy.fromfunction" title="numpy.fromfunction"><code>fromfunction</code></a>(function,&nbsp;shape,&nbsp;\*[,&nbsp;dtype])</td>
      <td>Construct an array by executing a function over each coordinate.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.fromiter.html#numpy.fromiter" title="numpy.fromiter"><code>fromiter</code></a>(iterable,&nbsp;dtype[,&nbsp;count])</td>
      <td>Create a new 1-dimensional array from an iterable object.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html#numpy.fromstring" title="numpy.fromstring"><code>fromstring</code></a>(string[,&nbsp;dtype,&nbsp;count,&nbsp;sep])</td>
      <td>A new 1-D array initialized from text data in a string.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html#numpy.loadtxt" title="numpy.loadtxt"><code>loadtxt</code></a>(fname[,&nbsp;dtype,&nbsp;comments,&nbsp;delimiter,&nbsp;…])</td>
      <td>Load data from a text file.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yy6fjtnj">Creating record arrays (numpy.rec)</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.core.records.array.html#numpy.core.records.array" title="numpy.core.records.array"><code>core.records.array</code></a>(obj[,&nbsp;dtype,&nbsp;shape,&nbsp;…])</td>
      <td>Construct a record array from a wide-variety of objects.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.core.records.fromarrays.html#numpy.core.records.fromarrays" title="numpy.core.records.fromarrays"><code>core.records.fromarrays</code></a>(arrayList[,&nbsp;dtype,&nbsp;…])</td>
      <td>Create a record array from a (flat) list of arrays</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.core.records.fromrecords.html#numpy.core.records.fromrecords" title="numpy.core.records.fromrecords"><code>core.records.fromrecords</code></a>(recList[,&nbsp;dtype,&nbsp;…])</td>
      <td>Create a recarray from a list of records in text form.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.core.records.fromstring.html#numpy.core.records.fromstring" title="numpy.core.records.fromstring"><code>core.records.fromstring</code></a>(datastring[,&nbsp;dtype,&nbsp;…])</td>
      <td>Create a record array from binary data</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.core.records.fromfile.html#numpy.core.records.fromfile" title="numpy.core.records.fromfile"><code>core.records.fromfile</code></a>(fd[,&nbsp;dtype,&nbsp;shape,&nbsp;…])</td>
      <td>Create an array from binary file data</td>
    </tr>
    </tbody>
      </table>

      <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
        <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y6lrgpf2">Creating character arrays (numpy.char)</a></caption>
        <thead>
        <tr style="font-size: 1.2em;">
          <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
          <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
        </tr>
        </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array" title="numpy.core.defchararray.array"><code>core.defchararray.array</code></a>(obj[,&nbsp;itemsize,&nbsp;…])</td>
      <td>Create a <a href="https://numpy.org/doc/stable/reference/generated/numpy.chararray.html#numpy.chararray" title="numpy.chararray"><code>chararray</code></a>.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.core.defchararray.asarray.html#numpy.core.defchararray.asarray" title="numpy.core.defchararray.asarray"><code>core.defchararray.asarray</code></a>(obj[,&nbsp;itemsize,&nbsp;…])</td>
      <td>Convert the input to a <a href="https://numpy.org/doc/stable/reference/generated/numpy.chararray.html#numpy.chararray" title="numpy.chararray"><code>chararray</code></a>, copying the data only if necessary.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y5gztb7q">Numerical ranges</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange" title="numpy.arange"><code>arange</code></a>([start,]&nbsp;stop[,&nbsp;step,][,&nbsp;dtype])</td>
      <td>Return evenly spaced values within a given interval.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace" title="numpy.linspace"><code>linspace</code></a>(start,&nbsp;stop[,&nbsp;num,&nbsp;endpoint,&nbsp;…])</td>
      <td>Return evenly spaced numbers over a specified interval.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.logspace.html#numpy.logspace" title="numpy.logspace"><code>logspace</code></a>(start,&nbsp;stop[,&nbsp;num,&nbsp;endpoint,&nbsp;base,&nbsp;…])</td>
      <td>Return numbers spaced evenly on a log scale.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html#numpy.geomspace" title="numpy.geomspace"><code>geomspace</code></a>(start,&nbsp;stop[,&nbsp;num,&nbsp;endpoint,&nbsp;…])</td>
      <td>Return numbers spaced evenly on a log scale (a geometric progression).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid" title="numpy.meshgrid"><code>meshgrid</code></a>(\*xi[,&nbsp;copy,&nbsp;sparse,&nbsp;indexing])</td>
      <td>Return coordinate matrices from coordinate vectors.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html#numpy.mgrid" title="numpy.mgrid"><code>mgrid</code></a></td>
      <td><em class="xref py py-obj">nd_grid</em> instance which returns a dense multi-dimensional “meshgrid”.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html#numpy.ogrid" title="numpy.ogrid"><code>ogrid</code></a></td>
      <td><em class="xref py py-obj">nd_grid</em> instance which returns an open multi-dimensional “meshgrid”.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yx9gl3p6">Building matrices</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.diag.html#numpy.diag" title="numpy.diag"><code>diag</code></a>(v[,&nbsp;k])</td>
      <td>Extract a diagonal or construct a diagonal array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.diagflat.html#numpy.diagflat" title="numpy.diagflat"><code>diagflat</code></a>(v[,&nbsp;k])</td>
      <td>Create a two-dimensional array with the flattened input as a diagonal.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.tri.html#numpy.tri" title="numpy.tri"><code>tri</code></a>(N[,&nbsp;M,&nbsp;k,&nbsp;dtype])</td>
      <td>An array with ones at and below the given diagonal and zeros elsewhere.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.tril.html#numpy.tril" title="numpy.tril"><code>tril</code></a>(m[,&nbsp;k])</td>
      <td>Lower triangle of an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.triu.html#numpy.triu" title="numpy.triu"><code>triu</code></a>(m[,&nbsp;k])</td>
      <td>Upper triangle of an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.vander.html#numpy.vander" title="numpy.vander"><code>vander</code></a>(x[,&nbsp;N,&nbsp;increasing])</td>
      <td>Generate a Vandermonde matrix.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y3b3p5yw">The Matrix class</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.mat.html#numpy.mat" title="numpy.mat"><code>mat</code></a>(data[,&nbsp;dtype])</td>
      <td>Interpret the input as a matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.bmat.html#numpy.bmat" title="numpy.bmat"><code>bmat</code></a>(obj[,&nbsp;ldict,&nbsp;gdict])</td>
      <td>Build a matrix object from a string, nested sequence, or array.</td>
    </tr>
    </tbody>
  </table>



## Array manipulation routines

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y2fcsphd">Basic operations</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.copyto.html#numpy.copyto" title="numpy.copyto"><code>copyto</code></a>(dst,&nbsp;src[,&nbsp;casting,&nbsp;where])</td>
      <td>Copies values from one array to another, broadcasting as necessary.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.shape.html#numpy.shape" title="numpy.shape"><code>shape</code></a>(a)</td>
      <td>Return the shape of an array.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y3b84lwy">Changing array shape</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape" title="numpy.reshape"><code>reshape</code></a>(a,&nbsp;newshape[,&nbsp;order])</td>
      <td>Gives a new shape to an array without changing its data.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ravel.html#numpy.ravel" title="numpy.ravel"><code>ravel</code></a>(a[,&nbsp;order])</td>
      <td>Return a contiguous flattened array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat" title="numpy.ndarray.flat"><code>ndarray.flat</code></a></td>
      <td>A 1-D iterator over the array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten" title="numpy.ndarray.flatten"><code>ndarray.flatten</code></a>([order])</td>
      <td>Return a copy of the array collapsed into one dimension.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y4dzggdw">Transpose-like operations</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html#numpy.moveaxis" title="numpy.moveaxis"><code>moveaxis</code></a>(a,&nbsp;source,&nbsp;destination)</td>
      <td>Move axes of an array to new positions.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html#numpy.rollaxis" title="numpy.rollaxis"><code>rollaxis</code></a>(a,&nbsp;axis[,&nbsp;start])</td>
      <td>Roll the specified axis backwards, until it lies in a given position.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html#numpy.swapaxes" title="numpy.swapaxes"><code>swapaxes</code></a>(a,&nbsp;axis1,&nbsp;axis2)</td>
      <td>Interchange two axes of an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T" title="numpy.ndarray.T"><code>ndarray.T</code></a></td>
      <td>The transposed array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose" title="numpy.transpose"><code>transpose</code></a>(a[,&nbsp;axes])</td>
      <td>Reverse or permute the axes of an array; returns the modified array.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yyh28g4w">Changing kind of array</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray" title="numpy.asarray"><code>asarray</code></a>(a[,&nbsp;dtype,&nbsp;order])</td>
      <td>Convert the input to an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html#numpy.asanyarray" title="numpy.asanyarray"><code>asanyarray</code></a>(a[,&nbsp;dtype,&nbsp;order])</td>
      <td>Convert the input to an ndarray, but pass ndarray subclasses through.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html#numpy.asmatrix" title="numpy.asmatrix"><code>asmatrix</code></a>(data[,&nbsp;dtype])</td>
      <td>Interpret the input as a matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asfarray.html#numpy.asfarray" title="numpy.asfarray"><code>asfarray</code></a>(a[,&nbsp;dtype])</td>
      <td>Return an array converted to a float type.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray" title="numpy.asfortranarray"><code>asfortranarray</code></a>(a[,&nbsp;dtype])</td>
      <td>Return an array (ndim &gt;= 1) laid out in Fortran order in memory.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray" title="numpy.ascontiguousarray"><code>ascontiguousarray</code></a>(a[,&nbsp;dtype])</td>
      <td>Return a contiguous array (ndim &gt;= 1) in memory (C order).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asarray_chkfinite.html#numpy.asarray_chkfinite" title="numpy.asarray_chkfinite"><code>asarray_chkfinite</code></a>(a[,&nbsp;dtype,&nbsp;order])</td>
      <td>Convert the input to an array, checking for NaNs or Infs.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.asscalar.html#numpy.asscalar" title="numpy.asscalar"><code>asscalar</code></a>(a)</td>
      <td>Convert an array of size 1 to its scalar equivalent.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.require.html#numpy.require" title="numpy.require"><code>require</code></a>(a[,&nbsp;dtype,&nbsp;requirements])</td>
      <td>Return an ndarray of the provided type that satisfies requirements.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y3rgd7gu">Joining arrays</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html#numpy.concatenate" title="numpy.concatenate"><code>concatenate</code></a>([axis,&nbsp;out])</td>
      <td>Join a sequence of arrays along an existing axis.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.stack.html#numpy.stack" title="numpy.stack"><code>stack</code></a>(arrays[,&nbsp;axis,&nbsp;out])</td>
      <td>Join a sequence of arrays along a new axis.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.block.html#numpy.block" title="numpy.block"><code>block</code></a>(arrays)</td>
      <td>Assemble an nd-array from nested lists of blocks.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack" title="numpy.vstack"><code>vstack</code></a>(tup)</td>
      <td>Stack arrays in sequence vertically (row wise).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy.hstack" title="numpy.hstack"><code>hstack</code></a>(tup)</td>
      <td>Stack arrays in sequence horizontally (column wise).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.dstack.html#numpy.dstack" title="numpy.dstack"><code>dstack</code></a>(tup)</td>
      <td>Stack arrays in sequence depth wise (along third axis).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html#numpy.column_stack" title="numpy.column_stack"><code>column_stack</code></a>(tup)</td>
      <td>Stack 1-D arrays as columns into a 2-D array.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yytgvopz">Splitting arrays</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.split.html#numpy.split" title="numpy.split"><code>split</code></a>(ary,&nbsp;indices_or_sections[,&nbsp;axis])</td>
      <td>Split an array into multiple sub-arrays as views into <em class="xref py py-obj">ary</em>.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split" title="numpy.array_split"><code>array_split</code></a>(ary,&nbsp;indices_or_sections[,&nbsp;axis])</td>
      <td>Split an array into multiple sub-arrays.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html#numpy.dsplit" title="numpy.dsplit"><code>dsplit</code></a>(ary,&nbsp;indices_or_sections)</td>
      <td>Split array into multiple sub-arrays along the 3rd axis (depth).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html#numpy.hsplit" title="numpy.hsplit"><code>hsplit</code></a>(ary,&nbsp;indices_or_sections)</td>
      <td>Split an array into multiple sub-arrays horizontally (column-wise).</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html#numpy.vsplit" title="numpy.vsplit"><code>vsplit</code></a>(ary,&nbsp;indices_or_sections)</td>
      <td>Split an array into multiple sub-arrays vertically (row-wise).</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y42xs9ne">Adding and removing elements</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.delete.html#numpy.delete" title="numpy.delete"><code>delete</code></a>(arr,&nbsp;obj[,&nbsp;axis])</td>
      <td>Return a new array with sub-arrays along an axis deleted.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.insert.html#numpy.insert" title="numpy.insert"><code>insert</code></a>(arr,&nbsp;obj,&nbsp;values[,&nbsp;axis])</td>
      <td>Insert values along the given axis before the given indices.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.append.html#numpy.append" title="numpy.append"><code>append</code></a>(arr,&nbsp;values[,&nbsp;axis])</td>
      <td>Append values to the end of an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.resize.html#numpy.resize" title="numpy.resize"><code>resize</code></a>(a,&nbsp;new_shape)</td>
      <td>Return a new array with the specified shape.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html#numpy.trim_zeros" title="numpy.trim_zeros"><code>trim_zeros</code></a>(filt[,&nbsp;trim])</td>
      <td>Trim the leading and/or trailing zeros from a 1-D array or sequence.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.unique.html#numpy.unique" title="numpy.unique"><code>unique</code></a>(ar[,&nbsp;return_index,&nbsp;return_inverse,&nbsp;…])</td>
      <td>Find the unique elements of an array.</td>
    </tr>
    </tbody>
      </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y5zlc8uz">Rearranging elements</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.flip.html#numpy.flip" title="numpy.flip"><code>flip</code></a>(m[,&nbsp;axis])</td>
      <td>Reverse the order of elements in an array along the given axis.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html#numpy.fliplr" title="numpy.fliplr"><code>fliplr</code></a>(m)</td>
      <td>Flip array in the left/right direction.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.flipud.html#numpy.flipud" title="numpy.flipud"><code>flipud</code></a>(m)</td>
      <td>Flip array in the up/down direction.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape" title="numpy.reshape"><code>reshape</code></a>(a,&nbsp;newshape[,&nbsp;order])</td>
      <td>Gives a new shape to an array without changing its data.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.roll.html#numpy.roll" title="numpy.roll"><code>roll</code></a>(a,&nbsp;shift[,&nbsp;axis])</td>
      <td>Roll array elements along a given axis.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.rot90.html#numpy.rot90" title="numpy.rot90"><code>rot90</code></a>(m[,&nbsp;k,&nbsp;axes])</td>
      <td>Rotate an array by 90 degrees in the plane specified by axes.</td>
    </tr>
    </tbody>
  </table>


## Linear Algebra

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y52df8mf">Matrix and vector products</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot" title="numpy.dot"><code>dot</code></a>(a,&nbsp;b[,&nbsp;out])</td>
      <td>Dot product of two arrays.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot" title="numpy.linalg.multi_dot"><code>linalg.multi_dot</code></a>(arrays,&nbsp;\*[,&nbsp;out])</td>
      <td>Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.vdot.html#numpy.vdot" title="numpy.vdot"><code>vdot</code></a>(a,&nbsp;b)</td>
      <td>Return the dot product of two vectors.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner" title="numpy.inner"><code>inner</code></a>(a,&nbsp;b)</td>
      <td>Inner product of two arrays.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.outer.html#numpy.outer" title="numpy.outer"><code>outer</code></a>(a,&nbsp;b[,&nbsp;out])</td>
      <td>Compute the outer product of two vectors.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul" title="numpy.matmul"><code>matmul</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;casting,&nbsp;order,&nbsp;…])</td>
      <td>Matrix product of two arrays.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html#numpy.tensordot" title="numpy.tensordot"><code>tensordot</code></a>(a,&nbsp;b[,&nbsp;axes])</td>
      <td>Compute tensor dot product along specified axes.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.einsum.html#numpy.einsum" title="numpy.einsum"><code>einsum</code></a>(subscripts,&nbsp;*operands[,&nbsp;out,&nbsp;dtype,&nbsp;…])</td>
      <td>Evaluates the Einstein summation convention on the operands.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html#numpy.einsum_path" title="numpy.einsum_path"><code>einsum_path</code></a>(subscripts,&nbsp;*operands[,&nbsp;optimize])</td>
      <td>Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html#numpy.linalg.matrix_power" title="numpy.linalg.matrix_power"><code>linalg.matrix_power</code></a>(a,&nbsp;n)</td>
      <td>Raise a square matrix to the (integer) power <em class="xref py py-obj">n</em>.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.kron.html#numpy.kron" title="numpy.kron"><code>kron</code></a>(a,&nbsp;b)</td>
      <td>Kronecker product of two arrays.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y4n8yc9n">Decompositions</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky" title="numpy.linalg.cholesky"><code>linalg.cholesky</code></a>(a)</td>
      <td>Cholesky decomposition.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html#numpy.linalg.qr" title="numpy.linalg.qr"><code>linalg.qr</code></a>(a[,&nbsp;mode])</td>
      <td>Compute the qr factorization of a matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd" title="numpy.linalg.svd"><code>linalg.svd</code></a>(a[,&nbsp;full_matrices,&nbsp;compute_uv,&nbsp;…])</td>
      <td>Singular Value Decomposition.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yy34rdxx">Matrix eigenvalues</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig" title="numpy.linalg.eig"><code>linalg.eig</code></a>(a)</td>
      <td>Compute the eigenvalues and right eigenvectors of a square array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh" title="numpy.linalg.eigh"><code>linalg.eigh</code></a>(a[,&nbsp;UPLO])</td>
      <td>Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals" title="numpy.linalg.eigvals"><code>linalg.eigvals</code></a>(a)</td>
      <td>Compute the eigenvalues of a general matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh" title="numpy.linalg.eigvalsh"><code>linalg.eigvalsh</code></a>(a[,&nbsp;UPLO])</td>
      <td>Compute the eigenvalues of a complex Hermitian or real symmetric matrix.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yyrmojqp">Norms and other numbers</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm" title="numpy.linalg.norm"><code>linalg.norm</code></a>(x[,&nbsp;ord,&nbsp;axis,&nbsp;keepdims])</td>
      <td>Matrix or vector norm.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html#numpy.linalg.cond" title="numpy.linalg.cond"><code>linalg.cond</code></a>(x[,&nbsp;p])</td>
      <td>Compute the condition number of a matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html#numpy.linalg.det" title="numpy.linalg.det"><code>linalg.det</code></a>(a)</td>
      <td>Compute the determinant of an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html#numpy.linalg.matrix_rank" title="numpy.linalg.matrix_rank"><code>linalg.matrix_rank</code></a>(M[,&nbsp;tol,&nbsp;hermitian])</td>
      <td>Return matrix rank of array using SVD method</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet" title="numpy.linalg.slogdet"><code>linalg.slogdet</code></a>(a)</td>
      <td>Compute the sign and (natural) logarithm of the determinant of an array.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.trace.html#numpy.trace" title="numpy.trace"><code>trace</code></a>(a[,&nbsp;offset,&nbsp;axis1,&nbsp;axis2,&nbsp;dtype,&nbsp;out])</td>
      <td>Return the sum along diagonals of the array.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y4lzb4sq">Solving equations and inverting matrices</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve" title="numpy.linalg.solve"><code>linalg.solve</code></a>(a,&nbsp;b)</td>
      <td>Solve a linear matrix equation, or system of linear scalar equations.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorsolve.html#numpy.linalg.tensorsolve" title="numpy.linalg.tensorsolve"><code>linalg.tensorsolve</code></a>(a,&nbsp;b[,&nbsp;axes])</td>
      <td>Solve the tensor equation <code class="docutils literal notranslate"><span class="pre">a</span> <span class="pre">x</span> <span class="pre">=</span> <span class="pre">b</code> for x.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq" title="numpy.linalg.lstsq"><code>linalg.lstsq</code></a>(a,&nbsp;b[,&nbsp;rcond])</td>
      <td>Return the least-squares solution to a linear matrix equation.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv" title="numpy.linalg.inv"><code>linalg.inv</code></a>(a)</td>
      <td>Compute the (multiplicative) inverse of a matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv" title="numpy.linalg.pinv"><code>linalg.pinv</code></a>(a[,&nbsp;rcond,&nbsp;hermitian])</td>
      <td>Compute the (Moore-Penrose) pseudo-inverse of a matrix.</td>
    </tr>
    <tr>
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorinv.html#numpy.linalg.tensorinv" title="numpy.linalg.tensorinv"><code>linalg.tensorinv</code></a>(a[,&nbsp;ind])</td>
      <td>Compute the ‘inverse’ of an N-dimensional array.</td>
    </tr>
    </tbody>
  </table>


## Sorting, searching, and counting

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y3zpsdcd">Sorting</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.sort.html#numpy.sort" title="numpy.sort"><code>sort</code></a>(a[,&nbsp;axis,&nbsp;kind,&nbsp;order])</td> 
      <td>Return a sorted copy of an array.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort" title="numpy.lexsort"><code>lexsort</code></a>(keys[,&nbsp;axis])</td> 
      <td>Perform an indirect stable sort using a sequence of keys.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.argsort.html#numpy.argsort" title="numpy.argsort"><code>argsort</code></a>(a[,&nbsp;axis,&nbsp;kind,&nbsp;order])</td> 
      <td>Returns the indices that would sort an array.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sort.html#numpy.ndarray.sort" title="numpy.ndarray.sort"><code>ndarray.sort</code></a>([axis,&nbsp;kind,&nbsp;order])</td> 
      <td>Sort an array in-place.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.msort.html#numpy.msort" title="numpy.msort"><code>msort</code></a>(a)</td> 
      <td>Return a copy of an array sorted along the first axis.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.sort_complex.html#numpy.sort_complex" title="numpy.sort_complex"><code>sort_complex</code></a>(a)</td> 
      <td>Sort a complex array using the real part first, then the imaginary part.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition" title="numpy.partition"><code>partition</code></a>(a,&nbsp;kth[,&nbsp;axis,&nbsp;kind,&nbsp;order])</td> 
      <td>Return a partitioned copy of an array.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html#numpy.argpartition" title="numpy.argpartition"><code>argpartition</code></a>(a,&nbsp;kth[,&nbsp;axis,&nbsp;kind,&nbsp;order])</td> 
      <td>Perform an indirect partition along the given axis using the algorithm specified by the <em class="xref py py-obj">kind</em> keyword.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y68fh56n">Searching</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.argmax.html#numpy.argmax" title="numpy.argmax"><code>argmax</code></a>(a[,&nbsp;axis,&nbsp;out])</td> 
      <td>Returns the indices of the maximum values along an axis.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html#numpy.nanargmax" title="numpy.nanargmax"><code>nanargmax</code></a>(a[,&nbsp;axis])</td> 
      <td>Return the indices of the maximum values in the specified axis ignoring NaNs.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.argmin.html#numpy.argmin" title="numpy.argmin"><code>argmin</code></a>(a[,&nbsp;axis,&nbsp;out])</td> 
      <td>Returns the indices of the minimum values along an axis.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html#numpy.nanargmin" title="numpy.nanargmin"><code>nanargmin</code></a>(a[,&nbsp;axis])</td> 
      <td>Return the indices of the minimum values in the specified axis ignoring NaNs.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html#numpy.argwhere" title="numpy.argwhere"><code>argwhere</code></a>(a)</td> 
      <td>Find the indices of array elements that are non-zero, grouped by element.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy.nonzero" title="numpy.nonzero"><code>nonzero</code></a>(a)</td> 
      <td>Return the indices of the elements that are non-zero.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html#numpy.flatnonzero" title="numpy.flatnonzero"><code>flatnonzero</code></a>(a)</td> 
      <td>Return indices that are non-zero in the flattened version of a.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where" title="numpy.where"><code>where</code></a>(condition,&nbsp;[x,&nbsp;y])</td> 
      <td>Return elements chosen from <em class="xref py py-obj">x</em> or <em class="xref py py-obj">y</em> depending on <em class="xref py py-obj">condition</em>.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy.searchsorted" title="numpy.searchsorted"><code>searchsorted</code></a>(a,&nbsp;v[,&nbsp;side,&nbsp;sorter])</td> 
      <td>Find indices where elements should be inserted to maintain order.</td>
    </tr>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.extract.html#numpy.extract" title="numpy.extract"><code>extract</code></a>(condition,&nbsp;arr)</td> 
      <td>Return the elements of an array that satisfy some condition.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y5o2mh9y">Counting</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr >
      <td><a href="https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero" title="numpy.count_nonzero"><code>count_nonzero</code></a>(a[,&nbsp;axis,&nbsp;keepdims])</td> 
      <td>Counts the number of non-zero values in the array <code class="docutils literal notranslate"><span class="pre">a</code>.</td>
    </tr>
    </tbody>
  </table>



