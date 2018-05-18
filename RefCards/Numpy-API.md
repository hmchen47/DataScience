# Numpy References

## [Array creation routines][000]

### Ones and zeros

| API | Description | Link |
|-----|-------------|------|
| `empty(shape[, dtype, order])` | Return a new array of given shape and type, without initializing entries. | [API][0001] |
| `empty_like(prototype[, dtype, order, subok])` | Return a new array with the same shape and type as a given array. | [API][0002] |
| `eye(N[, M, k, dtype, order])` | Return a 2-D array with ones on the diagonal and zeros elsewhere. | [API][0003] |
| `identity(n[, dtype])` | Return the identity array. | [API][0004] |
| `ones(shape[, dtype, order])` | Return a new array of given shape and type, filled with ones. | [API][0005] |
| `ones_like(a[, dtype, order, subok])` | Return an array of ones with the same shape and type as a given array. | [API][0006] |
| `zeros(shape[, dtype, order])` | Return a new array of given shape and type, filled with zeros. | [API][0007] |
| `zeros_like(a[, dtype, order, subok])` | Return an array of zeros with the same shape and type as a given array. | [API][0008] |
| `full(shape, fill_value[, dtype, order])` | Return a new array of given shape and type, filled with fill_value. | [API][0009] |
| `full_like(a, fill_value[, dtype, order, subok])` | Return a full array with the same shape and type as a given array. | [API][0010] |

### From existing data

| API | Description | Link |
|-----|-------------|------|
| `array(object[, dtype, copy, order, subok, ndmin])` | Create an array.|[API][0011] |
| `asarray(a[, dtype, order])` | Convert the input to an array.|[API][0012] |
| `asanyarray(a[, dtype, order])` | Convert the input to an ndarray, but pass ndarray subclasses through.|[API][0013] |
| `ascontiguousarray(a[, dtype])` | Return a contiguous array in memory (C order).|[API][0014] |
| `asmatrix(data[, dtype])` | Interpret the input as a matrix.|[API][0015] |
| `copy(a[, order])` | Return an array copy of the given object.|[API][0016] |
| `frombuffer(buffer[, dtype, count, offset])` | Interpret a buffer as a 1-dimensional array.|[API][0017] |
| `fromfile(file[, dtype, count, sep])` | Construct an array from data in a text or binary file.|[API][0018] |
| `fromfunction(function, shape, **kwargs)` | Construct an array by executing a function over each coordinate.|[API][0019] |
| `fromiter(iterable, dtype[, count])` | Create a new 1-dimensional array from an iterable object.|[API][00120] |
| `fromstring(string[, dtype, count, sep])` | A new 1-D array initialized from text data in a string.|[API][0021] |
| `loadtxt(fname[, dtype, comments, delimiter, …])` | Load data from a text file.|[API][0022] |

### Creating record arrays (numpy.rec)

Note: `numpy.rec` is the preferred alias for `numpy.core.records`.

| API | Description | Link |
|-----|-------------|------|
| `core.records.array(obj[, dtype, shape, …])` | Construct a record array from a wide-variety of objects. | [API][0023] |
| `core.records.fromarrays(arrayList[, dtype, …])` | create a record array from a (flat) list of arrays | [API][0024] |
| `core.records.fromrecords(recList[, dtype, …])` | create a recarray from a list of records in text form | [API][0025] |
| `core.records.fromstring(datastring[, dtype, …])` | create a (read-only) record array from binary data contained in a string | [API][0026] |
| `core.records.fromfile(fd[, dtype, shape, …])` | Create an array from binary file data | [API][0027] |

### Creating character arrays (numpy.char)

Note: `numpy.char` is the preferred alias for `numpy.core.defchararray`.

| API | Description | Link |
|-----|-------------|------|
| `core.defchararray.array(obj[, itemsize, …])` | Create a chararray. | [API][0028]
| `core.defchararray.asarray(obj[, itemsize, …])` | Convert the input to a chararray, copying the data only if necessary. | [API][0029]

### Numerical ranges

| API | Description | Link |
|-----|-------------|------|
| `arange([start,] stop[, step,][, dtype])` | Return evenly spaced values within a given interval. | [API][0030] |
| `linspace(start, stop[, num, endpoint, …])` | Return evenly spaced numbers over a specified interval. | [API][0031] |
| `logspace(start, stop[, num, endpoint, base, …])` | Return numbers spaced evenly on a log scale. | [API][0032] |
| `geomspace(start, stop[, num, endpoint, dtype])` | Return numbers spaced evenly on a log scale (a geometric progression). | [API][0033] |
| `meshgrid(*xi, **kwargs)` | Return coordinate matrices from coordinate vectors. | [API][0034] |
| `mgrid` | nd_grid instance which returns a dense multi-dimensional “meshgrid”. | [API][0035] |
| `ogrid` | nd_grid instance which returns an open multi-dimensional “meshgrid”. | [API][0036] |

### Building matrices

| API | Description | Link |
|-----|-------------|------|
| `diag(v[, k])` | Extract a diagonal or construct a diagonal array. | [API][0037] |
| `diagflat(v[, k])` | Create a two-dimensional array with the flattened input as a diagonal. | [API][0038] |
| `tri(N[, M, k, dtype])` | An array with ones at and below the given diagonal and zeros elsewhere. | [API][0039] |
| `tril(m[, k])` | Lower triangle of an array. | [API][0040] |
| `triu(m[, k])` | Upper triangle of an array. | [API][0041] |
| `vander(x[, N, increasing])` | Generate a Vandermonde matrix. | [API][0042] |

### The Matrix class

| API | Description | Link |
|-----|-------------|------|
| `mat(data[, dtype])` | Interpret the input as a matrix. | [API][0043] |
| `bmat(obj[, ldict, gdict])` | Build a matrix object from a string, nested sequence, or array. | [API][0044] |


## [Array manipulation routines][0045]

### Basic operations

| API | Description | Link |
|-----|-------------|------|
| `copyto(dst, src[, casting, where])` | Copies values from one array to another, broadcasting as necessary. | [API][0046] |

### Changing array shape

| API | Description | Link |
|-----|-------------|------|
| `reshape(a, newshape[, order])` | Gives a new shape to an array without changing its data. | [API][0047] |
| `ravel(a[, order])` | Return a contiguous flattened array. | [API][0048] |
| `ndarray.flat` | A 1-D iterator over the array. | [API][0049] |
| `ndarray.flatten([order])` | Return a copy of the array collapsed into one dimension. | [API][0050] |

### Transpose-like operations

| API | Description | Link |
|-----|-------------|------|
| `moveaxis(a, source, destination)` | Move axes of an array to new positions. | [API][0051] |
| `rollaxis(a, axis[, start])` | Roll the specified axis backwards, until it lies in a given position. | [API][0052] |
| `swapaxes(a, axis1, axis2)` | Interchange two axes of an array. | [API][0053] |
| `ndarray.T` | Same as self.transpose(), except that self is returned if self.ndim < 2. | [API][0054] |
| `transpose(a[, axes])` | Permute the dimensions of an array. | [API][0055] |

### Changing number of dimensions

| API | Description | Link |
|-----|-------------|------|
| `atleast_1d(*arys)` | Convert inputs to arrays with at least one dimension. | [API][0056] |
| `atleast_2d(*arys)` | View inputs as arrays with at least two dimensions. | [API][0057] |
| `atleast_3d(*arys)` | View inputs as arrays with at least three dimensions. | [API][0058] |
| `broadcast` | Produce an object that mimics broadcasting. | [API][0059] |
| `broadcast_to(array, shape[, subok])` | Broadcast an array to a new shape. | [API][0060] |
| `broadcast_arrays(*args, **kwargs)` | Broadcast any number of arrays against each other. | [API][0061] |
| `expand_dims(a, axis)` | Expand the shape of an array. | [API][0062] |
| `squeeze(a[, axis])` | Remove single-dimensional entries from the shape of an array. | [API][0063] |

### Changing kind of array

| API | Description | Link |
|-----|-------------|------|
| `asarray(a[, dtype, order])` | Convert the input to an array. | [API][0064] |
| `asanyarray(a[, dtype, order])` | Convert the input to an ndarray, but pass ndarray subclasses through. | [API][0065] |
| `asmatrix(data[, dtype])` | Interpret the input as a matrix. | [API][0066] |
| `asfarray(a[, dtype])` | Return an array converted to a float type. | [API][0067] |
| `asfortranarray(a[, dtype])` | Return an array laid out in Fortran order in memory. | [API][0068] |
| `ascontiguousarray(a[, dtype])` | Return a contiguous array in memory (C order). | [API][0069] |
| `asarray_chkfinite(a[, dtype, order])` | Convert the input to an array, checking for NaNs or Infs. | [API][0070] |
| `asscalar(a)` | Convert an array of size 1 to its scalar equivalent. | [API][0071] |
| `require(a[, dtype, requirements])` | Return an ndarray of the provided type that satisfies requirements. | [API][0072] |

### Joining arrays

| API | Description | Link |
|-----|-------------|------|
| `concatenate((a1, a2, …)[, axis, out])` | Join a sequence of arrays along an existing axis. | [API][0073] |
| `stack(arrays[, axis, out])` | Join a sequence of arrays along a new axis. | [API][0074] |
| `column_stack(tup)` | Stack 1-D arrays as columns into a 2-D array. | [API][0075] |
| `dstack(tup)` | Stack arrays in sequence depth wise (along third axis). | [API][0076] |
| `hstack(tup)` | Stack arrays in sequence horizontally (column wise). | [API][0077] |
| `vstack(tup)` | Stack arrays in sequence vertically (row wise). | [API][0078] |
| `block(arrays)` | Assemble an nd-array from nested lists of blocks. | [API][0079] |

### Splitting arrays

| API | Description | Link |
|-----|-------------|------|
| `split(ary, indices_or_sections[, axis])` | Split an array into multiple sub-arrays. | [API][0080] |
| `array_split(ary, indices_or_sections[, axis])` | Split an array into multiple sub-arrays. | [API][0081] |
| `dsplit(ary, indices_or_sections)` | Split array into multiple sub-arrays along the 3rd axis (depth). | [API][0082] |
| `hsplit(ary, indices_or_sections)` | Split an array into multiple sub-arrays horizontally (column-wise). | [API][0083] |
| `vsplit(ary, indices_or_sections)` | Split an array into multiple sub-arrays vertically (row-wise). | [API][0084] |

### Tiling arrays

| API | Description | Link |
|-----|-------------|------|
| `tile(A, reps)` | Construct an array by repeating A the number of times given by reps. | [API][0085] |
| `repeat(a, repeats[, axis])` | Repeat elements of an array. | [API][0086] |

### Adding and removing elements

| API | Description | Link |
|-----|-------------|------|
| `delete(arr, obj[, axis])` | Return a new array with sub-arrays along an axis deleted. | [API][0087] |
| `insert(arr, obj, values[, axis])` | Insert values along the given axis before the given indices. | [API][0088] |
| `append(arr, values[, axis])` | Append values to the end of an array. | [API][0089] |
| `resize(a, new_shape)` | Return a new array with the specified shape. | [API][0090] |
| `trim_zeros(filt[, trim])` | Trim the leading and/or trailing zeros from a 1-D array or sequence. | [API][0091] |
| `unique(ar[, return_index, return_inverse, …])` | Find the unique elements of an array. | [API][0092] |

### Rearranging elements

| API | Description | Link |
|-----|-------------|------|
| `flip(m[, axis])` | Reverse the order of elements in an array along the given axis. | [API][0093] |
| `fliplr(m)` | Flip array in the left/right direction. | [API][0094] |
| `flipud(m)` | Flip array in the up/down direction. | [API][0095] |
| `reshape(a, newshape[, order])` | Gives a new shape to an array without changing its data. | [API][0096] |
| `roll(a, shift[, axis])` | Roll array elements along a given axis. | [API][0097] |
| `rot90(m[, k, axes])` | Rotate an array by 90 degrees in the plane specified by axes. | [API][0098] |

## [Binary operations][0099]

### Elementwise bit operations 

| API | Description | Link |
|-----|-------------|------|
| `bitwise_and(x1, x2, /[, out, where, …])` | Compute the bit-wise AND of two arrays element-wise. | [API][0100] |
| `bitwise_or(x1, x2, /[, out, where, casting, …])` | Compute the bit-wise OR of two arrays element-wise. | [API][0101] |
| `bitwise_xor(x1, x2, /[, out, where, …])` | Compute the bit-wise XOR of two arrays element-wise. | [API][0102] |
| `invert(x, /[, out, where, casting, order, …])` | Compute bit-wise inversion, or bit-wise NOT, element-wise. | [API][0103] |
| `left_shift(x1, x2, /[, out, where, casting, …])` | Shift the bits of an integer to the left. | [API][0104] |
| `right_shift(x1, x2, /[, out, where, …])` | Shift the bits of an integer to the right. | [API][0105] |

### Bit packing

| API | Description | Link |
|-----|-------------|------|
| `packbits(myarray[, axis])` | Packs the elements of a binary-valued array into bits in a uint8 array. | [API][0106] |
| `unpackbits(myarray[, axis])` | Unpacks elements of a uint8 array into a binary-valued output array. | [API][0107] |

### Output formatting

| API | Description | Link |
|-----|-------------|------|
| `binary_repr(num[, width])` | Return the binary representation of the input number as a string. | [API][0108] |


## [String operations][0109]


### String operations

| API | Description | Link |
|-----|-------------|------|
| `add(x1, x2)` | Return element-wise string concatenation for two arrays of str or unicode. | [API][0110] |
| `multiply(a, i)` | Return (a * i), that is string multiple concatenation, element-wise. | [API][0111] |
| `mod(a, values)` | Return (a % i), that is pre-Python 2.6 string formatting (iterpolation), element-wise for a pair of array_likes of str or unicode. | [API][0112] |
| `capitalize(a)` | Return a copy of a with only the first character of each element capitalized. | [API][0113] |
| `center(a, width[, fillchar])` | Return a copy of a with its elements centered in a string of length width. | [API][0114] |
| `decode(a[, encoding, errors])` | Calls str.decode element-wise. | [API][0115] |
| `encode(a[, encoding, errors])` | Calls str.encode element-wise. | [API][0116] |
| `join(sep, seq)` | Return a string which is the concatenation of the strings in the sequence seq. | [API][0117] |
| `ljust(a, width[, fillchar])` | Return an array with the elements of a left-justified in a string of length width. | [API][0118] |
| `lower(a)` | Return an array with the elements converted to lowercase. | [API][0119] |
| `lstrip(a[, chars])` | For each element in a, return a copy with the leading characters removed. | [API][0120] |
| `partition(a, sep)` | Partition each element in a around sep. | [API][0121] |
| `replace(a, old, new[, count])` | For each element in a, return a copy of the string with all occurrences of substring old replaced by new. | [API][0122] |
| `rjust(a, width[, fillchar])` | Return an array with the elements of a right-justified in a string of length width. | [API][0123] |
| `rpartition(a, sep)` | Partition (split) each element around the right-most separator. | [API][0124] |
| `rsplit(a[, sep, maxsplit])` | For each element in a, return a list of the words in the string, using sep as the delimiter string. | [API][0125] |
| `rstrip(a[, chars])` | For each element in a, return a copy with the trailing characters removed. | [API][0126] |
| `split(a[, sep, maxsplit])` | For each element in a, return a list of the words in the string, using sep as the delimiter string. | [API][0127] |
| `splitlines(a[, keepends])` | For each element in a, return a list of the lines in the element, breaking at line boundaries. | [API][0128] |
| `strip(a[, chars])` | For each element in a, return a copy with the leading and trailing characters removed. | [API][0129] |
| `swapcase(a)` | Return element-wise a copy of the string with uppercase characters converted to lowercase and vice versa. | [API][0130] |
| `title(a)` | Return element-wise title cased version of string or unicode. | [API][0131] |
| `translate(a, table[, deletechars])` | For each element in a, return a copy of the string where all characters occurring in the optional argument deletechars  are removed, and the remaining characters have been mapped through the given translation table. | [API][0132] |
| `upper(a)` | Return an array with the elements converted to uppercase. | [API][0133] |
| `zfill(a, width)` | Return the numeric string left-filled with zeros | [API][0134] |

### Comparison

Unlike the standard numpy comparison operators, the ones in the char module strip trailing whitespace characters before performing the comparison.

| API | Description | Link |
|-----|-------------|------|
| `equal(x1, x2)` | Return (x1 == x2) element-wise. | [API][0135] |
| `not_equal(x1, x2)` | Return (x1 != x2) element-wise. | [API][0136] |
| `greater_equal(x1, x2)` | Return (x1 >= x2) element-wise. | [API][0137] |
| `less_equal(x1, x2)` | Return (x1 <= x2) element-wise. | [API][0138] |
| `greater(x1, x2)` | Return (x1 > x2) element-wise. | [API][0139] |
| `less(x1, x2)` | Return (x1 < x2) element-wise. | [API][0140] |

### String information

| API | Description | Link |
|-----|-------------|------|
| `count(a, sub[, start, end])` | Returns an array with the number of non-overlapping occurrences of substring sub in the range [start, end]. | [API][0141] |
| `find(a, sub[, start, end])` | For each element, return the lowest index in the string where substring sub is found. | [API][0142] |
| `index(a, sub[, start, end])` | Like find, but raises ValueError when the substring is not found. | [API][0143] |
| `isalpha(a)` | Returns true for each element if all characters in the string are alphabetic and there is at least one character, false otherwise. | [API][0144] |
| `isdecimal(a)` | For each element, return True if there are only decimal characters in the element. | [API][0145] |
| `isdigit(a)` | Returns true for each element if all characters in the string are digits and there is at least one character, false otherwise. | [API][0146] |
| `islower(a)` | Returns true for each element if all cased characters in the string are lowercase and there is at least one cased character, false otherwise. | [API][0147] |
| `isnumeric(a)` | For each element, return True if there are only numeric characters in the element. | [API][0148] |
| `isspace(a)` | Returns true for each element if there are only whitespace characters in the string and there is at least one character, false otherwise. | [API][0149] |
| `istitle(a)` | Returns true for each element if the element is a titlecased string and there is at least one character, false otherwise. | [API][0150] |
| `isupper(a)` | Returns true for each element if all cased characters in the string are uppercase and there is at least one character, false otherwise. | [API][0151] |
| `rfind(a, sub[, start, end])` | For each element in a, return the highest index in the string where substring sub is found, such that sub is contained within | [start, end]. | [API][0152] |
| `rindex(a, sub[, start, end])` | Like rfind, but raises ValueError when the substring sub is not found. | [API][0153] |
| `startswith(a, prefix[, start, end])` | Returns a boolean array which is True where the string element in a starts with prefix, otherwise False. | [API][0154] |

### Convenience class

| API | Description | Link |
|-----|-------------|------|
| `chararray(shape[, itemsize, unicode, …])` | Provides a convenient view on arrays of string and unicode values. | [API][0155] |


## [C-Types Foreign Function Interface (numpy.ctypeslib)][0156]

| API | Description | Link |
|-----|-------------|------|
| `numpy.ctypeslib.as_array(obj, shape=None)` | Create a numpy array from a ctypes array or a ctypes POINTER. The numpy array shares the memory with the ctypes object. | [API][0156] |
| `numpy.ctypeslib.as_ctypes(obj)` | Create and return a ctypes object from a numpy array. Actually anything that exposes the `__array_interface__` is accepted. | [API][0156] |
| `numpy.ctypeslib.load_library(libname, loader_path)` | It is possible to load a library using `>>> lib = ctypes.cdll[<full_path_name>]`; `libname`: str, Name of the library, which can have ‘lib’ as a prefix, but without an extension; `loader_path`: str | [API][0156] |
| `numpy.ctypeslib.ndpointer(dtype=None, ndim=None, shape=None, flags=None)` | Array-checking restype/argtypes; `flags`: str or tuple of str, {C_CONTIGUOUS / C / CONTIGUOUS, F_CONTIGUOUS / F / FORTRAN, OWNDATA / O, WRITEABLE / W, ALIGNED / A, WRITEBACKIFCOPY / X, UPDATEIFCOPY / U} | [API][0156] |

## [Datetime Support Functions][0157]


| API | Description | Link |
|-----|-------------|------|
| `datetime_as_string(arr[, unit, timezone, …])` | Convert an array of datetimes into an array of strings. | [API][0158] |
| `datetime_data(dtype, /)` | Get information about the step size of a date or time type. | [API][0159] |

### Business Day Functions

| API | Description | Link |
|-----|-------------|------|
| `busdaycalendar([weekmask, holidays])` | A business day calendar object that efficiently stores information defining valid days for the busday family of functions. | [API][0160] |
| `is_busday(dates[, weekmask, holidays, …])` | Calculates which of the given dates are valid days, and which are not. | [API][0161] |
| `busday_offset(dates, offsets[, roll, …])` | First adjusts the date to fall on a valid day according to the roll rule, then applies offsets to the given dates  counted in valid days. | [API][0162] |
| `busday_count(begindates, enddates[, …])` | Counts the number of valid days between begindates and enddates, not including the day of enddates. | [API][0163] |



## [Data type routines][0163]


| API | Description | Link |
|-----|-------------|------|
| `can_cast(from_, to[, casting])` | Returns True if cast between data types can occur according to the casting rule. | [API][0164]
| `promote_types(type1, type2)` | Returns the data type with the smallest size and smallest scalar kind to which both type1 and type2 may be safely cast. | [API][0165]
| `min_scalar_type(a)` | For scalar a, returns the data type with the smallest size and smallest scalar kind which can hold its value. | [API][0166]
| `result_type(*arrays_and_dtypes)` | Returns the type that results from applying the NumPy type promotion rules to the arguments. | [API][0167]
| `common_type(*arrays)` | Return a scalar type which is common to the input arrays. | [API][0168]
| `obj2sctype(rep[, default])` | Return the scalar dtype or NumPy equivalent of Python type of an object. | [API][0169]

### Creating data types 

| API | Description | Link |
|-----|-------------|------|
| `dtype(obj[, align, copy])` | Create a data type object. | [API][0170]
| `format_parser(formats, names, titles[, …])` | Class to convert formats, names, titles description to a dtype. | [API][0171]

### Data type information

| API | Description | Link |
|-----|-------------|------|
| `finfo(dtype)` | Machine limits for floating point types. | [API][0172]
| `iinfo(type)` | Machine limits for integer types. | [API][0173]
| `MachAr([float_conv, int_conv, …])` | Diagnosing machine parameters. | [API][0174]

### Data type testing

| API | Description | Link |
|-----|-------------|------|
| `issctype(rep)` | Determines whether the given object represents a scalar data-type. | [API][0175]
| `issubdtype(arg1, arg2)` | Returns True if first argument is a typecode lower/equal in type hierarchy. | [API][0176]
| `issubsctype(arg1, arg2)` | Determine if the first argument is a subclass of the second argument. | [API][0177]
| `issubclass_(arg1, arg2)` | Determine if a class is a subclass of a second class. | [API][0178]
| `find_common_type(array_types, scalar_types)` | Determine common type following standard coercion rules. | [API][0179]

### Miscellaneous

| API | Description | Link |
|-----|-------------|------|
| `typename(char)` | Return a description for the given data type code. | [API][0180]
| `sctype2char(sctype)` | Return the string representation of a scalar dtype. | [API][0181]
| `mintypecode(typechars[, typeset, default])` | Return the character for the minimum-size type to which given types can be safely cast. | [API][0182]


## [Mathematical functions with automatic domain (numpy.emath)][0183]

Note: `numpy.emath` is a preferred alias for `numpy.lib.scimath`, available after `numpy` is imported.

## [Floating point error handling][0184]

### Setting and getting error handling

| API | Description | Link |
|-----|-------------|------|
| `seterr([all, divide, over, under, invalid])` | Set how floating-point errors are handled. | [API][0185] |
| `geterr()` | Get the current way of handling floating-point errors. | [API][0186] |
| `seterrcall(func)` | Set the floating-point error callback function or log object. | [API][0187] |
| `geterrcall()` | Return the current callback function used on floating-point errors. | [API][0188] |
| `errstate(**kwargs)` | Context manager for floating-point error handling. | [API][0189] |

### Internal functions

| API | Description | Link |
|-----|-------------|------|
| `seterrobj(errobj)` | Set the object that defines floating-point error handling. | [API][0190] |
| `geterrobj()` | Return the current object that defines floating-point error handling. | [API][0191] |

## [Discrete Fourier Transform (numpy.fft)][0192]

### Standard FFTs

| API | Description | Link |
|-----|-------------|------|
| `fft(a[, n, axis, norm])` | Compute the one-dimensional discrete Fourier Transform. | [API][0193]
| `ifft(a[, n, axis, norm])` | Compute the one-dimensional inverse discrete Fourier Transform. | [API][0194]
| `fft2(a[, s, axes, norm])` | Compute the 2-dimensional discrete Fourier Transform | [API][0195]
| `ifft2(a[, s, axes, norm])` | Compute the 2-dimensional inverse discrete Fourier Transform. | [API][0196]
| `fftn(a[, s, axes, norm])` | Compute the N-dimensional discrete Fourier Transform. | [API][0197]
| `ifftn(a[, s, axes, norm])` | Compute the N-dimensional inverse discrete Fourier Transform. | [API][0198]

### Real FFTs

| API | Description | Link |
|-----|-------------|------|
| `rfft(a[, n, axis, norm])` | Compute the one-dimensional discrete Fourier Transform for real input. | [API][0199]
| `irfft(a[, n, axis, norm])` | Compute the inverse of the n-point DFT for real input. | [API][0200]
| `rfft2(a[, s, axes, norm])` | Compute the 2-dimensional FFT of a real array. | [API][0201]
| `irfft2(a[, s, axes, norm])` | Compute the 2-dimensional inverse FFT of a real array. | [API][0202]
| `rfftn(a[, s, axes, norm])` | Compute the N-dimensional discrete Fourier Transform for real input. | [API][0203]
| `irfftn(a[, s, axes, norm])` | Compute the inverse of the N-dimensional FFT of real input. | [API][0204]

### Hermitian FFTs

| API | Description | Link |
|-----|-------------|------|
| `hfft(a[, n, axis, norm])` | Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum. | [API][0205]
| `ihfft(a[, n, axis, norm])` | Compute the inverse FFT of a signal that has Hermitian symmetry. | [API][0206]

### Helper routines

| API | Description | Link |
|-----|-------------|------|
| `fftfreq(n[, d])` | Return the Discrete Fourier Transform sample frequencies. | [API][0207]
| `rfftfreq(n[, d])` | Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft). | [API][0208]
| `fftshift(x[, axes])` | Shift the zero-frequency component to the center of the spectrum. | [API][0209]
| `ifftshift(x[, axes])` | The inverse of fftshift. | [API][0210]


## [Financial functions][0211]

### Simple financial functions

| API | Description | Link |
|-----|-------------|------|
| `fv(rate, nper, pmt, pv[, when])` | Compute the future value. | [API][0212] |
| `pv(rate, nper, pmt[, fv, when])` | Compute the present value. | [API][0213] |
| `npv(rate, values)` | Returns the NPV (Net Present Value) of a cash flow series. | [API][0214] |
| `pmt(rate, nper, pv[, fv, when])` | Compute the payment against loan principal plus interest. | [API][0215] |
| `ppmt(rate, per, nper, pv[, fv, when])` | Compute the payment against loan principal. | [API][0216] |
| `ipmt(rate, per, nper, pv[, fv, when])` | Compute the interest portion of a payment. | [API][0217] |
| `irr(values)` | Return the Internal Rate of Return (IRR). | [API][0218] |
| `mirr(values, finance_rate, reinvest_rate)` | Modified internal rate of return. | [API][0219] |
| `nper(rate, pmt, pv[, fv, when])` | Compute the number of periodic payments. | [API][0220] |
| `rate(nper, pmt, pv, fv[, when, guess, tol, …])` | Compute the rate of interest per period. | [API][0221] |


## [Functional programming][0222]

| API | Description | Link |
|-----|-------------|------|
| `apply_along_axis(func1d, axis, arr, *args, …)` | Apply a function to 1-D slices along the given axis. | [API][0223] |
| `apply_over_axes(func, a, axes)` | Apply a function repeatedly over multiple axes. | [API][0224] |
| `vectorize(pyfunc[, otypes, doc, excluded, …])` | Generalized function class. | [API][0225] |
| `frompyfunc(func, nin, nout)` | Takes an arbitrary Python function and returns a NumPy ufunc. | [API][0226] |
| `piecewise(x, condlist, funclist, *args, **kw)` | Evaluate a piecewise-defined function. | [API][0227] |


## [NumPy-specific help functions][0228]

### Finding help

| API | Description | Link |
|-----|-------------|------|
| `lookfor(what[, module, import_modules, …])` | Do a keyword search on docstrings. | [API][0229] |

### Reading help

| API | Description | Link |
|-----|-------------|------|
| `info([object, maxwidth, output, toplevel])` | Get help information for a function, class, or module. | [API][0230] |
| `source(object[, output])` | Print or write to a file the source code for a NumPy object. | [API][0231] |


## [Indexing routines][0232]

### Generating index arrays

| API | Description | Link |
|-----|-------------|------|
| `c_` | Translates slice objects to concatenation along the second axis. | [API][0233] |
| `r_` | Translates slice objects to concatenation along the first axis. | [API][0234] |
| `s_` | A nicer way to build up index tuples for arrays. | [API][0235] |
| `nonzero(a)` | Return the indices of the elements that are non-zero. | [API][0236] |
| `where(condition, [x, y])` | Return elements, either from x or y, depending on condition. | [API][0237] |
| `indices(dimensions[, dtype])` | Return an array representing the indices of a grid. | [API][0238] |
| `ix_(*args)` | Construct an open mesh from multiple sequences. | [API][0239] |
| `ogrid` | nd_grid instance which returns an open multi-dimensional “meshgrid”. | [API][0240] |
| `ravel_multi_index(multi_index, dims[, mode, …])` | Converts a tuple of index arrays into an array of flat indices, applying boundary modes to the multi-index. | [API][0241] |
| `unravel_index(indices, dims[, order])` | Converts a flat index or array of flat indices into a tuple of coordinate arrays. | [API][0242] |
| `diag_indices(n[, ndim])` | Return the indices to access the main diagonal of an array. | [API][0243] |
| `diag_indices_from(arr)` | Return the indices to access the main diagonal of an n-dimensional array. | [API][0244] |
| `mask_indices(n, mask_func[, k])` | Return the indices to access (n, n) arrays, given a masking function. | [API][0245] |
| `tril_indices(n[, k, m])` | Return the indices for the lower-triangle of an (n, m) array. | [API][0246] |
| `tril_indices_from(arr[, k])` | Return the indices for the lower-triangle of arr. | [API][0247] |
| `triu_indices(n[, k, m])` | Return the indices for the upper-triangle of an (n, m) array. | [API][0248] |
| `triu_indices_from(arr[, k])` | Return the indices for the upper-triangle of arr. | [API][0249] |

### Indexing-like operations

| API | Description | Link |
|-----|-------------|------|
| `take(a, indices[, axis, out, mode])` | Take elements from an array along an axis. | [API][0250] |
| `choose(a, choices[, out, mode])` | Construct an array from an index array and a set of arrays to choose from. | [API][0251] |
| `compress(condition, a[, axis, out])` | Return selected slices of an array along given axis. | [API][0252] |
| `diag(v[, k])` | Extract a diagonal or construct a diagonal array. | [API][0253] |
| `diagonal(a[, offset, axis1, axis2])` | Return specified diagonals. | [API][0254] |
| `select(condlist, choicelist[, default])` | Return an array drawn from elements in choicelist, depending on conditions. | [API][0255] |
| `lib.stride_tricks.as_strided(x[, shape, …])` | Create a view into the array with the given shape and strides. | [API][0256] |

### Inserting data into arrays

| API | Description | Link |
|-----|-------------|------|
| `place(arr, mask, vals)` | Change elements of an array based on conditional and input values. | [API][0257] |
| `put(a, ind, v[, mode])` | Replaces specified elements of an array with given values. | [API][0258] |
| `putmask(a, mask, values)` | Changes elements of an array based on conditional and input values. | [API][0259] |
| `fill_diagonal(a, val[, wrap])` | Fill the main diagonal of the given array of any dimensionality. | [API][0260] |

### Iterating over arrays

| API | Description | Link |
|-----|-------------|------|
| `nditer` | Efficient multi-dimensional iterator object to iterate over arrays. | [API][0261] |
| `ndenumerate(arr)` | Multidimensional index iterator. | [API][0262] |
| `ndindex(*shape)` | An N-dimensional iterator object to index arrays. | [API][0263] |
| `nested_iters` | Create nditers for use in nested loops | [API][0264] |
| `flatiter` | Flat iterator object to iterate over arrays. | [API][0265] |
| `lib.Arrayterator(var[, buf_size])` | Buffered iterator for big arrays. | [API][0266] |


## [Input and output][0267]

### NumPy binary files (NPY, NPZ)

| API | Description | Link |
|-----|-------------|------|
| `load(file[, mmap_mode, allow_pickle, …])` | Load arrays or pickled objects from .npy, .npz or pickled files. | [API][0268] |
| `save(file, arr[, allow_pickle, fix_imports])` | Save an array to a binary file in NumPy .npy format. | [API][0269] |
| `savez(file, *args, **kwds)` | Save several arrays into a single file in uncompressed .npz format. | [API][0270] |
| `savez_compressed(file, *args, **kwds)` | Save several arrays into a single file in compressed .npz format. | [API][0271] |

The format of these binary file types is documented in [`numpy.lib.format`](https://www.numpy.org/devdocs/reference/generated/numpy.lib.format.html#module-numpy.lib.format)

### Text files

| API | Description | Link |
|-----|-------------|------|
| `loadtxt(fname[, dtype, comments, delimiter, …])` | Load data from a text file. | [API][0272] |
| `savetxt(fname, X[, fmt, delimiter, newline, …])` | Save an array to a text file. | [API][0273] |
| `genfromtxt(fname[, dtype, comments, …])` | Load data from a text file, with missing values handled as specified. | [API][0274] |
| `fromregex(file, regexp, dtype[, encoding])` | Construct an array from a text file, using regular expression parsing. | [API][0275] |
| `fromstring(string[, dtype, count, sep])` | A new 1-D array initialized from text data in a string. | [API][0276] |
| `ndarray.tofile(fid[, sep, format])` | Write array to a file as text or binary (default). | [API][0277] |
| `ndarray.tolist()` | Return the array as a (possibly nested) list. | [API][0278] |

### Raw binary files

| API | Description | Link |
|-----|-------------|------|
| `fromfile(file[, dtype, count, sep])` | Construct an array from data in a text or binary file. | [API][0279] |
| `ndarray.tofile(fid[, sep, format])` | Write array to a file as text or binary (default). | [API][0280] |

### String formatting

| API | Description | Link |
|-----|-------------|------|
| `array2string(a[, max_line_width, precision, …])` | Return a string representation of an array. | [API][0281] |
| `array_repr(arr[, max_line_width, precision, …])` | Return the string representation of an array. | [API][0282] |
| `array_str(a[, max_line_width, precision, …])` | Return a string representation of the data in an array. | [API][0283] |
| `format_float_positional(x[, precision, …])` | Format a floating-point scalar as a decimal string in positional notation. | [API][0284] |
| `format_float_scientific(x[, precision, …])` | Format a floating-point scalar as a decimal string in scientific notation. | [API][0285] |

### Memory mapping files

| API | Description | Link |
|-----|-------------|------|
| `memmap` | Create a memory-map to an array stored in a binary file on disk. | [API][0286] |

### Text formatting options

| API | Description | Link |
|-----|-------------|------|
| `set_printoptions([precision, threshold, …])` | Set printing options. | [API][0287] |
| `get_printoptions()` | Return the current print options. | [API][0288] |
| `set_string_function(f[, repr])` | Set a Python function to be used when pretty printing arrays. | [API][0289] |

### Base-n representations

| API | Description | Link |
|-----|-------------|------|
| `binary_repr(num[, width])` | Return the binary representation of the input number as a string. | [API][0290] |
| `base_repr(number[, base, padding])` | Return a string representation of a number in the given base system. | [API][0291] |

### Data sources

| API | Description | Link |
|-----|-------------|------|
| `DataSource([destpath])` | A generic data source file (file, http, ftp, …). | [API][0292] |

### Binary Format Description

| API | Description | Link |
|-----|-------------|------|
| `lib.format` | Binary serialization | [API][0293] |


## [Linear algebra (numpy.linalg)][0294]

### Matrix and vector products

| API | Description | Link |
|-----|-------------|------|
| `dot(a, b[, out])` | Dot product of two arrays. | [API][0295] |
| `linalg.multi_dot(arrays)` | Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order. | [API][0296] |
| `vdot(a, b)` | Return the dot product of two vectors. | [API][0297] |
| `inner(a, b)` | Inner product of two arrays. | [API][0298] |
| `outer(a, b[, out])` | Compute the outer product of two vectors. | [API][0299] |
| `matmul(a, b[, out])` | Matrix product of two arrays. | [API][0300] |
| `tensordot(a, b[, axes])` | Compute tensor dot product along specified axes for arrays >= 1-D. | [API][0301] |
| `einsum(subscripts, *operands[, out, dtype, …])` | Evaluates the Einstein summation convention on the operands. | [API][0302] |
| `einsum_path(subscripts, *operands[, optimize])` | Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays. | [API][0303] |
| `linalg.matrix_power(a, n)` | Raise a square matrix to the (integer) power n. | [API][0304] |
| `kron(a, b)` | Kronecker product of two arrays. | [API][0305] |

### Decompositions

| API | Description | Link |
|-----|-------------|------|
| `linalg.cholesky(a)` | Cholesky decomposition. | [API][0306] |
| `linalg.qr(a[, mode])` | Compute the qr factorization of a matrix. | [API][0307] |
| `linalg.svd(a[, full_matrices, compute_uv])` | Singular Value Decomposition. | [API][0308] |

### Matrix eigenvalues

| API | Description | Link |
|-----|-------------|------|
| `linalg.eig(a)` | Compute the eigenvalues and right eigenvectors of a square array. | [API][0309] |
| `linalg.eigh(a[, UPLO])` | Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix. | [API][0310] |
| `linalg.eigvals(a)` | Compute the eigenvalues of a general matrix. | [API][0311] |
| `linalg.eigvalsh(a[, UPLO])` | Compute the eigenvalues of a Hermitian or real symmetric matrix. | [API][0312] |

### Norms and other numbers

| API | Description | Link |
|-----|-------------|------|
| `linalg.norm(x[, ord, axis, keepdims])` | Matrix or vector norm. | [API][0313] |
| `linalg.cond(x[, p])` | Compute the condition number of a matrix. | [API][0314] |
| `linalg.det(a)` | Compute the determinant of an array. | [API][0315] |
| `linalg.matrix_rank(M[, tol, hermitian])` | Return matrix rank of array using SVD method | [API][0316] |
| `linalg.slogdet(a)` | Compute the sign and (natural) logarithm of the determinant of an array. | [API][0317] |
| `trace(a[, offset, axis1, axis2, dtype, out])` | Return the sum along diagonals of the array. | [API][0318] |

### Solving equations and inverting matrices

| API | Description | Link |
|-----|-------------|------|
| `linalg.solve(a, b)` | Solve a linear matrix equation, or system of linear scalar equations. | [API][0319] |
| `linalg.tensorsolve(a, b[, axes])` | Solve the tensor equation a x = b for x. | [API][0320] |
| `linalg.lstsq(a, b[, rcond])` | Return the least-squares solution to a linear matrix equation. | [API][0321] |
| `linalg.inv(a)` | Compute the (multiplicative) inverse of a matrix. | [API][0322] |
| `linalg.pinv(a[, rcond])` | Compute the (Moore-Penrose) pseudo-inverse of a matrix. | [API][0323] |
| `linalg.tensorinv(a[, ind])` | Compute the ‘inverse’ of an N-dimensional array. | [API][0324] |

### Exceptions

| API | Description | Link |
|-----|-------------|------|
| `linalg.LinAlgError` | Generic Python-exception-derived object raised by linalg functions. | [API][0325] |


## [Logic functions][0326]

### Truth value testing

| API | Description | Link |
|-----|-------------|------|
| `all(a[, axis, out, keepdims])` | Test whether all array elements along a given axis evaluate to True. | [API][0327] |
| `any(a[, axis, out, keepdims])` | Test whether any array element along a given axis evaluates to True. | [API][0328] |

### Array contents

| API | Description | Link |
|-----|-------------|------|
| `isfinite(x, /[, out, where, casting, order, …])` | Test element-wise for finiteness (not infinity or not Not a Number). | [API][0329] |
| `isinf(x, /[, out, where, casting, order, …])` | Test element-wise for positive or negative infinity. | [API][0330] |
| `isnan(x, /[, out, where, casting, order, …])` | Test element-wise for NaN and return result as a boolean array. | [API][0331] |
| `isnat(x, /[, out, where, casting, order, …])` | Test element-wise for NaT (not a time) and return result as a boolean array. | [API][0332] |
| `isneginf(x[, out])` | Test element-wise for negative infinity, return result as bool array. | [API][0333] |
| `isposinf(x[, out])` | Test element-wise for positive infinity, return result as bool array. | [API][0334] |

### Array type testing

| API | Description | Link |
|-----|-------------|------|
| `iscomplex(x)` | Returns a bool array, where True if input element is complex. | [API][0335] |
| `iscomplexobj(x)` | Check for a complex type or an array of complex numbers. | [API][0336] |
| `isfortran(a)` | Returns True if the array is Fortran contiguous but not C contiguous. | [API][0337] |
| `isreal(x)` | Returns a bool array, where True if input element is real. | [API][0338] |
| `isrealobj(x)` | Return True if x is a not complex type or an array of complex numbers. | [API][0339] |
| `isscalar(num)` | Returns True if the type of num is a scalar type. | [API][0340] |

### Logical operations

| API | Description | Link |
|-----|-------------|------|
| `logical_and(x1, x2, /[, out, where, …])` | Compute the truth value of x1 AND x2 element-wise. | [API][0341] |
| `logical_or(x1, x2, /[, out, where, casting, …])` | Compute the truth value of x1 OR x2 element-wise. | [API][0342] |
| `logical_not(x, /[, out, where, casting, …])` | Compute the truth value of NOT x element-wise. | [API][0343] |
| `logical_xor(x1, x2, /[, out, where, …])` | Compute the truth value of x1 XOR x2, element-wise. | [API][0344] |

### Comparison

| API | Description | Link |
|-----|-------------|------|
| `allclose(a, b[, rtol, atol, equal_nan])` | Returns True if two arrays are element-wise equal within a tolerance. | [API][0345] |
| `isclose(a, b[, rtol, atol, equal_nan])` | Returns a boolean array where two arrays are element-wise equal within a tolerance. | [API][0346] |
| `array_equal(a1, a2)` | True if two arrays have the same shape and elements, False otherwise. | [API][0347] |
| `array_equiv(a1, a2)` | Returns True if input arrays are shape consistent and all elements equal. | [API][0348] |
| `greater(x1, x2, /[, out, where, casting, …])` | Return the truth value of (x1 > x2) element-wise. | [API][0349] |
| `greater_equal(x1, x2, /[, out, where, …])` | Return the truth value of (x1 >= x2) element-wise. | [API][0350] |
| `less(x1, x2, /[, out, where, casting, …])` | Return the truth value of (x1 < x2) element-wise. | [API][0351] |
| `less_equal(x1, x2, /[, out, where, casting, …])` | Return the truth value of (x1 =< x2) element-wise. | [API][0352] |
| `equal(x1, x2, /[, out, where, casting, …])` | Return (x1 == x2) element-wise. | [API][0353] |
| `not_equal(x1, x2, /[, out, where, casting, …])` | Return (x1 != x2) element-wise. | [API][0354] |


## [Masked array operations][0355]

### Constants

| API | Description | Link |
|-----|-------------|------|
| `ma.MaskType` | alias of numpy.bool_ | [API][0356] |

### Creation

#### From existing data

| API | Description | Link |
|-----|-------------|------|
| `ma.masked_array` | alias of numpy.ma.core.MaskedArray | [API][0357] |
| `ma.array(data[, dtype, copy, order, mask, …])` | An array class with possibly masked values. | [API][0358] |
| `ma.copy(self, *args, **params) a.copy(order=)` | Return a copy of the array. | [API][0359] |
| `ma.frombuffer(buffer[, dtype, count, offset])` | Interpret a buffer as a 1-dimensional array. | [API][0360] |
| `ma.fromfunction(function, shape, **kwargs)` | Construct an array by executing a function over each coordinate. | [API][0361] |
| `ma.MaskedArray.copy([order])` | Return a copy of the array. | [API][0362] |

#### Ones and zeros

| API | Description | Link |
|-----|-------------|------|
| `ma.empty(shape[, dtype, order])` | Return a new array of given shape and type, without initializing entries. | [API][0363] |
| `ma.empty_like(prototype[, dtype, order, subok])` | Return a new array with the same shape and type as a given array. | [API][0364] |
| `ma.masked_all(shape[, dtype])` | Empty masked array with all elements masked. | [API][0365] |
| `ma.masked_all_like(arr)` | Empty masked array with the properties of an existing array. | [API][0366] |
| `ma.ones(shape[, dtype, order])` | Return a new array of given shape and type, filled with ones. | [API][0367] |
| `ma.zeros(shape[, dtype, order])` | Return a new array of given shape and type, filled with zeros. | [API][0368] |

### Inspecting the array

| API | Description | Link |
|-----|-------------|------|
| `ma.all(self[, axis, out, keepdims])` | Returns True if all elements evaluate to True. | [API][0369] |
| `ma.any(self[, axis, out, keepdims])` | Returns True if any of the elements of a evaluate to True. | [API][0370] |
| `ma.count(self[, axis, keepdims])` | Count the non-masked elements of the array along the given axis. | [API][0371] |
| `ma.count_masked(arr[, axis])` | Count the number of masked elements along the given axis. | [API][0372] |
| `ma.getmask(a)` | Return the mask of a masked array, or nomask. | [API][0373] |
| `ma.getmaskarray(arr)` | Return the mask of a masked array, or full boolean array of False. | [API][0374] |
| `ma.getdata(a[, subok])` | Return the data of a masked array as an ndarray. | [API][0375] |
| `ma.nonzero(self)` | Return the indices of unmasked elements that are not zero. | [API][0376] |
| `ma.shape(obj)` | Return the shape of an array. | [API][0377] |
| `ma.size(obj[, axis])` | Return the number of elements along a given axis. | [API][0378] |
| `ma.is_masked(x)` | Determine whether input has masked values. | [API][0379] |
| `ma.is_mask(m)` | Return True if m is a valid, standard mask. | [API][0380] |
| `ma.MaskedArray.data` | Return the current data, as a view of the original underlying data. | [API][0381] |
| `ma.MaskedArray.mask` | Mask | [API][0382] |
| `ma.MaskedArray.recordmask` | Return the mask of the records. | [API][0383] |
| `ma.MaskedArray.all([axis, out, keepdims])` | Returns True if all elements evaluate to True. | [API][0384] |
| `ma.MaskedArray.any([axis, out, keepdims])` | Returns True if any of the elements of a evaluate to True. | [API][0385] |
| `ma.MaskedArray.count([axis, keepdims])` | Count the non-masked elements of the array along the given axis. | [API][0386] |
| `ma.MaskedArray.nonzero()` | Return the indices of unmasked elements that are not zero. | [API][0387] |
| `ma.shape(obj)` | Return the shape of an array. | [API][0388] |
| `ma.size(obj[, axis])` | Return the number of elements along a given axis. | [API][0389] |

### Manipulating a MaskedArray

#### Changing the shape

| API | Description | Link |
|-----|-------------|------|
| `ma.ravel(self[, order])` | Returns a 1D version of self, as a view. | [API][0390] |
| `ma.reshape(a, new_shape[, order])` | Returns an array containing the same data with a new shape. | [API][0391] |
| `ma.resize(x, new_shape)` | Return a new masked array with the specified size and shape. | [API][0392] |
| `ma.MaskedArray.flatten([order])` | Return a copy of the array collapsed into one dimension. | [API][0393] |
| `ma.MaskedArray.ravel([order])` | Returns a 1D version of self, as a view. | [API][0394] |
| `ma.MaskedArray.reshape(*s, **kwargs)` | Give a new shape to the array without changing its data. | [API][0395] |
| `ma.MaskedArray.resize(newshape[, refcheck, …])` |  | [API][0396] |

#### Modifying axes

| API | Description | Link |
|-----|-------------|------|
| `ma.swapaxes(self, *args, …)` | Return a view of the array with axis1 and axis2 interchanged. | [API][0397] |
| `ma.transpose(a[, axes])` | Permute the dimensions of an array. | [API][0398] |
| `ma.MaskedArray.swapaxes(axis1, axis2)` | Return a view of the array with axis1 and axis2 interchanged. | [API][0399] |
| `ma.MaskedArray.transpose(*axes)` | Returns a view of the array with axes transposed. | [API][0400] |

#### Changing the number of dimensions

| API | Description | Link |
|-----|-------------|------|
| `ma.atleast_1d(*arys)` | Convert inputs to arrays with at least one dimension. | [API][0401] |
| `ma.atleast_2d(*arys)` | View inputs as arrays with at least two dimensions. | [API][0402] |
| `ma.atleast_3d(*arys)` | View inputs as arrays with at least three dimensions. | [API][0403] |
| `ma.expand_dims(x, axis)` | Expand the shape of an array. | [API][0404] |
| `ma.squeeze(a[, axis])` | Remove single-dimensional entries from the shape of an array. | [API][0405] |
| `ma.MaskedArray.squeeze([axis])` | Remove single-dimensional entries from the shape of a. | [API][0406] |
| `ma.column_stack(tup)` | Stack 1-D arrays as columns into a 2-D array. | [API][0407] |
| `ma.concatenate(arrays[, axis])` | Concatenate a sequence of arrays along the given axis. | [API][0408] |
| `ma.dstack(tup)` | Stack arrays in sequence depth wise (along third axis). | [API][0409] |
| `ma.hstack(tup)` | Stack arrays in sequence horizontally (column wise). | [API][0410] |
| `ma.hsplit(ary, indices_or_sections)` | Split an array into multiple sub-arrays horizontally (column-wise). | [API][0411] |
| `ma.mr_` | Translate slice objects to concatenation along the first axis. | [API][0412] |
| `ma.row_stack(tup)` | Stack arrays in sequence vertically (row wise). | [API][0413] |
| `ma.vstack(tup)` | Stack arrays in sequence vertically (row wise). | [API][0414] |

#### Joining arrays

| API | Description | Link |
|-----|-------------|------|
| `ma.column_stack(tup)` | Stack 1-D arrays as columns into a 2-D array. | [API][0415] |
| `ma.concatenate(arrays[, axis])` | Concatenate a sequence of arrays along the given axis. | [API][0416] |
| `ma.append(a, b[, axis])` | Append values to the end of an array. | [API][0417] |
| `ma.dstack(tup)` | Stack arrays in sequence depth wise (along third axis). | [API][0418] |
| `ma.hstack(tup)` | Stack arrays in sequence horizontally (column wise). | [API][0419] |
| `ma.vstack(tup)` | Stack arrays in sequence vertically (row wise). | [API][0420] |

### Operations on masks

#### Creating a mask

| API | Description | Link |
|-----|-------------|------|
| `ma.make_mask(m[, copy, shrink, dtype])` | Create a boolean mask from an array. | [API][0421] |
| `ma.make_mask_none(newshape[, dtype])` | Return a boolean mask of the given shape, filled with False. | [API][0422] |
| `ma.mask_or(m1, m2[, copy, shrink])` | Combine two masks with the logical_or operator. | [API][0423] |
| `ma.make_mask_descr(ndtype)` | Construct a dtype description list from a given dtype. | [API][0424] |

#### Accessing a mask

| `ma.getmask(a)` | Return the mask of a masked array, or nomask. | [API][0425] |
| `ma.getmaskarray(arr)` | Return the mask of a masked array, or full boolean array of False. | [API][0426] |
| `ma.masked_array.mask` | Mask | [API][0427] |

#### Finding masked data

| API | Description | Link |
|-----|-------------|------|
| `ma.flatnotmasked_contiguous(a)` | Find contiguous unmasked data in a masked array along the given axis. | [API][0428] |
| `ma.flatnotmasked_edges(a)` | Find the indices of the first and last unmasked values. | [API][0429] |
| `ma.notmasked_contiguous(a[, axis])` | Find contiguous unmasked data in a masked array along the given axis. | [API][0430] |
| `ma.notmasked_edges(a[, axis])` | Find the indices of the first and last unmasked values along an axis. | [API][0431] |
| `ma.clump_masked(a)` | Returns a list of slices corresponding to the masked clumps of a 1-D array. | [API][0432] |
| `ma.clump_unmasked(a)` | Return list of slices corresponding to the unmasked clumps of a 1-D array. | [API][0433] |

#### Modifying a mask

| API | Description | Link |
|-----|-------------|------|
| `ma.mask_cols(a[, axis])` | Mask columns of a 2D array that contain masked values. | [API][0434] |
| `ma.mask_or(m1, m2[, copy, shrink])` | Combine two masks with the logical_or operator. | [API][0435] |
| `ma.mask_rowcols(a[, axis])` | Mask rows and/or columns of a 2D array that contain masked values. | [API][0436] |
| `ma.mask_rows(a[, axis])` | Mask rows of a 2D array that contain masked values. | [API][0437] |
| `ma.harden_mask(self)` | Force the mask to hard. | [API][0438] |
| `ma.soften_mask(self)` | Force the mask to soft. | [API][0439] |
| `ma.MaskedArray.harden_mask()` | Force the mask to hard. | [API][0440] |
| `ma.MaskedArray.soften_mask()` | Force the mask to soft. | [API][0441] |
| `ma.MaskedArray.shrink_mask()` | Reduce a mask to nomask when possible. | [API][0442] |
| `ma.MaskedArray.unshare_mask()` | Copy the mask and set the sharedmask flag to False. | [API][0443] |

### Conversion operations

#### `>` to a masked array

| API | Description | Link |
|-----|-------------|------|
| `ma.asarray(a[, dtype, order])` | Convert the input to a masked array of the given data-type. | [API][0444] |
| `ma.asanyarray(a[, dtype])` | Convert the input to a masked array, conserving subclasses. | [API][0445] |
| `ma.fix_invalid(a[, mask, copy, fill_value])` | Return input with invalid data masked and replaced by a fill value. | [API][0446] |
| `ma.masked_equal(x, value[, copy])` | Mask an array where equal to a given value. | [API][0447] |
| `ma.masked_greater(x, value[, copy])` | Mask an array where greater than a given value. | [API][0448] |
| `ma.masked_greater_equal(x, value[, copy])` | Mask an array where greater than or equal to a given value. | [API][0449] |
| `ma.masked_inside(x, v1, v2[, copy])` | Mask an array inside a given interval. | [API][0450] |
| `ma.masked_invalid(a[, copy])` | Mask an array where invalid values occur (NaNs or infs). | [API][0451] |
| `ma.masked_less(x, value[, copy])` | Mask an array where less than a given value. | [API][0452] |
| `ma.masked_less_equal(x, value[, copy])` | Mask an array where less than or equal to a given value. | [API][0453] |
| `ma.masked_not_equal(x, value[, copy])` | Mask an array where not equal to a given value. | [API][0454] |
| `ma.masked_object(x, value[, copy, shrink])` | Mask the array x where the data are exactly equal to value. | [API][0455] |
| `ma.masked_outside(x, v1, v2[, copy])` | Mask an array outside a given interval. | [API][0456] |
| `ma.masked_values(x, value[, rtol, atol, …])` | Mask using floating point equality. | [API][0457] |
| `ma.masked_where(condition, a[, copy])` | Mask an array where a condition is met. | [API][0584] |

#### `>` to a ndarray

| API | Description | Link |
|-----|-------------|------|
| `ma.compress_cols(a)` | Suppress whole columns of a 2-D array that contain masked values. | [API][0459] |
| `ma.compress_rowcols(x[, axis])` | Suppress the rows and/or columns of a 2-D array that contain masked values. | [API][0460] |
| `ma.compress_rows(a)` | Suppress whole rows of a 2-D array that contain masked values. | [API][0461] |
| `ma.compressed(x)` | Return all the non-masked data as a 1-D array. | [API][0462] |
| `ma.filled(a[, fill_value])` | Return input as an array with masked data replaced by a fill value. | [API][0463] |
| `ma.MaskedArray.compressed()` | Return all the non-masked data as a 1-D array. | [API][0464] |
| `ma.MaskedArray.filled([fill_value])` | Return a copy of self, with masked values filled with a given value. | [API][0465] |

#### `>` to another object

| API | Description | Link |
|-----|-------------|------|
| `ma.MaskedArray.tofile(fid[, sep, format])` | Save a masked array to a file in binary format. | [API][0466] |
| `ma.MaskedArray.tolist([fill_value])` | Return the data portion of the masked array as a hierarchical Python list. | [API][0467] |
| `ma.MaskedArray.torecords()` | Transforms a masked array into a flexible-type array. | [API][0468] |
| `ma.MaskedArray.tobytes([fill_value, order])` | Return the array data as a string containing the raw bytes in the array. | [API][0469] |

#### Pickling and unpickling

| API | Description | Link |
|-----|-------------|------|
| `ma.dump(a, F)` | Pickle a masked array to a file. | [API][0470] |
| `ma.dumps(a)` | Return a string corresponding to the pickling of a masked array. | [API][0471] |
| `ma.load(F)` | Wrapper around cPickle.load which accepts either a file-like object or a filename. | [API][0472] |
| `ma.loads(strg)` | Load a pickle from the current string. | [API][0473] |

#### Filling a masked array

| API | Description | Link |
|-----|-------------|------|
| `ma.common_fill_value(a, b)` | Return the common filling value of two masked arrays, if any. | [API][0474] |
| `ma.default_fill_value(obj)` | Return the default fill value for the argument object. | [API][0475] |
| `ma.maximum_fill_value(obj)` | Return the minimum value that can be represented by the dtype of an object. | [API][0476] |
| `ma.maximum_fill_value(obj)` | Return the minimum value that can be represented by the dtype of an object. | [API][0477] |
| `ma.set_fill_value(a, fill_value)` | Set the filling value of a, if a is a masked array. | [API][0478] |
| `ma.MaskedArray.get_fill_value()` | Return the filling value of the masked array. | [API][0479] |
| `ma.MaskedArray.set_fill_value([value])` | Set the filling value of the masked array. | [API][0480] |
| `ma.MaskedArray.fill_value` | Filling value. | [API][0481] |

### Masked arrays arithmetics

#### Arithmetics

| API | Description | Link |
|-----|-------------|------|
| `ma.anom(self[, axis, dtype])` | Compute the anomalies (deviations from the arithmetic mean) along the given axis. | [API][0482] |
| `ma.anomalies(self[, axis, dtype])` | Compute the anomalies (deviations from the arithmetic mean) along the given axis. | [API][0483] |
| `ma.average(a[, axis, weights, returned])` | Return the weighted average of array over the given axis. | [API][0484] |
| `ma.conjugate(x, /[, out, where, casting, …])` | Return the complex conjugate, element-wise. | [API][0485] |
| `ma.corrcoef(x[, y, rowvar, bias, …])` | Return Pearson product-moment correlation coefficients. | [API][0486] |
| `ma.cov(x[, y, rowvar, bias, allow_masked, ddof])` | Estimate the covariance matrix. | [API][0487] |
| `ma.cumsum(self[, axis, dtype, out])` | Return the cumulative sum of the array elements over the given axis. | [API][0488] |
| `ma.cumprod(self[, axis, dtype, out])` | Return the cumulative product of the array elements over the given axis. | [API][0489] |
| `ma.mean(self[, axis, dtype, out, keepdims])` | Returns the average of the array elements along given axis. | [API][0490] |
| `ma.median(a[, axis, out, overwrite_input, …])` | Compute the median along the specified axis. | [API][0491] |
| `ma.power(a, b[, third])` | Returns element-wise base array raised to power from second array. | [API][0492] |
| `ma.prod(self[, axis, dtype, out, keepdims])` | Return the product of the array elements over the given axis. | [API][0493] |
| `ma.std(self[, axis, dtype, out, ddof, keepdims])` | Returns the standard deviation of the array elements along given axis. | [API][0494] |
| `ma.sum(self[, axis, dtype, out, keepdims])` | Return the sum of the array elements over the given axis. | [API][0495] |
| `ma.var(self[, axis, dtype, out, ddof, keepdims])` | Compute the variance along the specified axis. | [API][0496] |
| `ma.MaskedArray.anom([axis, dtype])` | Compute the anomalies (deviations from the arithmetic mean) along the given axis. | [API][0497] |
| `ma.MaskedArray.cumprod([axis, dtype, out])` | Return the cumulative product of the array elements over the given axis. | [API][0498] |
| `ma.MaskedArray.cumsum([axis, dtype, out])` | Return the cumulative sum of the array elements over the given axis. | [API][0499] |
| `ma.MaskedArray.mean([axis, dtype, out, keepdims])` | Returns the average of the array elements along given axis. | [API][0500] |
| `ma.MaskedArray.prod([axis, dtype, out, keepdims])` | Return the product of the array elements over the given axis. | [API][0501] |
| `ma.MaskedArray.std([axis, dtype, out, ddof, …])` | Returns the standard deviation of the array elements along given axis. | [API][0502] |
| `ma.MaskedArray.sum([axis, dtype, out, keepdims])` | Return the sum of the array elements over the given axis. | [API][0503] |
| `ma.MaskedArray.var([axis, dtype, out, ddof, …])` | Compute the variance along the specified axis. | [API][0504] |

#### Minimum/maximum

| API | Description | Link |
|-----|-------------|------|
| `ma.argmax(self[, axis, fill_value, out])` | Returns array of indices of the maximum values along the given axis. | [API][0505] |
| `ma.argmin(self[, axis, fill_value, out])` | Return array of indices to the minimum values along the given axis. | [API][0506] |
| `ma.max(obj[, axis, out, fill_value, keepdims])` | Return the maximum along a given axis. | [API][0507] |
| `ma.min(obj[, axis, out, fill_value, keepdims])` | Return the minimum along a given axis. | [API][0508] |
| `ma.ptp(obj[, axis, out, fill_value, keepdims])` | Return (maximum - minimum) along the given dimension (i.e. | [API][0509] |
| `ma.MaskedArray.argmax([axis, fill_value, out])` | Returns array of indices of the maximum values along the given axis. | [API][0510] |
| `ma.MaskedArray.argmin([axis, fill_value, out])` | Return array of indices to the minimum values along the given axis. | [API][0511] |
| `ma.MaskedArray.max([axis, out, fill_value, …])` | Return the maximum along a given axis. | [API][0512] |
| `ma.MaskedArray.min([axis, out, fill_value, …])` | Return the minimum along a given axis. | [API][0513] |
| `ma.MaskedArray.ptp([axis, out, fill_value, …])` | Return (maximum - minimum) along the given dimension (i.e. | [API][0514] |

#### Sorting

| API | Description | Link |
|-----|-------------|------|
| `ma.argsort(a[, axis, kind, order, endwith, …])` | Return an ndarray of indices that sort the array along the specified axis. | [API][0515] |
| `ma.sort(a[, axis, kind, order, endwith, …])` | Sort the array, in-place | [API][0516] |
| `ma.MaskedArray.argsort([axis, kind, order, …])` | Return an ndarray of indices that sort the array along the specified axis. | [API][0517] |
| `ma.MaskedArray.sort([axis, kind, order, …])` | Sort the array, in-place | [API][0518] |

#### Algebra

| API | Description | Link |
|-----|-------------|------|
| `ma.diag(v[, k])` | Extract a diagonal or construct a diagonal array. | [API][0519] |
| `ma.dot(a, b[, strict, out])` | Return the dot product of two arrays. | [API][0520] |
| `ma.identity(n[, dtype])` | Return the identity array. | [API][0521] |
| `ma.inner(a, b)` | Inner product of two arrays. | [API][0522] |
| `ma.innerproduct(a, b)` | Inner product of two arrays. | [API][0523] |
| `ma.outer(a, b)` | Compute the outer product of two vectors. | [API][0524] |
| `ma.outerproduct(a, b)` | Compute the outer product of two vectors. | [API][0525] |
| `ma.trace(self[, offset, axis1, axis2, …])` | Return the sum along diagonals of the array. | [API][0526] |
| `ma.transpose(a[, axes])` | Permute the dimensions of an array. | [API][0527] |
| `ma.MaskedArray.trace([offset, axis1, axis2, …])` | Return the sum along diagonals of the array. | [API][0528] |
| `ma.MaskedArray.transpose(*axes)` | Returns a view of the array with axes transposed. | [API][0529] |

#### Polynomial fit

| API | Description | Link |
|-----|-------------|------|
| `ma.vander(x[, n])` | Generate a Vandermonde matrix. | [API][0530] |
| `ma.polyfit(x, y, deg[, rcond, full, w, cov])` | Least squares polynomial fit. | [API][0531] |

#### Clipping and rounding

| API | Description | Link |
|-----|-------------|------|
| `ma.around` | Round an array to the given number of decimals. | [API][0532] |
| `ma.clip(a, a_min, a_max[, out])` | Clip (limit) the values in an array. | [API][0533] |
| `ma.round(a[, decimals, out])` | Return a copy of a, rounded to ‘decimals’ places. | [API][0534] |
| `ma.MaskedArray.clip([min, max, out])` | Return an array whose values are limited to [min, max]. | [API][0535] |
| `ma.MaskedArray.round([decimals, out])` | Return each element rounded to the given number of decimals. | [API][0536] |

#### Miscellanea

| API | Description | Link |
|-----|-------------|------|
| `ma.allequal(a, b[, fill_value])` | Return True if all entries of a and b are equal, using fill_value as a truth value where either or both are masked. | [API][0537] |
| `ma.allclose(a, b[, masked_equal, rtol, atol])` | Returns True if two arrays are element-wise equal within a tolerance. | [API][0538] |
| `ma.apply_along_axis(func1d, axis, arr, …)` | Apply a function to 1-D slices along the given axis. | [API][0539] |
| `ma.arange([start,] stop[, step,][, dtype])` | Return evenly spaced values within a given interval. | [API][0540] |
| `ma.choose(indices, choices[, out, mode])` | Use an index array to construct a new array from a set of choices. | [API][0541] |
| `ma.ediff1d(arr[, to_end, to_begin])` | Compute the differences between consecutive elements of an array. | [API][0542] |
| `ma.indices(dimensions[, dtype])` | Return an array representing the indices of a grid. | [API][0543] |
| `ma.where(condition[, x, y])` | Return a masked array with elements from x or y, depending on condition. | [API][0544] |


## [Mathematical functions][0545]

### Trigonometric functions

| API | Description | Link |
|-----|-------------|------|
| `sin(x, /[, out, where, casting, order, …])` | Trigonometric sine, element-wise. | [API][0546] |
| `cos(x, /[, out, where, casting, order, …])` | Cosine element-wise. | [API][0547] |
| `tan(x, /[, out, where, casting, order, …])` | Compute tangent element-wise. | [API][0548] |
| `arcsin(x, /[, out, where, casting, order, …])` | Inverse sine, element-wise. | [API][0549] |
| `arccos(x, /[, out, where, casting, order, …])` | Trigonometric inverse cosine, element-wise. | [API][0550] |
| `arctan(x, /[, out, where, casting, order, …])` | Trigonometric inverse tangent, element-wise. | [API][0551] |
| `hypot(x1, x2, /[, out, where, casting, …])` | Given the “legs” of a right triangle, return its hypotenuse. | [API][0552] |
| `arctan2(x1, x2, /[, out, where, casting, …])` | Element-wise arc tangent of x1/x2 choosing the quadrant correctly. | [API][0553] |
| `degrees(x, /[, out, where, casting, order, …])` | Convert angles from radians to degrees. | [API][0554] |
| `radians(x, /[, out, where, casting, order, …])` | Convert angles from degrees to radians. | [API][0555] |
| `unwrap(p[, discont, axis])` | Unwrap by changing deltas between values to 2*pi complement. | [API][0556] |
| `deg2rad(x, /[, out, where, casting, order, …])` | Convert angles from degrees to radians. | [API][0557] |
| `rad2deg(x, /[, out, where, casting, order, …])` | Convert angles from radians to degrees. | [API][0558] |

### Hyperbolic functions

| API | Description | Link |
|-----|-------------|------|
| `sinh(x, /[, out, where, casting, order, …])` | Hyperbolic sine, element-wise. | [API][0559] |
| `cosh(x, /[, out, where, casting, order, …])` | Hyperbolic cosine, element-wise. | [API][0560] |
| `tanh(x, /[, out, where, casting, order, …])` | Compute hyperbolic tangent element-wise. | [API][0561] |
| `arcsinh(x, /[, out, where, casting, order, …])` | Inverse hyperbolic sine element-wise. | [API][0562] |
| `arccosh(x, /[, out, where, casting, order, …])` | Inverse hyperbolic cosine, element-wise. | [API][0563] |
| `arctanh(x, /[, out, where, casting, order, …])` | Inverse hyperbolic tangent element-wise. | [API][0564] |

### Rounding

| API | Description | Link |
|-----|-------------|------|
| `around(a[, decimals, out])` | Evenly round to the given number of decimals. | [API][0565] |
| `round_(a[, decimals, out])` | Round an array to the given number of decimals. | [API][0566] |
| `rint(x, /[, out, where, casting, order, …])` | Round elements of the array to the nearest integer. | [API][0567] |
| `fix(x[, out])` | Round to nearest integer towards zero. | [API][0568] |
| `floor(x, /[, out, where, casting, order, …])` | Return the floor of the input, element-wise. | [API][0569] |
| `ceil(x, /[, out, where, casting, order, …])` | Return the ceiling of the input, element-wise. | [API][0570] |
| `trunc(x, /[, out, where, casting, order, …])` | Return the truncated value of the input, element-wise. | [API][0571] |

### Sums, products, differences

| API | Description | Link |
|-----|-------------|------|
| `prod(a[, axis, dtype, out, keepdims, initial])` | Return the product of array elements over a given axis. | [API][0572] |
| `sum(a[, axis, dtype, out, keepdims, initial])` | Sum of array elements over a given axis. | [API][0573] |
| `nanprod(a[, axis, dtype, out, keepdims])` | Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones. | [API][0574] |
| `nansum(a[, axis, dtype, out, keepdims])` | Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero. | [API][0575] |
| `cumprod(a[, axis, dtype, out])` | Return the cumulative product of elements along a given axis. | [API][0576] |
| `cumsum(a[, axis, dtype, out])` | Return the cumulative sum of the elements along a given axis. | [API][0577] |
| `nancumprod(a[, axis, dtype, out])` | Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one. | [API][0578] |
| `nancumsum(a[, axis, dtype, out])` | Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs) as zero. | [API][0579] |
| `diff(a[, n, axis])` | Calculate the n-th discrete difference along the given axis. | [API][0580] |
| `ediff1d(ary[, to_end, to_begin])` | The differences between consecutive elements of an array. | [API][0581] |
| `gradient(f, *varargs, **kwargs)` | Return the gradient of an N-dimensional array. | [API][0582] |
| `cross(a, b[, axisa, axisb, axisc, axis])` | Return the cross product of two (arrays of) vectors. | [API][0583] |
| `trapz(y[, x, dx, axis])` | Integrate along the given axis using the composite trapezoidal rule. | [API][0584] |

### Exponents and logarithms

| API | Description | Link |
|-----|-------------|------|
| `exp(x, /[, out, where, casting, order, …])` | Calculate the exponential of all elements in the input array. | [API][0585] |
| `expm1(x, /[, out, where, casting, order, …])` | Calculate exp(x) - 1 for all elements in the array. | [API][0586] |
| `exp2(x, /[, out, where, casting, order, …])` | Calculate 2**p for all p in the input array. | [API][0587] |
| `log(x, /[, out, where, casting, order, …])` | Natural logarithm, element-wise. | [API][0588] |
| `log10(x, /[, out, where, casting, order, …])` | Return the base 10 logarithm of the input array, element-wise. | [API][0589] |
| `log2(x, /[, out, where, casting, order, …])` | Base-2 logarithm of x. | [API][0590] |
| `log1p(x, /[, out, where, casting, order, …])` | Return the natural logarithm of one plus the input array, element-wise. | [API][0591] |
| `logaddexp(x1, x2, /[, out, where, casting, …])` | Logarithm of the sum of exponentiations of the inputs. | [API][0592] |
| `logaddexp2(x1, x2, /[, out, where, casting, …])` | Logarithm of the sum of exponentiations of the inputs in base-2. | [API][0593] |

### Other special functions

| API | Description | Link |
|-----|-------------|------|
| `i0(x)` | Modified Bessel function of the first kind, order 0. | [API][0594] |
| `sinc(x)` | Return the sinc function. | [API][0595] |

### Floating point routines

| API | Description | Link |
|-----|-------------|------|
| `signbit(x, /[, out, where, casting, order, …])` | Returns element-wise True where signbit is set (less than zero). | [API][0596] |
| `copysign(x1, x2, /[, out, where, casting, …])` | Change the sign of x1 to that of x2, element-wise. | [API][0597] |
| `frexp(x[, out1, out2], / [[, out, where, …])` | Decompose the elements of x into mantissa and twos exponent. | [API][0598] |
| `ldexp(x1, x2, /[, out, where, casting, …])` | Returns x1 * 2**x2, element-wise. | [API][0599] |
| `nextafter(x1, x2, /[, out, where, casting, …])` | Return the next floating-point value after x1 towards x2, element-wise. | [API][0600] |
| `spacing(x, /[, out, where, casting, order, …])` | Return the distance between x and the nearest adjacent number. | [API][0601] |

### Rational routines

| API | Description | Link |
|-----|-------------|------|
| `lcm(x1, x2, /[, out, where, casting, order, …])` | Returns the lowest common multiple of |x1| and |x2| | [API][0602] |
| `gcd(x1, x2, /[, out, where, casting, order, …])` | Returns the greatest common divisor of |x1| and |x2| | [API][0603] |

### Arithmetic operations

| API | Description | Link |
|-----|-------------|------|
| `add(x1, x2, /[, out, where, casting, order, …])` | Add arguments element-wise. | [API][0604] |
| `reciprocal(x, /[, out, where, casting, …])` | Return the reciprocal of the argument, element-wise. | [API][0605] |
| `positive(x, /[, out, where, casting, order, …])` | Numerical positive, element-wise. | [API][0606] |
| `negative(x, /[, out, where, casting, order, …])` | Numerical negative, element-wise. | [API][0607] |
| `multiply(x1, x2, /[, out, where, casting, …])` | Multiply arguments element-wise. | [API][0608] |
| `divide(x1, x2, /[, out, where, casting, …])` | Returns a true division of the inputs, element-wise. | [API][0609] |
| `power(x1, x2, /[, out, where, casting, …])` | First array elements raised to powers from second array, element-wise. | [API][0610] |
| `subtract(x1, x2, /[, out, where, casting, …])` | Subtract arguments, element-wise. | [API][0611] |
| `true_divide(x1, x2, /[, out, where, …])` | Returns a true division of the inputs, element-wise. | [API][0612] |
| `floor_divide(x1, x2, /[, out, where, …])` | Return the largest integer smaller or equal to the division of the inputs. | [API][0613] |
| `float_power(x1, x2, /[, out, where, …])` | First array elements raised to powers from second array, element-wise. | [API][0614] |
| `fmod(x1, x2, /[, out, where, casting, …])` | Return the element-wise remainder of division. | [API][0615] |
| `mod(x1, x2, /[, out, where, casting, order, …])` | Return element-wise remainder of division. | [API][0616] |
| `modf(x[, out1, out2], / [[, out, where, …])` | Return the fractional and integral parts of an array, element-wise. | [API][0617] |
| `remainder(x1, x2, /[, out, where, casting, …])` | Return element-wise remainder of division. | [API][0618] |
| `divmod(x1, x2[, out1, out2], / [[, out, …])` | Return element-wise quotient and remainder simultaneously. | [API][0619] |

### Handling complex numbers

| API | Description | Link |
|-----|-------------|------|
| `angle(z[, deg])` | Return the angle of the complex argument. | [API][0620] |
| `real(val)` | Return the real part of the complex argument. | [API][0621] |
| `imag(val)` | Return the imaginary part of the complex argument. | [API][0622] |
| `conj(x, /[, out, where, casting, order, …])` | Return the complex conjugate, element-wise. | [API][0623] |

### Miscellaneous

| API | Description | Link |
|-----|-------------|------|
| `convolve(a, v[, mode])` | Returns the discrete, linear convolution of two one-dimensional sequences. | [API][0624] |
| `clip(a, a_min, a_max[, out])` | Clip (limit) the values in an array. | [API][0625] |
| `sqrt(x, /[, out, where, casting, order, …])` | Return the positive square-root of an array, element-wise. | [API][0626] |
| `cbrt(x, /[, out, where, casting, order, …])` | Return the cube-root of an array, element-wise. | [API][0627] |
| `square(x, /[, out, where, casting, order, …])` | Return the element-wise square of the input. | [API][0628] |
| `absolute(x, /[, out, where, casting, order, …])` | Calculate the absolute value element-wise. | [API][0629] |
| `fabs(x, /[, out, where, casting, order, …])` | Compute the absolute values element-wise. | [API][0630] |
| `sign(x, /[, out, where, casting, order, …])` | Returns an element-wise indication of the sign of a number. | [API][0631] |
| `heaviside(x1, x2, /[, out, where, casting, …])` | Compute the Heaviside step function. | [API][0632] |
| `maximum(x1, x2, /[, out, where, casting, …])` | Element-wise maximum of array elements. | [API][0633] |
| `minimum(x1, x2, /[, out, where, casting, …])` | Element-wise minimum of array elements. | [API][0634] |
| `fmax(x1, x2, /[, out, where, casting, …])` | Element-wise maximum of array elements. | [API][0635] |
| `fmin(x1, x2, /[, out, where, casting, …])` | Element-wise minimum of array elements. | [API][0636] |
| `nan_to_num(x[, copy])` | Replace NaN with zero and infinity with large finite numbers. | [API][0637] |
| `real_if_close(a[, tol])` | If complex input returns a real array if complex parts are close to zero. | [API][0638] |
| `interp(x, xp, fp[, left, right, period])` | One-dimensional linear interpolation. | [API][0639] |


## [Matrix library (numpy.matlib)][0640]

This module contains all functions in the [numpy](https://www.numpy.org/devdocs/reference/index.html#module-numpy) namespace, with the following replacement functions that return [matrices](https://www.numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix) instead of [ndarrays](https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray).

Functions that are also in the numpy namespace and return `matrices`

| API | Description | Link |
|-----|-------------|------|
| `mat(data[, dtype])` | Interpret the input as a matrix. | [API][0641] |
| `matrix(data[, dtype, copy])` | Returns a matrix from an array-like object, or from a string of data. | [API][0642] |
| `asmatrix(data[, dtype])` | Interpret the input as a matrix. | [API][0643] |
| `bmat(obj[, ldict, gdict])` | Build a matrix object from a string, nested sequence, or array. | [API][0644] |

### Replacement functions in matlib

| API | Description | Link |
|-----|-------------|------|
| `empty(shape[, dtype, order])` | Return a new matrix of given shape and type, without initializing entries. | [API][0645] |
| `zeros(shape[, dtype, order])` | Return a matrix of given shape and type, filled with zeros. | [API][0646] |
| `ones(shape[, dtype, order])` | Matrix of ones. | [API][0647] |
| `eye(n[, M, k, dtype, order])` | Return a matrix with ones on the diagonal and zeros elsewhere. | [API][0648] |
| `identity(n[, dtype])` | Returns the square identity matrix of given size. | [API][0649] |
| `repmat(a, m, n)` | Repeat a 0-D to 2-D array or matrix MxN times. | [API][0650] |
| `rand(*args)` | Return a matrix of random values with given shape. | [API][0651] |
| `randn(*args)` | Return a random matrix with data from the “standard normal” distribution. | [API][0652] |


## [Miscellaneous routines][0653]

### Buffer objects

+ getbuffer
+ newbuffer

### Performance tuning

| API | Description | Link |
|-----|-------------|------|
| `setbufsize(size)` | Set the size of the buffer used in ufuncs. | [API][0654] |
| `getbufsize()` | Return the size of the buffer used in ufuncs. | [API][0655] |

### Memory ranges

| API | Description | Link |
|-----|-------------|------|
| `shares_memory(a, b[, max_work])` | Determine if two arrays share memory | [API][0656] |
| `may_share_memory(a, b[, max_work])` | Determine if two arrays might share memory | [API][0657] |

### Array mixins

| API | Description | Link |
|-----|-------------|------|
| `lib.mixins.NDArrayOperatorsMixin` | Mixin defining all operator special methods using __array_ufunc__. | [API][0658] |

### NumPy version comparison

| API | Description | Link |
|-----|-------------|------|
| `lib.NumpyVersion(vstring)` | Parse and compare numpy version strings. | [API][0659] |









--------------------------------------------

[0000]: https://www.numpy.org/devdocs/reference/routines.array-creation.html
[0001]: https://www.numpy.org/devdocs/reference/generated/numpy.empty.html#numpy.empty
[0002]: https://www.numpy.org/devdocs/reference/generated/numpy.empty_like.html#numpy.empty_like
[0003]: https://www.numpy.org/devdocs/reference/generated/numpy.eye.html#numpy.eye
[0004]: https://www.numpy.org/devdocs/reference/generated/numpy.identity.html#numpy.identity
[0005]: https://www.numpy.org/devdocs/reference/generated/numpy.ones.html#numpy.ones
[0006]: https://www.numpy.org/devdocs/reference/generated/numpy.ones_like.html#numpy.ones_like
[0007]: https://www.numpy.org/devdocs/reference/generated/numpy.zeros.html#numpy.zeros
[0008]: https://www.numpy.org/devdocs/reference/generated/numpy.zeros_like.html#numpy.zeros_like
[0009]: https://www.numpy.org/devdocs/reference/generated/numpy.full.html#numpy.full
[0010]: https://www.numpy.org/devdocs/reference/generated/numpy.full_like.html#numpy.full_like
[0011]: https://www.numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array
[0012]: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray
[0013]: https://www.numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray
[0014]: https://www.numpy.org/devdocs/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray
[0015]: https://www.numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix
[0016]: https://www.numpy.org/devdocs/reference/generated/numpy.copy.html#numpy.copy
[0017]: https://www.numpy.org/devdocs/reference/generated/numpy.frombuffer.html#numpy.frombuffer
[0018]: https://www.numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile
[0019]: https://www.numpy.org/devdocs/reference/generated/numpy.fromfunction.html#numpy.fromfunction
[0020]: https://www.numpy.org/devdocs/reference/generated/numpy.fromiter.html#numpy.fromiter
[0021]: https://www.numpy.org/devdocs/reference/generated/numpy.fromstring.html#numpy.fromstring
[0022]: https://www.numpy.org/devdocs/reference/generated/numpy.loadtxt.html#numpy.loadtxt
[0023]: https://www.numpy.org/devdocs/reference/generated/numpy.core.records.array.html#numpy.core.records.array
[0024]: https://www.numpy.org/devdocs/reference/generated/numpy.core.records.fromarrays.html#numpy.core.records.fromarrays
[0025]: https://www.numpy.org/devdocs/reference/generated/numpy.core.records.fromrecords.html#numpy.core.records.fromrecords
[0026]: https://www.numpy.org/devdocs/reference/generated/numpy.core.records.fromstring.html#numpy.core.records.fromstring
[0027]: https://www.numpy.org/devdocs/reference/generated/numpy.core.records.fromfile.html#numpy.core.records.fromfile
[0028]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array
[0029]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.asarray.html#numpy.core.defchararray.asarray
[0030]: https://www.numpy.org/devdocs/reference/generated/numpy.arange.html#numpy.arange
[0031]: https://www.numpy.org/devdocs/reference/generated/numpy.linspace.html#numpy.linspace
[0032]: https://www.numpy.org/devdocs/reference/generated/numpy.logspace.html#numpy.logspace
[0033]: https://www.numpy.org/devdocs/reference/generated/numpy.geomspace.html#numpy.geomspace
[0034]: https://www.numpy.org/devdocs/reference/generated/numpy.meshgrid.html#numpy.meshgrid
[0035]: https://www.numpy.org/devdocs/reference/generated/numpy.mgrid.html#numpy.mgrid
[0036]: https://www.numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid
[0037]: https://www.numpy.org/devdocs/reference/generated/numpy.diag.html#numpy.diag
[0038]: https://www.numpy.org/devdocs/reference/generated/numpy.diagflat.html#numpy.diagflat
[0039]: https://www.numpy.org/devdocs/reference/generated/numpy.tri.html#numpy.tri
[0040]: https://www.numpy.org/devdocs/reference/generated/numpy.tril.html#numpy.tril
[0041]: https://www.numpy.org/devdocs/reference/generated/numpy.triu.html#numpy.triu
[0042]: https://www.numpy.org/devdocs/reference/generated/numpy.vander.html#numpy.vander
[0043]: https://www.numpy.org/devdocs/reference/generated/numpy.mat.html#numpy.mat
[0044]: https://www.numpy.org/devdocs/reference/generated/numpy.bmat.html#numpy.bmat
[0045]: https://www.numpy.org/devdocs/reference/routines.array-manipulation.html
[0046]: https://www.numpy.org/devdocs/reference/generated/numpy.copyto.html#numpy.copyto
[0047]: https://www.numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape
[0048]: https://www.numpy.org/devdocs/reference/generated/numpy.ravel.html#numpy.ravel
[0049]: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat
[0050]: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten
[0051]: https://www.numpy.org/devdocs/reference/generated/numpy.moveaxis.html#numpy.moveaxis
[0052]: https://www.numpy.org/devdocs/reference/generated/numpy.rollaxis.html#numpy.rollaxis
[0053]: https://www.numpy.org/devdocs/reference/generated/numpy.swapaxes.html#numpy.swapaxes
[0054]: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T
[0055]: https://www.numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose
[0056]: https://www.numpy.org/devdocs/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d
[0057]: https://www.numpy.org/devdocs/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d
[0058]: https://www.numpy.org/devdocs/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d
[0059]: https://www.numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast
[0060]: https://www.numpy.org/devdocs/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to
[0061]: https://www.numpy.org/devdocs/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays
[0062]: https://www.numpy.org/devdocs/reference/generated/numpy.expand_dims.html#numpy.expand_dims
[0063]: https://www.numpy.org/devdocs/reference/generated/numpy.squeeze.html#numpy.squeeze
[0064]: https://www.numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray
[0065]: https://www.numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray
[0066]: https://www.numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix
[0067]: https://www.numpy.org/devdocs/reference/generated/numpy.asfarray.html#numpy.asfarray
[0068]: https://www.numpy.org/devdocs/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray
[0069]: https://www.numpy.org/devdocs/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray
[0070]: https://www.numpy.org/devdocs/reference/generated/numpy.asarray_chkfinite.html#numpy.asarray_chkfinite
[0071]: https://www.numpy.org/devdocs/reference/generated/numpy.asscalar.html#numpy.asscalar
[0072]: https://www.numpy.org/devdocs/reference/generated/numpy.require.html#numpy.require
[0073]: https://www.numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate
[0074]: https://www.numpy.org/devdocs/reference/generated/numpy.stack.html#numpy.stack
[0075]: https://www.numpy.org/devdocs/reference/generated/numpy.dstack.html#numpy.dstack
[0076]: https://www.numpy.org/devdocs/reference/generated/numpy.dstack.html#numpy.dstack
[0077]: https://www.numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack
[0078]: https://www.numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack
[0079]: https://www.numpy.org/devdocs/reference/generated/numpy.block.html#numpy.block
[0080]: https://www.numpy.org/devdocs/reference/generated/numpy.split.html#numpy.split
[0081]: https://www.numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split
[0082]: https://www.numpy.org/devdocs/reference/generated/numpy.dsplit.html#numpy.dsplit
[0083]: https://www.numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit
[0084]: https://www.numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit
[0085]: https://www.numpy.org/devdocs/reference/generated/numpy.tile.html#numpy.tile
[0086]: https://www.numpy.org/devdocs/reference/generated/numpy.repeat.html#numpy.repeat
[0087]: https://www.numpy.org/devdocs/reference/generated/numpy.delete.html#numpy.delete
[0088]: https://www.numpy.org/devdocs/reference/generated/numpy.insert.html#numpy.insert
[0089]: https://www.numpy.org/devdocs/reference/generated/numpy.append.html#numpy.append
[0090]: https://www.numpy.org/devdocs/reference/generated/numpy.resize.html#numpy.resize
[0091]: https://www.numpy.org/devdocs/reference/generated/numpy.trim_zeros.html#numpy.trim_zeros
[0092]: https://www.numpy.org/devdocs/reference/generated/numpy.unique.html#numpy.unique
[0093]: https://www.numpy.org/devdocs/reference/generated/numpy.flip.html#numpy.flip
[0094]: https://www.numpy.org/devdocs/reference/generated/numpy.fliplr.html#numpy.fliplr
[0095]: https://www.numpy.org/devdocs/reference/generated/numpy.flipud.html#numpy.flipud
[0096]: https://www.numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape
[0097]: https://www.numpy.org/devdocs/reference/generated/numpy.roll.html#numpy.roll
[0098]: https://www.numpy.org/devdocs/reference/generated/numpy.rot90.html#numpy.rot90
[0099]: https://www.numpy.org/devdocs/reference/routines.bitwise.html
[0100]: https://www.numpy.org/devdocs/reference/generated/numpy.bitwise_and.html#numpy.bitwise_and
[0101]: https://www.numpy.org/devdocs/reference/generated/numpy.bitwise_or.html#numpy.bitwise_or
[0102]: https://www.numpy.org/devdocs/reference/generated/numpy.bitwise_xor.html#numpy.bitwise_xor
[0103]: https://www.numpy.org/devdocs/reference/generated/numpy.invert.html#numpy.invert
[0104]: https://www.numpy.org/devdocs/reference/generated/numpy.left_shift.html#numpy.left_shift
[0105]: https://www.numpy.org/devdocs/reference/generated/numpy.right_shift.html#numpy.right_shift
[0106]: https://www.numpy.org/devdocs/reference/generated/numpy.packbits.html#numpy.packbits
[0107]: https://www.numpy.org/devdocs/reference/generated/numpy.unpackbits.html#numpy.unpackbits
[0108]: https://www.numpy.org/devdocs/reference/generated/numpy.binary_repr.html#numpy.binary_repr
[0109]: https://www.numpy.org/devdocs/reference/routines.char.html
[0110]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.add.html#numpy.core.defchararray.add
[0111]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.multiply.html#numpy.core.defchararray.multiply
[0112]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.mod.html#numpy.core.defchararray.mod
[0113]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.capitalize.html#numpy.core.defchararray.capitalize
[0114]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.center.html#numpy.core.defchararray.center
[0115]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.decode.html#numpy.core.defchararray.decode
[0116]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.encode.html#numpy.core.defchararray.encode
[0117]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.join.html#numpy.core.defchararray.join
[0118]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.ljust.html#numpy.core.defchararray.ljust
[0119]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.lower.html#numpy.core.defchararray.lower
[0120]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.lstrip.html#numpy.core.defchararray.lstrip
[0121]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.partition.html#numpy.core.defchararray.partition
[0122]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.replace.html#numpy.core.defchararray.replace
[0123]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.rjust.html#numpy.core.defchararray.rjust
[0124]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.rpartition.html#numpy.core.defchararray.rpartition
[0125]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.rsplit.html#numpy.core.defchararray.rsplit
[0126]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.rstrip.html#numpy.core.defchararray.rstrip
[0127]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.split.html#numpy.core.defchararray.split
[0128]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.splitlines.html#numpy.core.defchararray.splitlines
[0129]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.strip.html#numpy.core.defchararray.strip
[0130]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.swapcase.html#numpy.core.defchararray.swapcase
[0131]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.title.html#numpy.core.defchararray.title
[0132]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.translate.html#numpy.core.defchararray.translate
[0133]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.upper.html#numpy.core.defchararray.upper
[0134]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.zfill.html#numpy.core.defchararray.zfill
[0135]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.equal.html#numpy.core.defchararray.equal
[0136]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.not_equal.html#numpy.core.defchararray.not_equal
[0137]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.greater_equal.html#numpy.core.defchararray.greater_equal
[0138]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.less_equal.html#numpy.core.defchararray.less_equal
[0139]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.greater.html#numpy.core.defchararray.greater
[0140]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.less.html#numpy.core.defchararray.less
[0141]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.count.html#numpy.core.defchararray.count
[0142]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.find.html#numpy.core.defchararray.find
[0143]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.index.html#numpy.core.defchararray.index
[0144]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.isalpha.html#numpy.core.defchararray.isalpha
[0145]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.isdecimal.html#numpy.core.defchararray.isdecimal
[0146]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.isdigit.html#numpy.core.defchararray.isdigit
[0147]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.islower.html#numpy.core.defchararray.islower
[0148]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.isnumeric.html#numpy.core.defchararray.isnumeric
[0149]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.isspace.html#numpy.core.defchararray.isspace
[0150]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.istitle.html#numpy.core.defchararray.istitle
[0151]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.isupper.html#numpy.core.defchararray.isupper
[0152]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.rfind.html#numpy.core.defchararray.rfind
[0153]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.rindex.html#numpy.core.defchararray.rindex
[0154]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.startswith.html#numpy.core.defchararray.startswith
[0155]: https://www.numpy.org/devdocs/reference/generated/numpy.core.defchararray.chararray.html#numpy.core.defchararray.chararray
[0156]: https://www.numpy.org/devdocs/reference/routines.ctypeslib.html
[0157]: https://www.numpy.org/devdocs/reference/routines.datetime.html
[0158]: https://www.numpy.org/devdocs/reference/generated/numpy.datetime_as_string.html#numpy.datetime_as_string
[0159]: https://www.numpy.org/devdocs/reference/generated/numpy.datetime_data.html#numpy.datetime_data
[0160]: https://www.numpy.org/devdocs/reference/generated/numpy.busdaycalendar.html#numpy.busdaycalendar
[0161]: https://www.numpy.org/devdocs/reference/generated/numpy.is_busday.html#numpy.is_busday
[0162]: https://www.numpy.org/devdocs/reference/generated/numpy.busday_offset.html#numpy.busday_offset
[0163]: https://www.numpy.org/devdocs/reference/generated/numpy.busday_count.html#numpy.busday_count
[0164]: https://www.numpy.org/devdocs/reference/routines.dtype.html
[0165]: https://www.numpy.org/devdocs/reference/generated/numpy.can_cast.html#numpy.can_cast
[0166]: https://www.numpy.org/devdocs/reference/generated/numpy.promote_types.html#numpy.promote_types
[0167]: https://www.numpy.org/devdocs/reference/generated/numpy.min_scalar_type.html#numpy.min_scalar_type
[0168]: https://www.numpy.org/devdocs/reference/generated/numpy.result_type.html#numpy.result_type
[0169]: https://www.numpy.org/devdocs/reference/generated/numpy.common_type.html#numpy.common_type
[0170]: https://www.numpy.org/devdocs/reference/generated/numpy.obj2sctype.html#numpy.obj2sctype
[0171]: https://www.numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype
[0172]: https://www.numpy.org/devdocs/reference/generated/numpy.format_parser.html#numpy.format_parser
[0173]: https://www.numpy.org/devdocs/reference/generated/numpy.finfo.html#numpy.finfo
[0174]: https://www.numpy.org/devdocs/reference/generated/numpy.iinfo.html#numpy.iinfo
[0175]: https://www.numpy.org/devdocs/reference/generated/numpy.MachAr.html#numpy.MachAr
[0176]: https://www.numpy.org/devdocs/reference/generated/numpy.issctype.html#numpy.issctype
[0177]: https://www.numpy.org/devdocs/reference/generated/numpy.issubdtype.html#numpy.issubdtype
[0178]: https://www.numpy.org/devdocs/reference/generated/numpy.issubsctype.html#numpy.issubsctype
[0179]: https://www.numpy.org/devdocs/reference/generated/numpy.find_common_type.html#numpy.find_common_type
[0180]: https://www.numpy.org/devdocs/reference/generated/numpy.typename.html#numpy.typename
[0181]: https://www.numpy.org/devdocs/reference/generated/numpy.sctype2char.html#numpy.sctype2char
[0182]: https://www.numpy.org/devdocs/reference/generated/numpy.mintypecode.html#numpy.mintypecode
[0183]: https://www.numpy.org/devdocs/reference/routines.emath.html
[0184]: https://www.numpy.org/devdocs/reference/routines.err.html
[0185]: https://www.numpy.org/devdocs/reference/generated/numpy.seterr.html#numpy.seterr
[0186]: https://www.numpy.org/devdocs/reference/generated/numpy.geterr.html#numpy.geterr
[0187]: https://www.numpy.org/devdocs/reference/generated/numpy.seterrcall.html#numpy.seterrcall
[0188]: https://www.numpy.org/devdocs/reference/generated/numpy.geterrcall.html#numpy.geterrcall
[0189]: https://www.numpy.org/devdocs/reference/generated/numpy.errstate.html#numpy.errstate
[0190]: https://www.numpy.org/devdocs/reference/generated/numpy.seterrobj.html#numpy.seterrobj
[0191]: https://www.numpy.org/devdocs/reference/generated/numpy.geterrobj.html#numpy.geterrobj
[0192]: https://www.numpy.org/devdocs/reference/routines.fft.html
[0193]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fft.html#numpy.fft.fft
[0194]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft
[0195]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fft2.html#numpy.fft.fft2
[0196]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2
[0197]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fftn.html#numpy.fft.fftn
[0198]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifftn.html#numpy.fft.ifftn
[0199]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft

[0200]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.irfft.html#numpy.fft.irfft
[0201]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfft2.html#numpy.fft.rfft2
[0202]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.irfft2.html#numpy.fft.irfft2
[0203]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfftn.html#numpy.fft.rfftn
[0204]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.irfftn.html#numpy.fft.irfftn
[0205]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.hfft.html#numpy.fft.hfft
[0206]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ihfft.html#numpy.fft.ihfft
[0207]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq
[0208]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfftfreq.html#numpy.fft.rfftfreq
[0209]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift
[0210]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifftshift.html#numpy.fft.ifftshift
[0211]: https://www.numpy.org/devdocs/reference/routines.financial.html
[0212]: https://www.numpy.org/devdocs/reference/generated/numpy.fv.html#numpy.fv
[0213]: https://www.numpy.org/devdocs/reference/generated/numpy.pv.html#numpy.pv
[0214]: https://www.numpy.org/devdocs/reference/generated/numpy.npv.html#numpy.npv
[0215]: https://www.numpy.org/devdocs/reference/generated/numpy.pmt.html#numpy.pmt
[0216]: https://www.numpy.org/devdocs/reference/generated/numpy.ppmt.html#numpy.ppmt
[0217]: https://www.numpy.org/devdocs/reference/generated/numpy.ipmt.html#numpy.ipmt
[0218]: https://www.numpy.org/devdocs/reference/generated/numpy.irr.html#numpy.irr
[0219]: https://www.numpy.org/devdocs/reference/generated/numpy.mirr.html#numpy.mirr
[0220]: https://www.numpy.org/devdocs/reference/generated/numpy.nper.html#numpy.nper
[0221]: https://www.numpy.org/devdocs/reference/generated/numpy.rate.html#numpy.rate
[0222]: https://www.numpy.org/devdocs/reference/routines.functional.html
[0223]: https://www.numpy.org/devdocs/reference/generated/numpy.apply_along_axis.html#numpy.apply_along_axis
[0224]: https://www.numpy.org/devdocs/reference/generated/numpy.apply_over_axes.html#numpy.apply_over_axes
[0225]: https://www.numpy.org/devdocs/reference/generated/numpy.vectorize.html#numpy.vectorize
[0226]: https://www.numpy.org/devdocs/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc
[0227]: https://www.numpy.org/devdocs/reference/generated/numpy.piecewise.html#numpy.piecewise
[0228]: https://www.numpy.org/devdocs/reference/routines.help.html
[0229]: https://www.numpy.org/devdocs/reference/generated/numpy.lookfor.html#numpy.lookfor
[0230]: https://www.numpy.org/devdocs/reference/generated/numpy.info.html#numpy.info
[0231]: https://www.numpy.org/devdocs/reference/generated/numpy.source.html#numpy.source
[0232]: https://www.numpy.org/devdocs/reference/routines.indexing.html
[0233]: https://www.numpy.org/devdocs/reference/generated/numpy.c_.html#numpy.c_
[0234]: https://www.numpy.org/devdocs/reference/generated/numpy.r_.html#numpy.r_
[0235]: https://www.numpy.org/devdocs/reference/generated/numpy.s_.html#numpy.s_
[0236]: https://www.numpy.org/devdocs/reference/generated/numpy.nonzero.html#numpy.nonzero
[0237]: https://www.numpy.org/devdocs/reference/generated/numpy.where.html#numpy.where
[0238]: https://www.numpy.org/devdocs/reference/generated/numpy.indices.html#numpy.indices
[0239]: https://www.numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_
[0240]: https://www.numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid
[0241]: https://www.numpy.org/devdocs/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index
[0242]: https://www.numpy.org/devdocs/reference/generated/numpy.unravel_index.html#numpy.unravel_index
[0243]: https://www.numpy.org/devdocs/reference/generated/numpy.diag_indices.html#numpy.diag_indices
[0244]: https://www.numpy.org/devdocs/reference/generated/numpy.diag_indices_from.html#numpy.diag_indices_from
[0245]: https://www.numpy.org/devdocs/reference/generated/numpy.mask_indices.html#numpy.mask_indices
[0246]: https://www.numpy.org/devdocs/reference/generated/numpy.tril_indices.html#numpy.tril_indices
[0247]: https://www.numpy.org/devdocs/reference/generated/numpy.tril_indices_from.html#numpy.tril_indices_from
[0248]: https://www.numpy.org/devdocs/reference/generated/numpy.triu_indices.html#numpy.triu_indices
[0249]: https://www.numpy.org/devdocs/reference/generated/numpy.triu_indices_from.html#numpy.triu_indices_from
[0250]: https://www.numpy.org/devdocs/reference/generated/numpy.take.html#numpy.take
[0251]: https://www.numpy.org/devdocs/reference/generated/numpy.choose.html#numpy.choose
[0252]: https://www.numpy.org/devdocs/reference/generated/numpy.compress.html#numpy.compress
[0253]: https://www.numpy.org/devdocs/reference/generated/numpy.diag.html#numpy.diag
[0254]: https://www.numpy.org/devdocs/reference/generated/numpy.diagonal.html#numpy.diagonal
[0255]: https://www.numpy.org/devdocs/reference/generated/numpy.select.html#numpy.select
[0256]: https://www.numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.as_strided.html#numpy.lib.stride_tricks.as_strided
[0257]: https://www.numpy.org/devdocs/reference/generated/numpy.place.html#numpy.place
[0258]: https://www.numpy.org/devdocs/reference/generated/numpy.put.html#numpy.put
[0259]: https://www.numpy.org/devdocs/reference/generated/numpy.putmask.html#numpy.putmask
[0260]: https://www.numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html#numpy.fill_diagonal
[0261]: https://www.numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer
[0262]: https://www.numpy.org/devdocs/reference/generated/numpy.ndenumerate.html#numpy.ndenumerate
[0263]: https://www.numpy.org/devdocs/reference/generated/numpy.ndindex.html#numpy.ndindex
[0264]: https://www.numpy.org/devdocs/reference/generated/numpy.nested_iters.html#numpy.nested_iters
[0265]: https://www.numpy.org/devdocs/reference/generated/numpy.flatiter.html#numpy.flatiter
[0266]: https://www.numpy.org/devdocs/reference/generated/numpy.lib.Arrayterator.html#numpy.lib.Arrayterator
[0267]: https://www.numpy.org/devdocs/reference/routines.io.html
[0268]: https://www.numpy.org/devdocs/reference/generated/numpy.load.html#numpy.load
[0269]: https://www.numpy.org/devdocs/reference/generated/numpy.save.html#numpy.save
[0270]: https://www.numpy.org/devdocs/reference/generated/numpy.savez.html#numpy.savez
[0271]: https://www.numpy.org/devdocs/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
[0272]: https://www.numpy.org/devdocs/reference/generated/numpy.loadtxt.html#numpy.loadtxt
[0273]: https://www.numpy.org/devdocs/reference/generated/numpy.savetxt.html#numpy.savetxt
[0274]: https://www.numpy.org/devdocs/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt
[0275]: https://www.numpy.org/devdocs/reference/generated/numpy.fromregex.html#numpy.fromregex
[0276]: https://www.numpy.org/devdocs/reference/generated/numpy.fromstring.html#numpy.fromstring
[0277]: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile
[0278]: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.tolist.html#numpy.ndarray.tolist
[0279]: https://www.numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile
[0280]: https://www.numpy.org/devdocs/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile
[0281]: https://www.numpy.org/devdocs/reference/generated/numpy.array2string.html#numpy.array2string
[0282]: https://www.numpy.org/devdocs/reference/generated/numpy.array_repr.html#numpy.array_repr
[0283]: https://www.numpy.org/devdocs/reference/generated/numpy.array_str.html#numpy.array_str
[0284]: https://www.numpy.org/devdocs/reference/generated/numpy.format_float_positional.html#numpy.format_float_positional
[0285]: https://www.numpy.org/devdocs/reference/generated/numpy.format_float_scientific.html#numpy.format_float_scientific
[0286]: https://www.numpy.org/devdocs/reference/generated/numpy.memmap.html#numpy.memmap
[0287]: https://www.numpy.org/devdocs/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions
[0288]: https://www.numpy.org/devdocs/reference/generated/numpy.get_printoptions.html#numpy.get_printoptions
[0289]: https://www.numpy.org/devdocs/reference/generated/numpy.set_string_function.html#numpy.set_string_function
[0290]: https://www.numpy.org/devdocs/reference/generated/numpy.binary_repr.html#numpy.binary_repr
[0291]: https://www.numpy.org/devdocs/reference/generated/numpy.base_repr.html#numpy.base_repr
[0292]: https://www.numpy.org/devdocs/reference/generated/numpy.DataSource.html#numpy.DataSource
[0293]: https://www.numpy.org/devdocs/reference/generated/numpy.lib.format.html#module-numpy.lib.format
[0294]: https://www.numpy.org/devdocs/reference/routines.linalg.html
[0295]: https://www.numpy.org/devdocs/reference/generated/numpy.dot.html#numpy.dot
[0296]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot
[0297]: https://www.numpy.org/devdocs/reference/generated/numpy.vdot.html#numpy.vdot
[0298]: https://www.numpy.org/devdocs/reference/generated/numpy.inner.html#numpy.inner
[0299]: https://www.numpy.org/devdocs/reference/generated/numpy.outer.html#numpy.outer

[0300]: https://www.numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul
[0301]: https://www.numpy.org/devdocs/reference/generated/numpy.tensordot.html#numpy.tensordot
[0302]: https://www.numpy.org/devdocs/reference/generated/numpy.einsum.html#numpy.einsum
[0303]: https://www.numpy.org/devdocs/reference/generated/numpy.einsum_path.html#numpy.einsum_path
[0304]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.matrix_power.html#numpy.linalg.matrix_power
[0305]: https://www.numpy.org/devdocs/reference/generated/numpy.kron.html#numpy.kron
[0306]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky
[0307]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.qr.html#numpy.linalg.qr
[0308]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd
[0309]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig
[0310]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh
[0311]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals
[0312]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh
[0313]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm
[0314]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.cond.html#numpy.linalg.cond
[0315]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.det.html#numpy.linalg.det
[0316]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.matrix_rank.html#numpy.linalg.matrix_rank
[0317]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet
[0318]: https://www.numpy.org/devdocs/reference/generated/numpy.trace.html#numpy.trace
[0319]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve
[0320]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.tensorsolve.html#numpy.linalg.tensorsolve
[0321]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq
[0322]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv
[0323]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv
[0324]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.tensorinv.html#numpy.linalg.tensorinv
[0325]: https://www.numpy.org/devdocs/reference/generated/numpy.linalg.LinAlgError.html#numpy.linalg.LinAlgError
[0326]: https://www.numpy.org/devdocs/reference/routines.logic.html
[0327]: https://www.numpy.org/devdocs/reference/generated/numpy.all.html#numpy.all
[0328]: https://www.numpy.org/devdocs/reference/generated/numpy.any.html#numpy.any
[0329]: https://www.numpy.org/devdocs/reference/generated/numpy.isfinite.html#numpy.isfinite
[0330]: https://www.numpy.org/devdocs/reference/generated/numpy.isinf.html#numpy.isinf
[0331]: https://www.numpy.org/devdocs/reference/generated/numpy.isnan.html#numpy.isnan
[0332]: https://www.numpy.org/devdocs/reference/generated/numpy.isnat.html#numpy.isnat
[0333]: https://www.numpy.org/devdocs/reference/generated/numpy.isneginf.html#numpy.isneginf
[0334]: https://www.numpy.org/devdocs/reference/generated/numpy.isposinf.html#numpy.isposinf
[0335]: https://www.numpy.org/devdocs/reference/generated/numpy.iscomplex.html#numpy.iscomplex
[0336]: https://www.numpy.org/devdocs/reference/generated/numpy.iscomplexobj.html#numpy.iscomplexobj
[0337]: https://www.numpy.org/devdocs/reference/generated/numpy.isfortran.html#numpy.isfortran
[0338]: https://www.numpy.org/devdocs/reference/generated/numpy.isreal.html#numpy.isreal
[0339]: https://www.numpy.org/devdocs/reference/generated/numpy.isrealobj.html#numpy.isrealobj
[0340]: https://www.numpy.org/devdocs/reference/generated/numpy.isscalar.html#numpy.isscalar
[0341]: https://www.numpy.org/devdocs/reference/generated/numpy.logical_and.html#numpy.logical_and
[0342]: https://www.numpy.org/devdocs/reference/generated/numpy.logical_or.html#numpy.logical_or
[0343]: https://www.numpy.org/devdocs/reference/generated/numpy.logical_not.html#numpy.logical_not
[0344]: https://www.numpy.org/devdocs/reference/generated/numpy.logical_xor.html#numpy.logical_xor
[0345]: https://www.numpy.org/devdocs/reference/generated/numpy.allclose.html#numpy.allclose
[0346]: https://www.numpy.org/devdocs/reference/generated/numpy.isclose.html#numpy.isclose
[0347]: https://www.numpy.org/devdocs/reference/generated/numpy.array_equal.html#numpy.array_equal
[0348]: https://www.numpy.org/devdocs/reference/generated/numpy.array_equiv.html#numpy.array_equiv
[0349]: https://www.numpy.org/devdocs/reference/generated/numpy.greater.html#numpy.greater
[0350]: https://www.numpy.org/devdocs/reference/generated/numpy.greater_equal.html#numpy.greater_equal
[0351]: https://www.numpy.org/devdocs/reference/generated/numpy.less.html#numpy.less
[0352]: https://www.numpy.org/devdocs/reference/generated/numpy.less_equal.html#numpy.less_equal
[0353]: https://www.numpy.org/devdocs/reference/generated/numpy.equal.html#numpy.equal
[0354]: https://www.numpy.org/devdocs/reference/generated/numpy.not_equal.html#numpy.not_equal
[0355]: https://www.numpy.org/devdocs/reference/routines.ma.html
[0356]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskType.html#numpy.ma.MaskType
[0357]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_array.html#numpy.ma.masked_array
[0358]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.array.html#numpy.ma.array
[0359]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.copy.html#numpy.ma.copy
[0360]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.frombuffer.html#numpy.ma.frombuffer
[0361]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.fromfunction.html#numpy.ma.fromfunction
[0362]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.copy.html#numpy.ma.MaskedArray.copy
[0363]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.empty.html#numpy.ma.empty
[0364]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.empty_like.html#numpy.ma.empty_like
[0365]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_all.html#numpy.ma.masked_all
[0366]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_all_like.html#numpy.ma.masked_all_like
[0367]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.ones.html#numpy.ma.ones
[0368]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.zeros.html#numpy.ma.zeros
[0369]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.all.html#numpy.ma.all
[0370]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.any.html#numpy.ma.any
[0371]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.count.html#numpy.ma.count
[0372]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.count_masked.html#numpy.ma.count_masked
[0373]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.getmask.html#numpy.ma.getmask
[0374]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.getmaskarray.html#numpy.ma.getmaskarray
[0375]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.getdata.html#numpy.ma.getdata
[0376]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.nonzero.html#numpy.ma.nonzero
[0377]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.shape.html#numpy.ma.shape
[0378]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.size.html#numpy.ma.size
[0379]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.is_masked.html#numpy.ma.is_masked
[0380]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.is_mask.html#numpy.ma.is_mask
[0381]: https://www.numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.data
[0382]: https://www.numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.mask
[0383]: https://www.numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.recordmask
[0384]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.all.html#numpy.ma.MaskedArray.all
[0385]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.any.html#numpy.ma.MaskedArray.any
[0386]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.count.html#numpy.ma.MaskedArray.count
[0387]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.nonzero.html#numpy.ma.MaskedArray.nonzero
[0388]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.shape.html#numpy.ma.shape
[0389]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.size.html#numpy.ma.size
[0390]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.ravel.html#numpy.ma.ravel
[0391]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.reshape.html#numpy.ma.reshape
[0392]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.resize.html#numpy.ma.resize
[0393]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.flatten.html#numpy.ma.MaskedArray.flatten
[0394]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.ravel.html#numpy.ma.MaskedArray.ravel
[0395]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.reshape.html#numpy.ma.MaskedArray.reshape
[0396]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.resize.html#numpy.ma.MaskedArray.resize
[0397]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.swapaxes.html#numpy.ma.swapaxes
[0398]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.transpose.html#numpy.ma.transpose
[0399]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.swapaxes.html#numpy.ma.MaskedArray.swapaxes

[0400]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.transpose.html#numpy.ma.MaskedArray.transpose
[0401]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.atleast_1d.html#numpy.ma.atleast_1d
[0402]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.atleast_2d.html#numpy.ma.atleast_2d
[0403]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.atleast_3d.html#numpy.ma.atleast_3d
[0404]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.expand_dims.html#numpy.ma.expand_dims
[0405]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.squeeze.html#numpy.ma.squeeze
[0406]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.squeeze.html#numpy.ma.MaskedArray.squeeze
[0407]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.column_stack.html#numpy.ma.column_stack
[0408]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.concatenate.html#numpy.ma.concatenate
[0409]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.dstack.html#numpy.ma.dstack
[0410]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.hstack.html#numpy.ma.hstack
[0411]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.hsplit.html#numpy.ma.hsplit
[0412]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.mr_.html#numpy.ma.mr_
[0413]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.row_stack.html#numpy.ma.row_stack
[0414]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.vstack.html#numpy.ma.vstack
[0415]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.column_stack.html#numpy.ma.column_stack
[0416]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.concatenate.html#numpy.ma.concatenate
[0417]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.append.html#numpy.ma.append
[0418]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.dstack.html#numpy.ma.dstack
[0419]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.hstack.html#numpy.ma.hstack
[0420]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.vstack.html#numpy.ma.vstack
[0421]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.make_mask.html#numpy.ma.make_mask
[0422]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.make_mask_none.html#numpy.ma.make_mask_none
[0423]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.mask_or.html#numpy.ma.mask_or
[0424]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.make_mask_descr.html#numpy.ma.make_mask_descr
[0425]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.getmask.html#numpy.ma.getmask
[0426]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.getmaskarray.html#numpy.ma.getmaskarray
[0427]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_array.mask.html#numpy.ma.masked_array.mask
[0428]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.flatnotmasked_contiguous.html#numpy.ma.flatnotmasked_contiguous
[0429]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.flatnotmasked_edges.html#numpy.ma.flatnotmasked_edges
[0430]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.notmasked_contiguous.html#numpy.ma.notmasked_contiguous
[0431]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.notmasked_edges.html#numpy.ma.notmasked_edges
[0432]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.clump_masked.html#numpy.ma.clump_masked
[0433]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.clump_unmasked.html#numpy.ma.clump_unmasked
[0434]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.mask_cols.html#numpy.ma.mask_cols
[0435]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.mask_or.html#numpy.ma.mask_or
[0436]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.mask_rowcols.html#numpy.ma.mask_rowcols
[0437]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.mask_rows.html#numpy.ma.mask_rows
[0438]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.harden_mask.html#numpy.ma.harden_mask
[0439]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.soften_mask.html#numpy.ma.soften_mask
[0440]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.harden_mask.html#numpy.ma.MaskedArray.harden_mask
[0441]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.soften_mask.html#numpy.ma.MaskedArray.soften_mask
[0442]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.shrink_mask.html#numpy.ma.MaskedArray.shrink_mask
[0443]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.unshare_mask.html#numpy.ma.MaskedArray.unshare_mask
[0444]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.asarray.html#numpy.ma.asarray
[0445]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.asanyarray.html#numpy.ma.asanyarray
[0446]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.fix_invalid.html#numpy.ma.fix_invalid
[0447]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_equal.html#numpy.ma.masked_equal
[0448]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_greater.html#numpy.ma.masked_greater
[0449]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_greater_equal.html#numpy.ma.masked_greater_equal
[0450]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_inside.html#numpy.ma.masked_inside
[0451]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_invalid.html#numpy.ma.masked_invalid
[0452]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_less.html#numpy.ma.masked_less
[0453]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_less_equal.html#numpy.ma.masked_less_equal
[0454]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_not_equal.html#numpy.ma.masked_not_equal
[0455]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_object.html#numpy.ma.masked_object
[0456]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_outside.html#numpy.ma.masked_outside
[0457]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_values.html#numpy.ma.masked_values
[0458]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.masked_where.html#numpy.ma.masked_where
[0459]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.compress_cols.html#numpy.ma.compress_cols
[0460]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.compress_rowcols.html#numpy.ma.compress_rowcols
[0461]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.compress_rows.html#numpy.ma.compress_rows
[0462]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.compressed.html#numpy.ma.compressed
[0463]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.filled.html#numpy.ma.filled
[0464]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.compressed.html#numpy.ma.MaskedArray.compressed
[0465]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.filled.html#numpy.ma.MaskedArray.filled
[0466]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.tofile.html#numpy.ma.MaskedArray.tofile
[0467]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.tolist.html#numpy.ma.MaskedArray.tolist
[0468]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.torecords.html#numpy.ma.MaskedArray.torecords
[0469]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.tobytes.html#numpy.ma.MaskedArray.tobytes
[0470]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.dump.html#numpy.ma.dump
[0471]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.dumps.html#numpy.ma.dumps
[0472]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.load.html#numpy.ma.load
[0473]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.loads.html#numpy.ma.loads
[0474]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.common_fill_value.html#numpy.ma.common_fill_value
[0475]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.default_fill_value.html#numpy.ma.default_fill_value
[0476]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.maximum_fill_value.html#numpy.ma.maximum_fill_value
[0477]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.maximum_fill_value.html#numpy.ma.maximum_fill_value
[0478]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.set_fill_value.html#numpy.ma.set_fill_value
[0479]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.get_fill_value.html#numpy.ma.MaskedArray.get_fill_value
[0480]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.set_fill_value.html#numpy.ma.MaskedArray.set_fill_value
[0481]: https://www.numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.fill_value
[0482]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.anom.html#numpy.ma.anom
[0483]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.anomalies.html#numpy.ma.anomalies
[0484]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.average.html#numpy.ma.average
[0485]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.conjugate.html#numpy.ma.conjugate
[0486]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.corrcoef.html#numpy.ma.corrcoef
[0487]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.cov.html#numpy.ma.cov
[0488]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.cumsum.html#numpy.ma.cumsum
[0489]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.cumprod.html#numpy.ma.cumprod
[0490]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.mean.html#numpy.ma.mean
[0491]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.median.html#numpy.ma.median
[0492]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.power.html#numpy.ma.power
[0493]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.prod.html#numpy.ma.prod
[0494]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.std.html#numpy.ma.std
[0495]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.sum.html#numpy.ma.sum
[0496]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.var.html#numpy.ma.var
[0497]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.anom.html#numpy.ma.MaskedArray.anom
[0498]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.cumprod.html#numpy.ma.MaskedArray.cumprod
[0499]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.cumsum.html#numpy.ma.MaskedArray.cumsum

[0500]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.mean.html#numpy.ma.MaskedArray.mean
[0501]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.prod.html#numpy.ma.MaskedArray.prod
[0502]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.std.html#numpy.ma.MaskedArray.std
[0503]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.sum.html#numpy.ma.MaskedArray.sum
[0504]: htt5s://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.var.html#numpy.ma.MaskedArray.var
[0505]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.argmax.html#numpy.ma.argmax
[0506]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.argmin.html#numpy.ma.argmin
[0507]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.max.html#numpy.ma.max
[0508]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.min.html#numpy.ma.min
[0509]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.ptp.html#numpy.ma.ptp
[0510]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.argmax.html#numpy.ma.MaskedArray.argmax
[0511]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.argmin.html#numpy.ma.MaskedArray.argmin
[0512]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.max.html#numpy.ma.MaskedArray.max
[0513]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.min.html#numpy.ma.MaskedArray.min
[0514]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.ptp.html#numpy.ma.MaskedArray.ptp
[0515]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.argsort.html#numpy.ma.argsort
[0516]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.sort.html#numpy.ma.sort
[0517]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.argsort.html#numpy.ma.MaskedArray.argsort
[0518]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.sort.html#numpy.ma.MaskedArray.sort
[0519]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.diag.html#numpy.ma.diag
[0520]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.dot.html#numpy.ma.dot
[0521]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.identity.html#numpy.ma.identity
[0522]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.inner.html#numpy.ma.inner
[0523]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.innerproduct.html#numpy.ma.innerproduct
[0524]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.outer.html#numpy.ma.outer
[0525]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.outerproduct.html#numpy.ma.outerproduct
[0526]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.trace.html#numpy.ma.trace
[0527]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.transpose.html#numpy.ma.transpose
[0528]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.trace.html#numpy.ma.MaskedArray.trace
[0529]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.transpose.html#numpy.ma.MaskedArray.transpose
[0530]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.vander.html#numpy.ma.vander
[0531]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.polyfit.html#numpy.ma.polyfit
[0532]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.around.html#numpy.ma.around
[0533]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.clip.html#numpy.ma.clip
[0534]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.round.html#numpy.ma.round
[0535]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.clip.html#numpy.ma.MaskedArray.clip
[0536]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.round.html#numpy.ma.MaskedArray.round
[0537]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.allequal.html#numpy.ma.allequal
[0538]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.allclose.html#numpy.ma.allclose
[0539]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.apply_along_axis.html#numpy.ma.apply_along_axis
[0540]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.arange.html#numpy.ma.arange
[0541]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.choose.html#numpy.ma.choose
[0542]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.ediff1d.html#numpy.ma.ediff1d
[0543]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.indices.html#numpy.ma.indices
[0544]: https://www.numpy.org/devdocs/reference/generated/numpy.ma.where.html#numpy.ma.where
[0545]: https://www.numpy.org/devdocs/reference/routines.math.html
[0546]: https://www.numpy.org/devdocs/reference/generated/numpy.sin.html#numpy.sin
[0547]: https://www.numpy.org/devdocs/reference/generated/numpy.cos.html#numpy.cos
[0548]: https://www.numpy.org/devdocs/reference/generated/numpy.tan.html#numpy.tan
[0549]: https://www.numpy.org/devdocs/reference/generated/numpy.arcsin.html#numpy.arcsin
[0550]: https://www.numpy.org/devdocs/reference/generated/numpy.arccos.html#numpy.arccos
[0551]: https://www.numpy.org/devdocs/reference/generated/numpy.arctan.html#numpy.arctan
[0552]: https://www.numpy.org/devdocs/reference/generated/numpy.hypot.html#numpy.hypot
[0553]: https://www.numpy.org/devdocs/reference/generated/numpy.arctan2.html#numpy.arctan2
[0554]: https://www.numpy.org/devdocs/reference/generated/numpy.degrees.html#numpy.degrees
[0555]: https://www.numpy.org/devdocs/reference/generated/numpy.radians.html#numpy.radians
[0556]: https://www.numpy.org/devdocs/reference/generated/numpy.unwrap.html#numpy.unwrap
[0557]: https://www.numpy.org/devdocs/reference/generated/numpy.deg2rad.html#numpy.deg2rad
[0558]: https://www.numpy.org/devdocs/reference/generated/numpy.rad2deg.html#numpy.rad2deg
[0559]: https://www.numpy.org/devdocs/reference/generated/numpy.sinh.html#numpy.sinh
[0560]: https://www.numpy.org/devdocs/reference/generated/numpy.cosh.html#numpy.cosh
[0561]: https://www.numpy.org/devdocs/reference/generated/numpy.tanh.html#numpy.tanh
[0562]: https://www.numpy.org/devdocs/reference/generated/numpy.arcsinh.html#numpy.arcsinh
[0563]: https://www.numpy.org/devdocs/reference/generated/numpy.arccosh.html#numpy.arccosh
[0564]: https://www.numpy.org/devdocs/reference/generated/numpy.arctanh.html#numpy.arctanh
[0565]: https://www.numpy.org/devdocs/reference/generated/numpy.around.html#numpy.around
[0566]: https://www.numpy.org/devdocs/reference/generated/numpy.round_.html#numpy.round_
[0567]: https://www.numpy.org/devdocs/reference/generated/numpy.rint.html#numpy.rint
[0568]: https://www.numpy.org/devdocs/reference/generated/numpy.fix.html#numpy.fix
[0569]: https://www.numpy.org/devdocs/reference/generated/numpy.floor.html#numpy.floor
[0570]: https://www.numpy.org/devdocs/reference/generated/numpy.ceil.html#numpy.ceil
[0571]: https://www.numpy.org/devdocs/reference/generated/numpy.trunc.html#numpy.trunc
[0572]: https://www.numpy.org/devdocs/reference/generated/numpy.prod.html#numpy.prod
[0573]: https://www.numpy.org/devdocs/reference/generated/numpy.sum.html#numpy.sum
[0574]: https://www.numpy.org/devdocs/reference/generated/numpy.nanprod.html#numpy.nanprod
[0575]: https://www.numpy.org/devdocs/reference/generated/numpy.nansum.html#numpy.nansum
[0576]: https://www.numpy.org/devdocs/reference/generated/numpy.cumprod.html#numpy.cumprod
[0577]: https://www.numpy.org/devdocs/reference/generated/numpy.cumsum.html#numpy.cumsum
[0578]: https://www.numpy.org/devdocs/reference/generated/numpy.nancumprod.html#numpy.nancumprod
[0579]: https://www.numpy.org/devdocs/reference/generated/numpy.nancumsum.html#numpy.nancumsum
[0580]: https://www.numpy.org/devdocs/reference/generated/numpy.diff.html#numpy.diff
[0581]: https://www.numpy.org/devdocs/reference/generated/numpy.ediff1d.html#numpy.ediff1d
[0582]: https://www.numpy.org/devdocs/reference/generated/numpy.gradient.html#numpy.gradient
[0583]: https://www.numpy.org/devdocs/reference/generated/numpy.cross.html#numpy.cross
[0584]: https://www.numpy.org/devdocs/reference/generated/numpy.trapz.html#numpy.trapz
[0585]: https://www.numpy.org/devdocs/reference/generated/numpy.exp.html#numpy.exp
[0586]: https://www.numpy.org/devdocs/reference/generated/numpy.expm1.html#numpy.expm1
[0587]: https://www.numpy.org/devdocs/reference/generated/numpy.exp2.html#numpy.exp2
[0588]: https://www.numpy.org/devdocs/reference/generated/numpy.log.html#numpy.log
[0589]: https://www.numpy.org/devdocs/reference/generated/numpy.log10.html#numpy.log10
[0590]: https://www.numpy.org/devdocs/reference/generated/numpy.log2.html#numpy.log2
[0591]: https://www.numpy.org/devdocs/reference/generated/numpy.log1p.html#numpy.log1p
[0592]: https://www.numpy.org/devdocs/reference/generated/numpy.logaddexp.html#numpy.logaddexp
[0593]: https://www.numpy.org/devdocs/reference/generated/numpy.logaddexp2.html#numpy.logaddexp2
[0594]: https://www.numpy.org/devdocs/reference/generated/numpy.i0.html#numpy.i0
[0595]: https://www.numpy.org/devdocs/reference/generated/numpy.sinc.html#numpy.sinc
[0596]: https://www.numpy.org/devdocs/reference/generated/numpy.signbit.html#numpy.signbit
[0597]: https://www.numpy.org/devdocs/reference/generated/numpy.copysign.html#numpy.copysign
[0598]: https://www.numpy.org/devdocs/reference/generated/numpy.frexp.html#numpy.frexp
[0599]: https://www.numpy.org/devdocs/reference/generated/numpy.ldexp.html#numpy.ldexp

[0600]: https://www.numpy.org/devdocs/reference/generated/numpy.nextafter.html#numpy.nextafter
[0601]: https://www.numpy.org/devdocs/reference/generated/numpy.spacing.html#numpy.spacing
[0602]: https://www.numpy.org/devdocs/reference/generated/numpy.lcm.html#numpy.lcm
[0603]: https://www.numpy.org/devdocs/reference/generated/numpy.gcd.html#numpy.gcd
[0604]: https://www.numpy.org/devdocs/reference/generated/numpy.add.html#numpy.add
[0605]: https://www.numpy.org/devdocs/reference/generated/numpy.reciprocal.html#numpy.reciprocal
[0606]: https://www.numpy.org/devdocs/reference/generated/numpy.positive.html#numpy.positive
[0607]: https://www.numpy.org/devdocs/reference/generated/numpy.negative.html#numpy.negative
[0608]: https://www.numpy.org/devdocs/reference/generated/numpy.multiply.html#numpy.multiply
[0609]: https://www.numpy.org/devdocs/reference/generated/numpy.divide.html#numpy.divide
[0610]: https://www.numpy.org/devdocs/reference/generated/numpy.power.html#numpy.power
[0611]: https://www.numpy.org/devdocs/reference/generated/numpy.subtract.html#numpy.subtract
[0612]: https://www.numpy.org/devdocs/reference/generated/numpy.true_divide.html#numpy.true_divide
[0613]: https://www.numpy.org/devdocs/reference/generated/numpy.floor_divide.html#numpy.floor_divide
[0614]: https://www.numpy.org/devdocs/reference/generated/numpy.float_power.html#numpy.float_power
[0615]: https://www.numpy.org/devdocs/reference/generated/numpy.fmod.html#numpy.fmod
[0616]: https://www.numpy.org/devdocs/reference/generated/numpy.mod.html#numpy.mod
[0617]: https://www.numpy.org/devdocs/reference/generated/numpy.modf.html#numpy.modf
[0618]: https://www.numpy.org/devdocs/reference/generated/numpy.remainder.html#numpy.remainder
[0619]: https://www.numpy.org/devdocs/reference/generated/numpy.divmod.html#numpy.divmod
[0620]: https://www.numpy.org/devdocs/reference/generated/numpy.angle.html#numpy.angle
[0621]: https://www.numpy.org/devdocs/reference/generated/numpy.real.html#numpy.real
[0622]: https://www.numpy.org/devdocs/reference/generated/numpy.imag.html#numpy.imag
[0623]: https://www.numpy.org/devdocs/reference/generated/numpy.conj.html#numpy.conj
[0624]: https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html#numpy.convolve
[0625]: https://www.numpy.org/devdocs/reference/generated/numpy.clip.html#numpy.clip
[0626]: https://www.numpy.org/devdocs/reference/generated/numpy.sqrt.html#numpy.sqrt
[0627]: https://www.numpy.org/devdocs/reference/generated/numpy.cbrt.html#numpy.cbrt
[0628]: https://www.numpy.org/devdocs/reference/generated/numpy.square.html#numpy.square
[0629]: https://www.numpy.org/devdocs/reference/generated/numpy.absolute.html#numpy.absolute
[0630]: https://www.numpy.org/devdocs/reference/generated/numpy.fabs.html#numpy.fabs
[0631]: https://www.numpy.org/devdocs/reference/generated/numpy.sign.html#numpy.sign
[0632]: https://www.numpy.org/devdocs/reference/generated/numpy.heaviside.html#numpy.heaviside
[0633]: https://www.numpy.org/devdocs/reference/generated/numpy.maximum.html#numpy.maximum
[0634]: https://www.numpy.org/devdocs/reference/generated/numpy.minimum.html#numpy.minimum
[0635]: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html#numpy.fmax
[0636]: https://www.numpy.org/devdocs/reference/generated/numpy.fmin.html#numpy.fmin
[0637]: https://www.numpy.org/devdocs/reference/generated/numpy.nan_to_num.html#numpy.nan_to_num
[0638]: https://www.numpy.org/devdocs/reference/generated/numpy.real_if_close.html#numpy.real_if_close
[0639]: https://www.numpy.org/devdocs/reference/generated/numpy.interp.html#numpy.interp
[0640]: https://www.numpy.org/devdocs/reference/routines.matlib.html
[0641]: https://www.numpy.org/devdocs/reference/generated/numpy.mat.html#numpy.mat
[0642]: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix
[0643]: https://www.numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix
[0644]: https://www.numpy.org/devdocs/reference/generated/numpy.bmat.html#numpy.bmat
[0645]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.empty.html#numpy.matlib.empty
[0646]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.zeros.html#numpy.matlib.zeros
[0647]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.ones.html#numpy.matlib.ones
[0648]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.eye.html#numpy.matlib.eye
[0649]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.identity.html#numpy.matlib.identity
[0650]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.repmat.html#numpy.matlib.repmat
[0651]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.rand.html#numpy.matlib.rand
[0652]: https://www.numpy.org/devdocs/reference/generated/numpy.matlib.randn.html#numpy.matlib.randn
[0653]: https://www.numpy.org/devdocs/reference/routines.other.html
[0654]: https://www.numpy.org/devdocs/reference/generated/numpy.setbufsize.html#numpy.setbufsize
[0655]: https://www.numpy.org/devdocs/reference/generated/numpy.getbufsize.html#numpy.getbufsize
[0656]: https://www.numpy.org/devdocs/reference/generated/numpy.shares_memory.html#numpy.shares_memory
[0657]: https://www.numpy.org/devdocs/reference/generated/numpy.may_share_memory.html#numpy.may_share_memory
[0658]: https://www.numpy.org/devdocs/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
[0659]: https://www.numpy.org/devdocs/reference/generated/numpy.lib.NumpyVersion.html#numpy.lib.NumpyVersion
[0660]: 
[0661]: 
[0662]: 
[0663]: 
[0664]: 
[0665]: 
[0666]: 
[0667]: 
[0668]: 
[0669]: 
[0670]: 
[0671]: 
[0672]: 
[0673]: 
[0674]: 
[0675]: 
[0676]: 
[0677]: 
[0678]: 
[0679]: 
[0680]: 
[0681]: 
[0682]: 
[0683]: 
[0684]: 
[0685]: 
[0686]: 
[0687]: 
[0688]: 
[0689]: 
[0690]: 
[0691]: 
[0692]: 
[0693]: 
[0694]: 
[0695]: 
[0696]: 
[0697]: 
[0698]: 
[0699]: 

[0700]: 
[0701]: 
[0702]: 
[0703]: 
[0704]: 
[0705]: 
[0706]: 
[0707]: 
[0708]: 
[0709]: 
[0710]: 
[0711]: 
[0712]: 
[0713]: 
[0714]: 
[0715]: 
[0716]: 
[0717]: 
[0718]: 
[0719]: 
[0720]: 
[0721]: 
[0722]: 
[0723]: 
[0724]: 
[0725]: 
[0726]: 
[0727]: 
[0728]: 
[0729]: 
[0730]: 
[0731]: 
[0732]: 
[0733]: 
[0734]: 
[0735]: 
[0736]: 
[0737]: 
[0738]: 
[0739]: 
[0740]: 
[0741]: 
[0742]: 
[0743]: 
[0744]: 
[0745]: 
[0746]: 
[0747]: 
[0748]: 
[0749]: 
[0750]: 
[0751]: 
[0752]: 
[0753]: 
[0754]: 
[0755]: 
[0756]: 
[0757]: 
[0758]: 
[0759]: 
[0760]: 
[0761]: 
[0762]: 
[0763]: 
[0764]: 
[0765]: 
[0766]: 
[0767]: 
[0768]: 
[0769]: 
[0770]: 
[0771]: 
[0772]: 
[0773]: 
[0774]: 
[0775]: 
[0776]: 
[0777]: 
[0778]: 
[0779]: 
[0780]: 
[0781]: 
[0782]: 
[0783]: 
[0784]: 
[0785]: 
[0786]: 
[0787]: 
[0788]: 
[0789]: 
[0790]: 
[0791]: 
[0792]: 
[0793]: 
[0794]: 
[0795]: 
[0796]: 
[0797]: 
[0798]: 
[0799]: 


