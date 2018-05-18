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


| `can_cast(from_, to[, casting])` | Returns True if cast between data types can occur according to the casting rule. | [API][0164]
| `promote_types(type1, type2)` | Returns the data type with the smallest size and smallest scalar kind to which both type1 and type2 may be safely cast. | [API][0165]
| `min_scalar_type(a)` | For scalar a, returns the data type with the smallest size and smallest scalar kind which can hold its value. | [API][0166]
| `result_type(*arrays_and_dtypes)` | Returns the type that results from applying the NumPy type promotion rules to the arguments. | [API][0167]
| `common_type(*arrays)` | Return a scalar type which is common to the input arrays. | [API][0168]
| `obj2sctype(rep[, default])` | Return the scalar dtype or NumPy equivalent of Python type of an object. | [API][0169]

### Creating data types 

| `dtype(obj[, align, copy])` | Create a data type object. | [API][0170]
| `format_parser(formats, names, titles[, …])` | Class to convert formats, names, titles description to a dtype. | [API][0171]

### Data type information

| `finfo(dtype)` | Machine limits for floating point types. | [API][0172]
| `iinfo(type)` | Machine limits for integer types. | [API][0173]
| `MachAr([float_conv, int_conv, …])` | Diagnosing machine parameters. | [API][0174]

### Data type testing

| `issctype(rep)` | Determines whether the given object represents a scalar data-type. | [API][0175]
| `issubdtype(arg1, arg2)` | Returns True if first argument is a typecode lower/equal in type hierarchy. | [API][0176]
| `issubsctype(arg1, arg2)` | Determine if the first argument is a subclass of the second argument. | [API][0177]
| `issubclass_(arg1, arg2)` | Determine if a class is a subclass of a second class. | [API][0178]
| `find_common_type(array_types, scalar_types)` | Determine common type following standard coercion rules. | [API][0179]

### Miscellaneous

| `typename(char)` | Return a description for the given data type code. | [API][0180]
| `sctype2char(sctype)` | Return the string representation of a scalar dtype. | [API][0181]
| `mintypecode(typechars[, typeset, default])` | Return the character for the minimum-size type to which given types can be safely cast. | [API][0182]


## [Mathematical functions with automatic domain (numpy.emath)][0183]

Note: `numpy.emath` is a preferred alias for `numpy.lib.scimath`, available after `numpy` is imported.

## [Floating point error handling][0184]

### Setting and getting error handling

| `seterr([all, divide, over, under, invalid])` | Set how floating-point errors are handled. | [API][0185] |
| `geterr()` | Get the current way of handling floating-point errors. | [API][0186] |
| `seterrcall(func)` | Set the floating-point error callback function or log object. | [API][0187] |
| `geterrcall()` | Return the current callback function used on floating-point errors. | [API][0188] |
| `errstate(**kwargs)` | Context manager for floating-point error handling. | [API][0189] |

### Internal functions

| `seterrobj(errobj)` | Set the object that defines floating-point error handling. | [API][0190] |
| `geterrobj()` | Return the current object that defines floating-point error handling. | [API][0191] |

## [Discrete Fourier Transform (numpy.fft)][0192]

### Standard FFTs

| `fft(a[, n, axis, norm])` | Compute the one-dimensional discrete Fourier Transform. | [API][0193]
| `ifft(a[, n, axis, norm])` | Compute the one-dimensional inverse discrete Fourier Transform. | [API][0194]
| `fft2(a[, s, axes, norm])` | Compute the 2-dimensional discrete Fourier Transform | [API][0195]
| `ifft2(a[, s, axes, norm])` | Compute the 2-dimensional inverse discrete Fourier Transform. | [API][0196]
| `fftn(a[, s, axes, norm])` | Compute the N-dimensional discrete Fourier Transform. | [API][0197]
| `ifftn(a[, s, axes, norm])` | Compute the N-dimensional inverse discrete Fourier Transform. | [API][0198]

### Real FFTs

| `rfft(a[, n, axis, norm])` | Compute the one-dimensional discrete Fourier Transform for real input. | [API][0199]
| `irfft(a[, n, axis, norm])` | Compute the inverse of the n-point DFT for real input. | [API][0200]
| `rfft2(a[, s, axes, norm])` | Compute the 2-dimensional FFT of a real array. | [API][0201]
| `irfft2(a[, s, axes, norm])` | Compute the 2-dimensional inverse FFT of a real array. | [API][0202]
| `rfftn(a[, s, axes, norm])` | Compute the N-dimensional discrete Fourier Transform for real input. | [API][0203]
| `irfftn(a[, s, axes, norm])` | Compute the inverse of the N-dimensional FFT of real input. | [API][0204]

### Hermitian FFTs

| `hfft(a[, n, axis, norm])` | Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum. | [API][0205]
| `ihfft(a[, n, axis, norm])` | Compute the inverse FFT of a signal that has Hermitian symmetry. | [API][0206]

### Helper routines

| `fftfreq(n[, d])` | Return the Discrete Fourier Transform sample frequencies. | [API][0207]
| `rfftfreq(n[, d])` | Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft). | [API][0208]
| `fftshift(x[, axes])` | Shift the zero-frequency component to the center of the spectrum. | [API][0209]
| `ifftshift(x[, axes])` | The inverse of fftshift. | [API][0210]



--------------------------------------------

[0]:    
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
[0192]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fft.html#numpy.fft.fft
[0193]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft
[0194]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fft2.html#numpy.fft.fft2
[0195]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2
[0196]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fftn.html#numpy.fft.fftn
[0197]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifftn.html#numpy.fft.ifftn
[0198]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft
[0199]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.irfft.html#numpy.fft.irfft
[0200]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfft2.html#numpy.fft.rfft2
[0201]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.irfft2.html#numpy.fft.irfft2
[0202]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfftn.html#numpy.fft.rfftn
[0203]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.irfftn.html#numpy.fft.irfftn
[0204]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.hfft.html#numpy.fft.hfft
[0205]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ihfft.html#numpy.fft.ihfft
[0206]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq
[0207]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.rfftfreq.html#numpy.fft.rfftfreq
[0208]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift
[0209]: https://www.numpy.org/devdocs/reference/generated/numpy.fft.ifftshift.html#numpy.fft.ifftshift
[0210]: 
[0211]: 
[0212]: 
[0213]: 
[0214]: 
[0215]: 
[0216]: 
[0217]: 
[0218]: 
[0219]: 
[0220]: 
[0221]: 
[0222]: 
[0223]: 
[0224]: 
[0225]: 
[0226]: 
[0227]: 
[0228]: 
[0229]: 
[0230]: 
[0231]: 
[0232]: 
[0233]: 
[0234]: 
[0235]: 
[0236]: 
[0237]: 
[0238]: 
[0239]: 
[0240]: 
[0241]: 
[0242]: 
[0243]: 
[0244]: 
[0245]: 
[0246]: 
[0247]: 
[0248]: 
[0249]: 
[0250]: 
[0251]: 
[0252]: 
[0253]: 
[0254]: 
[0255]: 
[0256]: 
[0257]: 
[0258]: 
[0259]: 
[0260]: 
[0261]: 
[0262]: 
[0263]: 
[0264]: 
[0265]: 
[0266]: 
[0267]: 
[0268]: 
[0269]: 
[0270]: 
[0271]: 
[0272]: 
[0273]: 
[0274]: 
[0275]: 
[0276]: 
[0277]: 
[0278]: 
[0279]: 
[0280]: 
[0281]: 
[0282]: 
[0283]: 
[0284]: 
[0285]: 
[0286]: 
[0287]: 
[0288]: 
[0289]: 
[0290]: 
[0291]: 
[0292]: 
[0293]: 
[0294]: 
[0295]: 
[0296]: 
[0297]: 
[0298]: 
[0299]: 
[0300]: 
[0301]: 
[0302]: 
[0303]: 
[0304]: 
[0305]: 
[0306]: 
[0307]: 
[0308]: 
[0309]: 
[0310]: 
[0311]: 
[0312]: 
[0313]: 
[0314]: 
[0315]: 
[0316]: 
[0317]: 
[0318]: 
[0319]: 
[0320]: 
[0321]: 
[0322]: 
[0323]: 
[0324]: 
[0325]: 
[0326]: 
[0327]: 
[0328]: 
[0329]: 
[0330]: 
[0331]: 
[0332]: 
[0333]: 
[0334]: 
[0335]: 
[0336]: 
[0337]: 
[0338]: 
[0339]: 
[0340]: 
[0341]: 
[0342]: 
[0343]: 
[0344]: 
[0345]: 
[0346]: 
[0347]: 
[0348]: 
[0349]: 
[0350]: 
[0351]: 
[0352]: 
[0353]: 
[0354]: 
[0355]: 
[0356]: 
[0357]: 
[0358]: 
[0359]: 
[0360]: 
[0361]: 
[0362]: 
[0363]: 
[0364]: 
[0365]: 
[0366]: 
[0367]: 
[0368]: 
[0369]: 
[0370]: 
[0371]: 
[0372]: 
[0373]: 
[0374]: 
[0375]: 
[0376]: 
[0377]: 
[0378]: 
[0379]: 
[0380]: 
[0381]: 
[0382]: 
[0383]: 
[0384]: 
[0385]: 
[0386]: 
[0387]: 
[0388]: 
[0389]: 
[0390]: 
[0391]: 
[0392]: 
[0393]: 
[0394]: 
[0395]: 
[0396]: 
[0397]: 
[0398]: 
[0399]: 
