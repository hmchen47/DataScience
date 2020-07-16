# Commonly Used Python Modules for Probability & Statistics

## Sets

+ [Set definition](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + define a set: `{...}` or `set(...)`
    + e.g., `Set1 = {1, 2}; print(Set1) $ {1, 2}`, `Set2 = set({2, 3}); print(Set2) ${2, 3}`
  + empty set: using only `set()` or `set({})`
    + e.g., `Empty1 = set(); type(Empty1) # set; print(Empty1) # set{}`
    + e.g., `Empty2 = set({}); type(Empty2) # set; print(Empty2) # set{}`
    + e.g., `NotASet = {}; type(NotASet) # dict`, `{}` not an empty set

+ [Membership](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + $\in$: `in`
  + $\notin$: `not in`

+ [Testing if empty set, size](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + test empty: `not`
  + size: `len()`
  + check if size is 0: `len() == 0`

+ [Intervals and Multiples](../Stats/ProbStatsPython/02-Sets.md#22-basic-sets)
  + $\{0, \dots, n-1\}$: `range(n)`
  + $\{m, \dots, n-1\}$: `range(m, n)`
  + $\{m,\, m+d,\, m+2d,\, \dots\} \leq n-1$: `range(m, n, d)`

+ [Set relationship](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + equality ($= \;\to$ `==`)
  + inequality ($\neq \;\to$ `!=`)
  + disjoint (`isdisjoint`)

+ [Subsets and supersets](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + subset ($\subseteq\; \to$ `<=` or `issubset`)
  + supserset ($\supseteq\; \to$ `>=` or `issuperset`)
  + strict subset ($\subset\; \to$ `<`)
  + strict supeerset ($\supset\; \to$ `>`)

+ [Union and Intersection](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + union ($\cup \to$ `|` or `union`)
  + intersection ($\cap \to$ `&` or `intersection`)

+ [Set- and Symmetric-Difference](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + set difference ($- \to$ `-` or `difference`)
  + symmetric difference ($\Delta \to$ `^` or `symmetric_difference`)

+ [Set operations - Summary](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + complement: $A^c$
  + intersection: $\cap \to$ `&` or `intersection`
  + union: $\cup \to$ `|` or `union`
  + difference: $- \to$ `-` or `difference`
  + symmetric difference: $\Delta \to$ `^` or `symmetric_difference`

+ [Cartesian products](../Stats/ProbStatsPython/02-Sets.md#26-cartesian-products)
  + tuples: $(a_1, \dots, a_n)$
  + ordered pairs: $(a, b)$
  + Python: `product` functon in `itertools` library

+ [Set size](../Stats/ProbStatsPython/03-Counting.md#31-counting)
  + size: `len`, e.g., `print(len({-1, 1})) # 2`
  + sum: `sum`, e.g., `print(sum({-1, 1})) # 0`
  + minimum: `min`, e.g., `print(min({-1, 1})) # -1`
  + maximum: `max`, e.g., `print(max({-1, 1})) # 1`
  + loops: `for <var> in <set>`

+ [Cartesian powers and exponential](../Stats/ProbStatsPython/03-Counting.md#35-cartesian-powers)
  + Cartesian: using `product` function in `itertools` library
  + exponential: `**`





## Numpy Math Related Functions

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y8ustxop">Sums, products, differences</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.prod.html#numpy.prod" title="numpy.prod"><code>prod</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;keepdims,&nbsp;…])</p></td>
      <td><p>Return the product of array elements over a given axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.sum.html#numpy.sum" title="numpy.sum"><code>sum</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;keepdims,&nbsp;…])</p></td>
      <td><p>Sum of array elements over a given axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanprod.html#numpy.nanprod" title="numpy.nanprod"><code>nanprod</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;keepdims])</p></td>
      <td><p>Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nansum.html#numpy.nansum" title="numpy.nansum"><code>nansum</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;keepdims])</p></td>
      <td><p>Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.cumprod.html#numpy.cumprod" title="numpy.cumprod"><code>cumprod</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out])</p></td>
      <td><p>Return the cumulative product of elements along a given axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.cumsum.html#numpy.cumsum" title="numpy.cumsum"><code>cumsum</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out])</p></td>
      <td><p>Return the cumulative sum of the elements along a given axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nancumprod.html#numpy.nancumprod" title="numpy.nancumprod"><code>nancumprod</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out])</p></td>
      <td><p>Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nancumsum.html#numpy.nancumsum" title="numpy.nancumsum"><code>nancumsum</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out])</p></td>
      <td><p>Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.diff.html#numpy.diff" title="numpy.diff"><code>diff</code></a>(a[,&nbsp;n,&nbsp;axis,&nbsp;prepend,&nbsp;append])</p></td>
      <td><p>Calculate the n-th discrete difference along the given axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.ediff1d.html#numpy.ediff1d" title="numpy.ediff1d"><code>ediff1d</code></a>(ary[,&nbsp;to_end,&nbsp;to_begin])</p></td>
      <td><p>The differences between consecutive elements of an array.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.gradient.html#numpy.gradient" title="numpy.gradient"><code>gradient</code></a>(f,&nbsp;\*varargs[,&nbsp;axis,&nbsp;edge_order])</p></td>
      <td><p>Return the gradient of an N-dimensional array.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.cross.html#numpy.cross" title="numpy.cross"><code>cross</code></a>(a,&nbsp;b[,&nbsp;axisa,&nbsp;axisb,&nbsp;axisc,&nbsp;axis])</p></td>
      <td><p>Return the cross product of two (arrays of) vectors.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.trapz.html#numpy.trapz" title="numpy.trapz"><code>trapz</code></a>(y[,&nbsp;x,&nbsp;dx,&nbsp;axis])</p></td>
      <td><p>Integrate along the given axis using the composite trapezoidal rule.</p></td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y9k56ajb">Exponents and logarithms</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.exp.html#numpy.exp" title="numpy.exp"><code >exp</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Calculate the exponential of all elements in the input array.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.expm1.html#numpy.expm1" title="numpy.expm1"><code >expm1</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Calculate <code class="docutils literal notranslate"><span class="pre">exp(x) <span class="pre">- <span class="pre">1</code> for all elements in the array.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.exp2.html#numpy.exp2" title="numpy.exp2"><code >exp2</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Calculate <em class="xref py py-obj">2**p</em> for all <em class="xref py py-obj">p</em> in the input array.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.log.html#numpy.log" title="numpy.log"><code >log</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Natural logarithm, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.log10.html#numpy.log10" title="numpy.log10"><code >log10</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Return the base 10 logarithm of the input array, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.log2.html#numpy.log2" title="numpy.log2"><code >log2</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Base-2 logarithm of <em class="xref py py-obj">x</em>.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.log1p.html#numpy.log1p" title="numpy.log1p"><code >log1p</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Return the natural logarithm of one plus the input array, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.logaddexp.html#numpy.logaddexp" title="numpy.logaddexp"><code >logaddexp</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Logarithm of the sum of exponentiations of the inputs.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.logaddexp2.html#numpy.logaddexp2" title="numpy.logaddexp2"><code >logaddexp2</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Logarithm of the sum of exponentiations of the inputs in base-2.</p></td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y8eygfus">Arithmetic operations</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.add.html#numpy.add" title="numpy.add"><code>add</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Add arguments element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.reciprocal.html#numpy.reciprocal" title="numpy.reciprocal"><code>reciprocal</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Return the reciprocal of the argument, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.positive.html#numpy.positive" title="numpy.positive"><code>positive</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Numerical positive, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.negative.html#numpy.negative" title="numpy.negative"><code>negative</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Numerical negative, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.multiply.html#numpy.multiply" title="numpy.multiply"><code>multiply</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Multiply arguments element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.divide.html#numpy.divide" title="numpy.divide"><code>divide</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Returns a true division of the inputs, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.power.html#numpy.power" title="numpy.power"><code>power</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>First array elements raised to powers from second array, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.subtract.html#numpy.subtract" title="numpy.subtract"><code>subtract</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Subtract arguments, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.true_divide.html#numpy.true_divide" title="numpy.true_divide"><code>true_divide</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;…])</p></td>
      <td><p>Returns a true division of the inputs, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.floor_divide.html#numpy.floor_divide" title="numpy.floor_divide"><code>floor_divide</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;…])</p></td>
      <td><p>Return the largest integer smaller or equal to the division of the inputs.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.float_power.html#numpy.float_power" title="numpy.float_power"><code>float_power</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;…])</p></td>
      <td><p>First array elements raised to powers from second array, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.fmod.html#numpy.fmod" title="numpy.fmod"><code>fmod</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Return the element-wise remainder of division.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.mod.html#numpy.mod" title="numpy.mod"><code>mod</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Return element-wise remainder of division.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.modf.html#numpy.modf" title="numpy.modf"><code>modf</code></a>(x[,&nbsp;out1,&nbsp;out2],&nbsp;/&nbsp;[[,&nbsp;out,&nbsp;where,&nbsp;…])</p></td>
      <td><p>Return the fractional and integral parts of an array, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.remainder.html#numpy.remainder" title="numpy.remainder"><code>remainder</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Return element-wise remainder of division.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.divmod.html#numpy.divmod" title="numpy.divmod"><code>divmod</code></a>(x1,&nbsp;x2[,&nbsp;out1,&nbsp;out2],&nbsp;/&nbsp;[[,&nbsp;out,&nbsp;…])</p></td>
      <td><p>Return element-wise quotient and remainder simultaneously.</p></td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y9aqcrsx">Miscellaneous</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.convolve.html#numpy.convolve" title="numpy.convolve"><code>convolve</code></a>(a,&nbsp;v[,&nbsp;mode])</p></td>
      <td><p>Returns the discrete, linear convolution of two one-dimensional sequences.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.clip.html#numpy.clip" title="numpy.clip"><code>clip</code></a>(a,&nbsp;a_min,&nbsp;a_max[,&nbsp;out])</p></td>
      <td><p>Clip (limit) the values in an array.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.sqrt.html#numpy.sqrt" title="numpy.sqrt"><code>sqrt</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Return the non-negative square-root of an array, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.cbrt.html#numpy.cbrt" title="numpy.cbrt"><code>cbrt</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Return the cube-root of an array, element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.square.html#numpy.square" title="numpy.square"><code>square</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Return the element-wise square of the input.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.absolute.html#numpy.absolute" title="numpy.absolute"><code>absolute</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Calculate the absolute value element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.fabs.html#numpy.fabs" title="numpy.fabs"><code>fabs</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Compute the absolute values element-wise.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.sign.html#numpy.sign" title="numpy.sign"><code>sign</code></a>(x,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;order,&nbsp;…])</p></td>
      <td><p>Returns an element-wise indication of the sign of a number.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.heaviside.html#numpy.heaviside" title="numpy.heaviside"><code>heaviside</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Compute the Heaviside step function.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.maximum.html#numpy.maximum" title="numpy.maximum"><code>maximum</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Element-wise maximum of array elements.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.minimum.html#numpy.minimum" title="numpy.minimum"><code>minimum</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Element-wise minimum of array elements.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.fmax.html#numpy.fmax" title="numpy.fmax"><code>fmax</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Element-wise maximum of array elements.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.fmin.html#numpy.fmin" title="numpy.fmin"><code>fmin</code></a>(x1,&nbsp;x2,&nbsp;/[,&nbsp;out,&nbsp;where,&nbsp;casting,&nbsp;…])</p></td>
      <td><p>Element-wise minimum of array elements.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nan_to_num.html#numpy.nan_to_num" title="numpy.nan_to_num"><code>nan_to_num</code></a>(x[,&nbsp;copy,&nbsp;nan,&nbsp;posinf,&nbsp;neginf])</p></td>
      <td><p>Replace NaN with zero and infinity with large finite numbers (default behaviour) or with the numbers defined by the user using the <a href="constants.html#numpy.nan" title="numpy.nan"><code>nan</code></a>,  <em class="xref py py-obj">posinf</em> and/or <em class="xref py py-obj">neginf</em> keywords.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.real_if_close.html#numpy.real_if_close" title="numpy.real_if_close"><code>real_if_close</code></a>(a[,&nbsp;tol])</p></td>
      <td><p>If input is complex with all imaginary parts close to zero, return  real parts.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.interp.html#numpy.interp" title="numpy.interp"><code>interp</code></a>(x,&nbsp;xp,&nbsp;fp[,&nbsp;left,&nbsp;right,&nbsp;period])</p></td>
      <td><p>One-dimensional linear interpolation.</p></td>
    </tr>
    </tbody>
  </table>

