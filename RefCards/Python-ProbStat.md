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

## Statistics Distributions

+ `scipy.stats.xxx` class: xxx = bernoulli, binom, poisson, geom<br/><br/>

  <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="../Stats/ProbStatsPython/08-DiscreteDist.md#lecture-notebook-8">Methods for Distribution</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Return Value(s)</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td style="font-weight: bold;">rvs(<code>args1</code>, loc=0, size=1, random_state=None)</td> <td>Random variates.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">pmf(<code>args2</code>, loc=0)</td>
      <td>Probability mass function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">logpmf(<code>args2</code>, loc=0)</td> <td>Log of the probability mass function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">cdf(<code>args2</code>, loc=0)</td> <td>Cumulative distribution function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">logcdf(<code>args2</code>, loc=0)</td> <td>Log of the cumulative distribution function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">sf(<code>args2</code>, loc=0)</td> <td>Survival function (also defined as <code>1 - cdf</code>, but <i>sf</i> is sometimes more accurate).</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">logsf(<code>args2</code>, loc=0)</td> <td>Log of the survival function.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">ppf(<code>args3</code>, loc=0)</td> <td>Percent point function (inverse of <code>cdf</code> — percentiles).</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">isf(<code>args3</code>, loc=0)</td> <td>Inverse survival function (inverse of <code>sf</code>).</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">stats(<code>args1</code>, loc=0, moments='mv')</td> <td>Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">entropy(<code>args1</code>, loc=0)</td> <td>(Differential) entropy of the RV.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">expect(func, args=(<code>args1</code>,), loc=0, lb=None, ub=None, conditional=False)</td> <td>Expected value of a function (of one argument) with respect to the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">median(<code>args1</code>, loc=0)</td> <td>Median of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">mean(<code>args1</code>, loc=0)</td> <td>Mean of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">var(<code>args1</code>, loc=0)</td> <td>Variance of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">std(<code>args1</code>, loc=0)</td> <td>Standard deviation of the distribution.</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">interval(alpha, <code>args1</code>, loc=0)</td> <td>Endpoints of the range that contains alpha percent of the distribution</td>
    </tr>
    <tr>
      <td colspan="2">
        <a href="https://tinyurl.com/y72ffzu9">bernoulli</a>: <code>args1</code> = <code>p</code>, <code>args2</code> = <code>k, p</code>, <code>args3</code> = <code>q, p</code><br/><br/>
        <a href="https://tinyurl.com/y8fdjfpy">binom</a>: <code>args1</code> = <code>n, p</code>, <code>args2</code> = <code>k, n, p</code>, <code>args3</code> = <code>q, n, p</code><br/><br/>
        <a href="https://tinyurl.com/ybx8l3ot">poisson</a>: <code>args1</code> = <code>mu</code>, <code>args2</code> = <code>k, mu</code>, <code>args3</code> = <code>q, mu</code><br/><br/>
        <a href="https://tinyurl.com/y9bbkpnt">geom</a>: <code>args1</code> = <code>p</code>, <code>args2</code> = <code>k, p</code>, <code>args3</code> = <code>q, p</code>
      </td>
    </tr>
    </tbody>
  </table>

## Numpy: common used functions

+ [function](../Stats/ProbStatsPython/08-DiscreteDist.md#lecture-notebook-8): `numpy.linspace`
  + Syntax: `numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`
  + Docstring
    + Return evenly spaced numbers over a specified interval.
    + Returns num evenly spaced samples, calculated over the interval [start, stop].
    + The endpoint of the interval can optionally be excluded.

+ [function](../Stats/ProbStatsPython/08-DiscreteDist.md#lecture-notebook-8): `numpy.histogram`
  + Syntax: `numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)`
  + Docstring: Compute the histogram of a set of data.

+ [function](../Stats/ProbStatsPython/08-DiscreteDist.md#lecture-notebook-8): `numpy.histogram_bin_edges`
  + Syntax: `numpy.histogram_bin_edges(a, bins=10, range=None, weights=None)`
  + Docstring: Function to calculate only the edges of the bins used by the histogram function.



## Pandas: common used functions

+ [function](../Stats/ProbStatsPython/08-DiscreteDist.md#lecture-notebook-8): `Series.value_counts`
  + Syntax: `Series.value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True)`
  + Docstring
    + Return a Series containing counts of unique values.
    + The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.



