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


## Random Sampling

+ Random sampling (`numpy.random`)<br/><br/>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y7djwhnh">Simple Random Data</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
      <tr><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.rand.html#numpy.random.rand" title="numpy.random.rand">rand</a>(d0,&nbsp;d1,&nbsp;...,&nbsp;dn)</td>
      <td>Random values in a given shape.</td>
      </tr>
      <tr class="row-even"><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.randn.html#numpy.random.randn" title="numpy.random.randn">randn</a>(d0,&nbsp;d1,&nbsp;...,&nbsp;dn)</td>
      <td>Return a sample (or samples) from the “standard normal” distribution.</td>
      </tr>
      <tr><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.randint.html#numpy.random.randint" title="numpy.random.randint">randint</a>(low[,&nbsp;high,&nbsp;size])</td>
      <td>Return random integers from <em class="xref py py-obj">low</em> (inclusive) to <em class="xref py py-obj">high</em> (exclusive).</td>
      </tr>
      <tr class="row-even"><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.random_integers.html#numpy.random.random_integers" title="numpy.random.random_integers">random_integers</a>(low[,&nbsp;high,&nbsp;size])</td>
      <td>Return random integers between <em class="xref py py-obj">low</em> and <em class="xref py py-obj">high</em>, inclusive.</td>
      </tr>
      <tr><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample" title="numpy.random.random_sample">random_sample</a>([size])</td>
      <td>Return random floats in the half-open interval [0.0, 1.0).</td>
      </tr>
      <tr class="row-even"><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.random.html#numpy.random.random" title="numpy.random.random">random</a>([size])</td>
      <td>Return random floats in the half-open interval [0.0, 1.0).</td>
      </tr>
      <tr><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.ranf.html#numpy.random.ranf" title="numpy.random.ranf">ranf</a>([size])</td>
      <td>Return random floats in the half-open interval [0.0, 1.0).</td>
      </tr>
      <tr class="row-even"><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.sample.html#numpy.random.sample" title="numpy.random.sample">sample</a>([size])</td>
      <td>Return random floats in the half-open interval [0.0, 1.0).</td>
      </tr>
      <tr><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.choice.html#numpy.random.choice" title="numpy.random.choice">choice</a>(a[,&nbsp;size,&nbsp;replace,&nbsp;p])</td>
      <td>Generates a random sample from a given 1-D array</td>
      </tr>
      <tr class="row-even"><td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.bytes.html#numpy.random.bytes" title="numpy.random.bytes">bytes</a>(length)</td>
      <td>Return random bytes.</td>
      </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y7djwhnh">Distributions</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.beta.html#numpy.random.beta" title="numpy.random.beta">beta</span></a>(a,&nbsp;b[,&nbsp;size])</td> <td>The Beta distribution over <code>[0, 1]</code>.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.binomial.html#numpy.random.binomial" title="numpy.random.binomial">binomial</a>(n,&nbsp;p[,&nbsp;size])</td> <td>Draw samples from a binomial distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.chisquare.html#numpy.random.chisquare" title="numpy.random.chisquare">chisquare</a>(df[,&nbsp;size])</td> <td>Draw samples from a chi-square distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.dirichlet.html#numpy.random.dirichlet" title="numpy.random.dirichlet">dirichlet</a>(alpha[,&nbsp;size])</td> <td>Draw samples from the Dirichlet distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.exponential.html#numpy.random.exponential" title="numpy.random.exponential">exponential</a>([scale,&nbsp;size])</td> <td>Exponential distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.f.html#numpy.random.f" title="numpy.random.f">f</a>(dfnum,&nbsp;dfden[,&nbsp;size])</td> <td>Draw samples from a F distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.gamma.html#numpy.random.gamma" title="numpy.random.gamma">gamma</a>(shape[,&nbsp;scale,&nbsp;size])</td> <td>Draw samples from a Gamma distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.geometric.html#numpy.random.geometric" title="numpy.random.geometric">geometric</a>(p[,&nbsp;size])</td> <td>Draw samples from the geometric distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.gumbel.html#numpy.random.gumbel" title="numpy.random.gumbel">gumbel</a>([loc,&nbsp;scale,&nbsp;size])</td> <td>Gumbel distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.hypergeometric.html#numpy.random.hypergeometric" title="numpy.random.hypergeometric">hypergeometric</a><br/>&nbsp;&nbsp;&nbsp;&nbsp(ngood,&nbsp;nbad,&nbsp;nsample[,&nbsp;size])</td> <td>Draw samples from a Hypergeometric distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.laplace.html#numpy.random.laplace" title="numpy.random.laplace">laplace</a>([loc,&nbsp;scale,&nbsp;size])</td> <td>Draw samples from the Laplace or double exponential distribution with specified location (or mean) and scale (decay).</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.logistic.html#numpy.random.logistic" title="numpy.random.logistic">logistic</a>([loc,&nbsp;scale,&nbsp;size])</td> <td>Draw samples from a Logistic distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.lognormal.html#numpy.random.lognormal" title="numpy.random.lognormal">lognormal</a>([mean,&nbsp;sigma,&nbsp;size])</td> <td>Return samples drawn from a log-normal distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.logseries.html#numpy.random.logseries" title="numpy.random.logseries">logseries</a>(p[,&nbsp;size])</td> <td>Draw samples from a Logarithmic Series distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.multinomial.html#numpy.random.multinomial" title="numpy.random.multinomial">multinomial</a>(n,&nbsp;pvals[,&nbsp;size])</td> <td>Draw samples from a multinomial distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.multivariate_normal.html#numpy.random.multivariate_normal" title="numpy.random.multivariate_normal">multivariate_normal</a>(mean,&nbsp;cov[,&nbsp;size])</td> <td>Draw random samples from a multivariate normal distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial" title="numpy.random.negative_binomial">negative_binomial</a>(n,&nbsp;p[,&nbsp;size])</td> <td>Draw samples from a negative_binomial distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare" title="numpy.random.noncentral_chisquare">noncentral_chisquare</a>(df,&nbsp;nonc[,&nbsp;size])</td> <td>Draw samples from a noncentral chi-square distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.noncentral_f.html#numpy.random.noncentral_f" title="numpy.random.noncentral_f">noncentral_f</a>(dfnum,&nbsp;dfden,&nbsp;nonc[,&nbsp;size])</td> <td>Draw samples from the noncentral F distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.normal.html#numpy.random.normal" title="numpy.random.normal">normal</a>([loc,&nbsp;scale,&nbsp;size])</td> <td>Draw random samples from a normal (Gaussian) distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.pareto.html#numpy.random.pareto" title="numpy.random.pareto">pareto</a>(a[,&nbsp;size])</td> <td>Draw samples from a Pareto II or Lomax distribution with specified shape.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.poisson.html#numpy.random.poisson" title="numpy.random.poisson">poisson</a>([lam,&nbsp;size])</td> <td>Draw samples from a Poisson distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.power.html#numpy.random.power" title="numpy.random.power">power</a>(a[,&nbsp;size])</td> <td>Draws samples in [0, 1] from a power distribution with positive exponent a - 1.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.rayleigh.html#numpy.random.rayleigh" title="numpy.random.rayleigh">rayleigh</a>([scale,&nbsp;size])</td> <td>Draw samples from a Rayleigh distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.standard_cauchy.html#numpy.random.standard_cauchy" title="numpy.random.standard_cauchy">standard_cauchy</a>([size])</td> <td>Standard Cauchy distribution with mode = 0.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.standard_exponential.html#numpy.random.standard_exponential" title="numpy.random.standard_exponential">standard_exponential</a>([size])</td> <td>Draw samples from the standard exponential distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.standard_gamma.html#numpy.random.standard_gamma" title="numpy.random.standard_gamma">standard_gamma</a>(shape[,&nbsp;size])</td> <td>Draw samples from a Standard Gamma distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.standard_normal.html#numpy.random.standard_normal" title="numpy.random.standard_normal">standard_normal</a>([size])</td> <td>Returns samples from a Standard Normal distribution (mean=0, stdev=1).</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.standard_t.html#numpy.random.standard_t" title="numpy.random.standard_t">standard_t</a>(df[,&nbsp;size])</td> <td>Standard Student’s t distribution with df degrees of freedom.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.triangular.html#numpy.random.triangular" title="numpy.random.triangular">triangular</a>(left,&nbsp;mode,&nbsp;right[,&nbsp;size])</td> <td>Draw samples from the triangular distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform" title="numpy.random.uniform">uniform</a>([low,&nbsp;high,&nbsp;size])</td> <td>Draw samples from a uniform distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.vonmises.html#numpy.random.vonmises" title="numpy.random.vonmises">vonmises</a>(mu,&nbsp;kappa[,&nbsp;size])</td> <td>Draw samples from a von Mises distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.wald.html#numpy.random.wald" title="numpy.random.wald">wald</a>(mean,&nbsp;scale[,&nbsp;size])</td> <td>Draw samples from a Wald, or Inverse Gaussian, distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.weibull.html#numpy.random.weibull" title="numpy.random.weibull">weibull</a>(a[,&nbsp;size])</td> <td>Weibull distribution.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.zipf.html#numpy.random.zipf" title="numpy.random.zipf">zipf</a>(a[,&nbsp;size])</td> <td>Draw samples from a Zipf distribution.</td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y7djwhnh">Random Generator</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState" title="numpy.random.RandomState">RandomState</a></td> <td>Container for the Mersenne Twister pseudo-random number generator.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.seed.html#numpy.random.seed" title="numpy.random.seed">seed</a>([seed])</td> <td>Seed the generator.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.get_state.html#numpy.random.get_state" title="numpy.random.get_state">get_state</a>()</td> <td>Return a tuple representing the internal state of the generator.</td>
    </tr>
    <tr>
      <td><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.set_state.html#numpy.random.set_state" title="numpy.random.set_state">set_state</a>(state)</td> <td>Set the internal state of the generator from a tuple.</td>
    </tr>
    </tbody>
  </table>



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



## Numpy Statistic Functions

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y79873lp">Order statistics</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.amin.html#numpy.amin" title="numpy.amin"><code>amin</code></a>(a[,&nbsp;axis,&nbsp;out,&nbsp;keepdims,&nbsp;initial,&nbsp;where])</p></td>
      <td><p>Return the minimum of an array or minimum along an axis.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.amax.html#numpy.amax" title="numpy.amax"><code>amax</code></a>(a[,&nbsp;axis,&nbsp;out,&nbsp;keepdims,&nbsp;initial,&nbsp;where])</p></td>
      <td><p>Return the maximum of an array or maximum along an axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanmin.html#numpy.nanmin" title="numpy.nanmin"><code>nanmin</code></a>(a[,&nbsp;axis,&nbsp;out,&nbsp;keepdims])</p></td>
      <td><p>Return minimum of an array or minimum along an axis, ignoring any NaNs.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanmax.html#numpy.nanmax" title="numpy.nanmax"><code>nanmax</code></a>(a[,&nbsp;axis,&nbsp;out,&nbsp;keepdims])</p></td>
      <td><p>Return the maximum of an array or maximum along an axis, ignoring any NaNs.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.ptp.html#numpy.ptp" title="numpy.ptp"><code>ptp</code></a>(a[,&nbsp;axis,&nbsp;out,&nbsp;keepdims])</p></td>
      <td><p>Range of values (maximum - minimum) along an axis.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.percentile.html#numpy.percentile" title="numpy.percentile"><code>percentile</code></a>(a,&nbsp;q[,&nbsp;axis,&nbsp;out,&nbsp;…])</p></td>
      <td><p>Compute the q-th percentile of the data along the specified axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanpercentile.html#numpy.nanpercentile" title="numpy.nanpercentile"><code>nanpercentile</code></a>(a,&nbsp;q[,&nbsp;axis,&nbsp;out,&nbsp;…])</p></td>
      <td><p>Compute the qth percentile of the data along the specified axis, while ignoring nan values.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.quantile.html#numpy.quantile" title="numpy.quantile"><code>quantile</code></a>(a,&nbsp;q[,&nbsp;axis,&nbsp;out,&nbsp;overwrite_input,&nbsp;…])</p></td>
      <td><p>Compute the q-th quantile of the data along the specified axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanquantile.html#numpy.nanquantile" title="numpy.nanquantile"><code>nanquantile</code></a>(a,&nbsp;q[,&nbsp;axis,&nbsp;out,&nbsp;…])</p></td>
      <td><p>Compute the qth quantile of the data along the specified axis, while ignoring nan values.</p></td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y8ustxop">Averages and variances</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.median.html#numpy.median" title="numpy.median"><code>median</code></a>(a[,&nbsp;axis,&nbsp;out,&nbsp;overwrite_input,&nbsp;keepdims])</p></td>
      <td><p>Compute the median along the specified axis.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.average.html#numpy.average" title="numpy.average"><code>average</code></a>(a[,&nbsp;axis,&nbsp;weights,&nbsp;returned])</p></td>
      <td><p>Compute the weighted average along the specified axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.mean.html#numpy.mean" title="numpy.mean"><code>mean</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;keepdims])</p></td>
      <td><p>Compute the arithmetic mean along the specified axis.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.std.html#numpy.std" title="numpy.std"><code>std</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;ddof,&nbsp;keepdims])</p></td>
      <td><p>Compute the standard deviation along the specified axis.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.var.html#numpy.var" title="numpy.var"><code>var</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;ddof,&nbsp;keepdims])</p></td>
      <td><p>Compute the variance along the specified axis.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanmedian.html#numpy.nanmedian" title="numpy.nanmedian"><code>nanmedian</code></a>(a[,&nbsp;axis,&nbsp;out,&nbsp;overwrite_input,&nbsp;…])</p></td>
      <td><p>Compute the median along the specified axis, while ignoring NaNs.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanmean.html#numpy.nanmean" title="numpy.nanmean"><code>nanmean</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;keepdims])</p></td>
      <td><p>Compute the arithmetic mean along the specified axis, ignoring NaNs.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanstd.html#numpy.nanstd" title="numpy.nanstd"><code>nanstd</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;ddof,&nbsp;keepdims])</p></td>
      <td><p>Compute the standard deviation along the specified axis, while ignoring NaNs.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.nanvar.html#numpy.nanvar" title="numpy.nanvar"><code>nanvar</code></a>(a[,&nbsp;axis,&nbsp;dtype,&nbsp;out,&nbsp;ddof,&nbsp;keepdims])</p></td>
      <td><p>Compute the variance along the specified axis, while ignoring NaNs.</p></td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/yc6sbejm">Correlating</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.corrcoef.html#numpy.corrcoef" title="numpy.corrcoef"><code>corrcoef</code></a>(x[,&nbsp;y,&nbsp;rowvar,&nbsp;bias,&nbsp;ddof])</p></td>
      <td><p>Return Pearson product-moment correlation coefficients.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.correlate.html#numpy.correlate" title="numpy.correlate"><code>correlate</code></a>(a,&nbsp;v[,&nbsp;mode])</p></td>
      <td><p>Cross-correlation of two 1-dimensional sequences.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.cov.html#numpy.cov" title="numpy.cov"><code>cov</code></a>(m[,&nbsp;y,&nbsp;rowvar,&nbsp;bias,&nbsp;ddof,&nbsp;fweights,&nbsp;…])</p></td>
      <td><p>Estimate a covariance matrix, given data and weights.</p></td>
    </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 55vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://tinyurl.com/y7vdpd8f">Histograms</a></caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Methods</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.histogram.html#numpy.histogram" title="numpy.histogram"><code>histogram</code></a>(a[,&nbsp;bins,&nbsp;range,&nbsp;normed,&nbsp;weights,&nbsp;…])</p></td>
      <td><p>Compute the histogram of a set of data.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.histogram2d.html#numpy.histogram2d" title="numpy.histogram2d"><code>histogram2d</code></a>(x,&nbsp;y[,&nbsp;bins,&nbsp;range,&nbsp;normed,&nbsp;…])</p></td>
      <td><p>Compute the bi-dimensional histogram of two data samples.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.histogramdd.html#numpy.histogramdd" title="numpy.histogramdd"><code>histogramdd</code></a>(sample[,&nbsp;bins,&nbsp;range,&nbsp;normed,&nbsp;…])</p></td>
      <td><p>Compute the multidimensional histogram of some data.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.bincount.html#numpy.bincount" title="numpy.bincount"><code>bincount</code></a>(x[,&nbsp;weights,&nbsp;minlength])</p></td>
      <td><p>Count number of occurrences of each value in array of non-negative ints.</p></td>
    </tr>
    <tr>
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges" title="numpy.histogram_bin_edges"><code>histogram_bin_edges</code></a>(a[,&nbsp;bins,&nbsp;range,&nbsp;weights])</p></td>
      <td><p>Function to calculate only the edges of the bins used by the <a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.histogram.html#numpy.histogram" title="numpy.histogram"><code>histogram</code></a> function.</p></td>
    </tr>
    <tr >
      <td><p><a href="https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.digitize.html#numpy.digitize" title="numpy.digitize"><code>digitize</code></a>(x,&nbsp;bins[,&nbsp;right])</p></td>
      <td><p>Return the indices of the bins to which each value in input array belongs.</p></td>
    </tr>
    </tbody>
  </table>




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

