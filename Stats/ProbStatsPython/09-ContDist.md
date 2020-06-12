# Topic 9: Continuous Distribution Families


## 9.1 Continuous Distributions

+ Discrete to continuous
  + discrete distribution: countable \# values (finite or countably-infinite)
  + continuous distributions: uncountable \# values, intervals

+ Motivation
  + anything physics
    + time: flight, delivery, disease, life
    + space: height, storm area
    + mass: pet, cookie
    + temperature: air, body
  + nearly continuous variables
    + cost; stock, house, pork bellies
    + rates: interest, exchange, unemployment

+ Probability density function (pdf)
  + replacing the discrete pmf
  + relative likelihood of $x$: $f(x) \ge 0$
  + area under curve (area): $\int_{-\infty}^\infty f(x)\,dx$

+ Comparison

  <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;"></th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Discrete</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Continuous</th>
    </tr>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Prob. Func.</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">mass (pmf)</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">density (pdf)</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">$\ge 0$</td> <td style="text-align: center;">$p(x) \ge 0$</td> <td style="text-align: center;">$f(x) \ge 0$</td>  </tr>
    <tr> <td style="text-align: center;">$\sum = 1$</td> <td style="text-align: center;">$\sum_x p(x) = 1$</td> <td style="text-align: center;">$\int_{-\infty}^\infty f(x) dx = 1$$</td> </tr>
    </tbody>
  </table>

+ Event probability
  + discrete: $P(A) = \sum_{x\in A} p(x)$
  + continuous: $P(A) = \int_{x \in A} f(x) dx$
  + typically interested in interval probability: $\Pr(a \le X \le b)$
  + AuC (area under curve) btw a and b
  + cumulative distribution function: $\Pr(X \le b) - \Pr(X \le a)$

+ Cumulative distribution function (CDF)

  <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <caption style="font-size: 1.0em; margin: 0.2em;">$F(X) \triangleq \Pr(X \le x)$</caption>
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;"></th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Discrete</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Continuous</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">PF $\to$ CDF</td> <td style="text-align: center;">$\displaystyle\sum_{u \le x} p(u)$</td> <td style="text-align: center;">$\int_{-\infty}^x f(u) du$</td> </tr>
    <tr> <td style="text-align: center;">CDF $\to$ PF</td> <td style="text-align: center;">$p(x) = F(x) - F(x^\ast)$</td> <td style="text-align: center;">$f(x) = F^\prime(x)$</td> </tr>
    <tr><td colspan="3">$x^\ast$: element preceding $x$</td></tr>
    </tbody>
  </table>

  + properties
    + $F(x) = $ integral
    + nondecreasing
    + $F(-\infty) = 0$
    + $F(\infty) = 1$
    + continuous

+ Example: uniform distribution
  + pdf

    \[ f(x) = \begin{cases} 1 & 0 \le x \le 1 \\ 0 & \text{otherwise} \end{cases} \]

  + unitary: will it $\sum? \qquad \text{A.U.C. } = 1 \cdot 1 = 1$

    \[ \int_{-\infty}^\infty f(x) dx = \int^1_0 1 dx = \left. x\right|^1_0 = 1 \]

  + CDF

    \[ F(x) = \int_{-\infty}^\infty  f(u) du = \begin{cases} 0 & x \le 0 \\ \int_0^x 1\,du = \left. u\right|_0^x = x & 0 \le x \le 1 \\ 1 & 1 \le x \end{cases} \]

    \[ F^\prime = \begin{cases} (0)^\prime = 0 & 0 \le x \\ (x)^\prime = 1 & 0 < x < 1 \\ (1)^\prime = 0 & 1 < x \end{cases} \]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yb4obz4o" ismap target="_blank">
        <img src="img/t09-02a.png" style="margin: 0.1em;" alt="PDF of Uniform Distribution" title="PDF of UNiform Distribution" height=150>
        <img src="img/t09-02b.png" style="margin: 0.1em;" alt="CDF of Uniform Distribution" title="CDF of UNiform Distribution" height=150>
      </a>
    </div>

+ Example: triangle
  + pdf

  \[ f(x) = \begin{cases} 2x & 0 \le x \le 1 \\ 0 & \text{otherwise} \end{cases} \]

  + unitary: will it $\sum? \qquad \text{Area under curve } = 2 \cdot 1 \cdot \frac12 = 1$

    \[ \int_{-\infty}^{\infty} f(x) dx = \int_0^1 2x\,dx = \left.x^2 \right|_0^1 = 1 - 0 = 1 \]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yb4obz4o" ismap target="_blank">
      <img src="img/t09-01.png" style="margin: 0.1em;" alt="Example PDF of Triangular Distribution" title="Example PDF of Triangular Distribution" width=150>
    </a>
  </div>

  + CDF

    \[ F(x) = \int_{-\infty}^x f(u)\,du = \begin{cases} 0 & x \le 0 \\ \int_0^x 2u\,du = \left.u^2 \right|_0^x = x^2 = f(x) & 0 \le x \le 1 \\ 1 & 1 \le x \end{cases} \]

    \[ F^\prime(x) = \begin{cases} (0)^\prime = 0 & x < 0 \\ (x^2)^\prime = 2x & 0 \le x \le 1 \\ (1)^\prime = 0 & 1 < x \end{cases} \]


+ Infinite support
  + power paw distribution: pdf

    \[ f(x) = \begin{cases} \frac{1}{x^2} & x \ge 1 \\ 0 & x < 1 \end{cases} \]

  + unitary

    \[ \int_{-\infty}^\infty f(x) dx = \int_1^\infty \frac{1}{u^2}\, du = \left.\frac{-1}{u} \right|_1^\infty = 1 \]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/ycl3n8dg" ismap target="_blank">
      <img src="https://tinyurl.com/y9ryy9a7" style="margin: 0.1em;" alt="Human height follows a power-law distribution. The blue region shows this power-law distribution, and compares it to the actual normal distribution of human height (red)." title="Human height follows a power-law distribution" width=350>
    </a>
  </div>

  + CDF

    \[ F(x) = \begin{cases} 0 & x \le 1 \\ \int_1^x \frac{1}{u^2} \,du = \left.\frac{-1}{u} \right|_1^x = 1 -\frac 1 x & x \ge 1 \end{cases} \]

    \[ F^\prime (x) = \begin{cases} (0)^\prime = 0 & x < 1 \\ (1 - \frac1 x)^\prime = \frac{1}{x^2} = f(x) & x \ge 1 \end{cases} \]

+ Interval probability

  \[ \Pr(a, b) = \Pr([a, b)) = \Pr((a, b]) = F(b) - F(a) \]

  + examples
    + uniform: $0 \le a \le b \le 1$

      \[ \Pr(a \le X \le b) = \begin{cases} \text{AuC } = (b-a) \cdot 1 = b-a \\ \int_a^b f(x)\,dx = \int_a^b 1\,dx = \left. x \right|_a^b = b-a \\ F(b) - F(a) = b-a \end{cases} \]

      + $\Pr(0.6 \le X \le 1.3) = \Pr(0.6 \le X \le 1) = 0.4$
      + $\Pr(0.6 \le X \le 1.3) = F(1.3) - F(0.6) = 1 - 0.6 = 0.4$
    + power law: $1 \le a \le b$

      \[ \Pr(a \le X \le b) = F(b) - F(a) = (1 - \frac 1 b) - (1 - \frac 1 a) = \frac1 a  - \frac 1 b \]

+ Differences between discrete and continuous

  <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Discrete</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Continuous</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">$p(x) \le 1$</td> <td style="text-align: center;">$f(x)$ can be $> 1$</td> </tr>
    <tr> <td style="text-align: center;">Generally $p(x) \neq 0$</td> <td style="text-align: center;">$p(x) = 0$</td> </tr>
    <tr> <td rowspan="3" style="text-align: center;">Generally $\Pr(X \le a) \neq \Pr(X < a)$</td> <td style="text-align: center;">$\Pr(X \le a) = \Pr(X < a) = F(a)$</td> </tr>
    <tr> <td style="text-align: center;">$\Pr(X \ge a) = \Pr(X > a) = 1 -F(a)$</td></tr>
    <tr> <td style="text-align: center;">$\Pr(a \le X \le b) = \Pr(a < X < b) = F(b) - F(a)$</td> </tr>
    </tbody>
  </table>

+ Expectation
  + discrete: $E[X] = \sum x \cdot p(x0)$
  + continuous: $E[X] = \int_{-\infty}^\infty xf(x)\, dx$
  + as discrete: average of many samples
  + properties
    + support set = [a, b]: $a \le E[X] \le b$
    + symmetry: $\exists\, \alpha, f(\alpha + x) = f(\alpha -x) \;\forall\, x \implies E[X] = \alpha$

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yb4obz4o" ismap target="_blank">
      <img src="img/t09-03.png" style="margin: 0.1em;" alt="Example of symmetry property" title="Example of symmetry property" width=250>
    </a>
  </div>

  + examples
    + uniform: $E[X] = \int_{-\infty}^\infty x f(x)\,dx = \int_0^1 x \,1\, dx = \left.\frac{x^2}{2}\right|_0^1 = \frac12$
    + triangle: $E[X] = \int_0^1 x \cdot \,2x \,dx = \left.\frac{2x^3}{3}\right|_0^1 = \frac23$
    + power law: $E[X] = \int_1^\infty x\, \frac{1}{x^2}\,dx = \int_1^\infty \frac 1 x \,dx = \left.\ln x \right|_1^\infty = \infty$

+ Variance
  + definition: $Var(X) \triangleq E[(X - \mu)^2]$
    + discrete: $Var(x) = \sum_x p(x) (x - \mu)^2$
    + continuous: $Var(x) = \int_{-\infty}^\infty f(x)(x - \mu)^2\,dx$
  + as for discrete: $Var(X) = E[X^2] - (E[X])^2$

    \[\begin{align*}
      E[(X - \mu)^2] &= \int (x - \mu)^2 f(x)\,dx = \int (x^2 - 2x\mu + \mu^2)f(x)\,dx \\
      &= \int x^2f(x)\,dx - 2\mu \int xf(x)\,dx + \mu^2 \\
      &= E[X^2] - 2\mu^2 + \mu^2 = E[X^2] - \mu^2
    \end{align*}\]

  + standard deviation: $\sigma = \sqrt{Var(X)}$
  
+ Examples
  + uniform:
    + mean: $[X] = \frac12$
    + $E[X^2] = \int_0^1 x^2 \,1\, dx = \left.\frac{x^3}{3}\right|_0^1 = \frac13$
    + variance: $Var(X) = E[X^2] - (E[X])^2 = \frac13 - \frac14 = \frac{1}{12}$
    + standard deviation: $\sigma = \frac{1}{\sqrt{12}} = \frac{1}{2\sqrt{3}}$
  + triangle
    + mean: $E[X] = \frac23$
    + $E[X^2] = \int_0^1 x^2 \, 2x \, dx = \left.\frac24 x^4 \right|_0^1 = \frac12$
    + variance: $Var(X) = E[X^2] - (E[X])^2 = \frac12 - (\frac23)^2 = \frac{9-8}{18} = \frac{1}{18}$
    + standard deviation: $\sigma = \frac{1}{\sqrt{18}} = \frac{1}{3\sqrt{2}}$

+ Discrete vs. Continuous

  <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;"></th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Discrete</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Continuous</th>
    </tr>
    </thead>
    <tbody>
    <tr> <th>Prob. Fun.</th> <td style="text-align: center;">pmf: p</td> <td style="text-align: center;">pdf: f</td> </tr>
    <tr> <th>$\ge 0$</th> <td style="text-align: center;">$p(x) \ge 0$</td> <td style="text-align: center;">$f(x) \ge 0$</td> </tr>
    <tr> <th>unitary</th> <td style="text-align: center;">$\sum p(x0 = 1$</td> <td style="text-align: center;">$\int f(x)\,dx = 1$</td> </tr>
    <tr> <th>$\Pr(A)$</th> <td style="text-align: center;">$\sum_{x \in A} p(x)$</td> <td style="text-align: center;">$\int_{x \in A} f(x)\,dx$</td> </tr>
    <tr> <th>$F(X)$</th> <td style="text-align: center;">$\sum_{u \le x} p(u)$</td> <td style="text-align: center;">$\int_{-\infty}^x f(u)\, dx$</td> </tr>
    <tr> <th>$\mu = E[X]$</th> <td style="text-align: center;">$\sum x p(x)$</td> <td style="text-align: center;">$\int xf(x)\, dx$</td> </tr>
    <tr> <th>$Var(X)$</th> <td style="text-align: center;">$\sum (x-\mu)^2 p(x)$</td> <td style="text-align: center;">$\int (x-\mu)^2 f(x)\,dx$</td> </tr>
    </tbody>
  </table>



+ [Original Slides](https://tinyurl.com/yb4obz4o)


### Problem Sets






### Video Links

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 9.2 Functions of Random Variables






+ [Original Slides]()


### Problem Sets






### Video Links

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 9.3 Uniform Distribution






+ [Original Slides]()


### Problem Sets






### Video Links

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 9.4 Exponential Distribution






+ [Original Slides]()


### Problem Sets






### Video Links

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 9.5 Gaussian Distribution






+ [Original Slides]()


### Problem Sets






### Video Links

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 9.6 Gaussian Distribution - Probabilities






+ [Original Slides]()


### Problem Sets






### Video Links

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 9







## Programming Assignment 9








