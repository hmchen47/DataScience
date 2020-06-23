# Topic10: Inequalities and Limit Theorems


## 10.1 Markov Inequality

+ Motivation: Probability bounds
  + bound probability of events (often bad)
  + e.g., excessive rain, heavy traffic, large loss, disease outbreak
  
+ Example: Markov's meerkats
  + average meerkat height: $10^{\prime\prime}$
    + can half the meerkats $\ge 40^{\prime\prime}$ tall?
    + No, if half meerkats were $\ge 40$ tall, average would be $\ge \frac12 \times 40^{\prime\prime} = 20^{\prime\prime} > 10^{\prime\prime}$
  + $F_{40}$: fraction of meerkats $\ge 40^{\prime\prime}$ tall
  + $F_{40} \cdot 40 > 10 \implies$ average would be $> 10$
  + $F_{40} \cdot 40 \le 10 \implies F_{40} \le 10/40 = \frac14$
  + general $\mu$: $F_{4 \cdot \mu} \cdot (4 \cdot \mu) \le \mu \to F_{4 \cdot \mu} \le \frac14$

+ Markov's Inequality
  + two forms
    + intuitive, memorable
    + direct, applicable, common
  + $X$: <span style="color: magenta;">nonnegative</span> r.v. (discrete or continuous) w/ finite mean $\mu$
  + intuitive, memorable

    \[ \forall\; \alpha \ge 1 \quad \Pr(X \ge \alpha) \mu \le \frac{1}{\alpha} \]

    + a nonnegative r.v. is at least $\alpha$ times $\le$ its mean w/ probability $\le \frac{1}{\alpha}$
  + direct proof, easier to apply, more common

    \[ a = \alpha \mu \quad \forall\; \alpha \ge \mu \quad \Pr(X \ge a) \le \frac{\mu}{a} \]

  + proof
    + proof for discrete r.v.'s, same proof works for continuous, just $\sum \to \int$
    + $\Pr(X \ge \alpha) \le \frac{\mu}{a}$
    + $\mu = \int_x x \cdot p(x) \,dx \ge \int_{x \ge a} x \cdot p(x) \, dx$

        \[ \mu = \sum_x x \cdot p(x) \ge \sum_{x \ge a} x \cdot p(x) \ge \sum_{x \ge a} a \cdot p(x) = a \cdot \Pr(X \ge a) \]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yac6dd37" ismap target="_blank">
        <img src="img/t10-01.png" style="margin: 0.1em;" alt="Illustrated proof of Markov inequality" title="Illustrated proof of Markov inequality" width=450>
      </a>
    </div>

+ Example: Citation counts
  + a journal paper cited 8 times on average
  + Y. Benjamin and Y. Hochberg, [Controlling the False Discovery Rate: a Practical and Powerful Approach to Multiple Testing](https://tinyurl.com/y8a4o7bw), J. R., Stat. Soc. B, 1995
    + popular (mutiple) hypothesis-testing paper
    + cited $\ge 40,000$ times
  + bound probability that a paper get cited $\ge 40,000$ times
    + $X$: \# paper citations
      + $X \ge 0 \quad \mu=8$
    + Markov:
      + $\Pr(X \ge a) \le \frac{\mu}{a}$
      + $\Pr(X \ge 40,000) \le \mu / 40K = 8 / 40K = 0.02\%$

+ Generalization?
  + can the Markov $\le$ be
    + generalized (conditions relaxed)?
    + strengthened?
  + generalization attempt: removing non-negative?
    + $X < 0 \implies \Pr(x \ge a)$ be closed to 1 for any $a$
    + $p(x) = \begin{cases} 1 - \epsilon & x = a \\ \epsilon & x = \frac{\mu - (1 - \epsilon)a}{\epsilon} \end{cases}$
    + $E[X] = \mu \implies p(X \ge a) = p(a) \approx 1 \to$ unable to remove
  + strengthening $\Pr(X \ge a) \le \frac{\mu}{a}$?
    + viz. the probability at most $\frac{\mu}{a}$
    + can the $\le$ hold with equality?

      \[ \mu = \sum_x x \cdot p(x) \ge \sum_{x \ge a} x \cdot p(x) \ge \sum_{x \ge a} a \cdot p(x) = a \cdot \Pr(X \ge a) \]

      + equality in 1st $\ge$: $\forall\, x \in (0, a), \quad p(x) = 0$
      + equality in 2nd $\ge$: $\forall \,x > a, \quad p(x) = 0$
    + only hold w/ $X \in \{0, a\}$

+ Properties of Markov's Inequality
  + applied to all non-negative random variables
  + can always be used
  + used to derive other inequalities: Chebyshev, Chernoff
  + limited to inequalities that hold for all distributions

+ Different views
  + from outside: $[0, a)$
    + $\Pr(X \ge a)$
    + upper bound $\Pr \le \frac{\mu}{a}$
  + from inside: $[0, a)$
    + $\Pr(X \le a)$
    + lower bound: $\Pr > 1 - \frac{\mu}{a}$


+ [Original Slides](https://tinyurl.com/yac6dd37)


### Problem Sets

0. A mob of 30 meerkats has an average height of 10”, and 10 of them are 30” tall. According to Markov's Inequality this is:<br/>
  a. Possible<br/>
  b. Impossible<br/>

  Ans: <span style="color: magenta;">b</span>
  Explanation: Impossible. For the average to be 10, the remaining 20 meerkats would need to have height zero.


1. Which of the following are correct versions of Markov’s Inequality for a nonnegative random variable  X :<br/>
  a. $\Pr(X \ge \alpha \mu) \le \frac{1}{\alpha}$<br/>
  b. $\Pr(X \ge \alpha \mu ) \le \mu \alpha$<br/>
  c. $\Pr(X \ge \mu) \le frac{1}{\alpha}$<br/>
  d. $\Pr(X \ge \alpha) \le \frac{\mu}{\alpha}$<br/>

  Ans: ad


2. Markov variations

  Upper bound $P(X \ge 3)$ when $X \ge 2$ and $E[X]=2.5$.

  Ans: <span style="color: magenta;">0.5</span><br/>
  Explanation: Let $Y=X−2$. Then $Y \ge 0$ and $E[Y]=E[X]−2=0.5$. By Markov's inequality, $P(X \ge 3) = P(Y \ge 1) \le \frac{E[Y]}{1}=0.5$. [StackExchange](https://tinyurl.com/yc6qj6vs)


3. a. In a town of 30 families, the average annual family income is \$80,000. What is the largest number of families that can have income at least \$100,000 according to Markov’s Inequality? (Note: The annual family income can be any non-negative number.)

  b. In the same town of 30 families, the average household size is 2.5. What is the largest number of families that can have at least 4 members according to Markov’s Inequality? (Note the household size can be any postive integer.)

  Ans: a. (24); b. (<span style="color: cyan;">15</span>)>br/>
    + This question can be answered using the Meerkat paradigm, or we can convert it to a probability question and use Markov's Inequality. Imagine that you pick one of the 30 families uniformly at random. The expected income is the average over all families, $80,000. The probability that the random family has income at least $100,000 is the number of families with such income, normalized by 30. By Markov's Inequality, this probability is at most  80000/100000=0.8 . Hence the number of families with such income is at most  30⋅0.8=24 .
    + Let $X$ be the size of a family picked uniformly at random. Then $X \ge 1$  and $E[X]=2.5$. Define $Y=X−1$. Then $Y \ge 0$ and $E[Y]=E[X]−1=1.5$. By Markov's Inequality $P(X \ge 4)=P(Y \ge 3) \le \frac{1.5}{3}=\frac12$. Hence the fraction of families with at least 4 members is at most $\frac12⋅30=15$.



### Lecture Video

<a href="https://tinyurl.com/y8gyrt4b" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 10.2 Chebyshev Inequalities

+ Motivation
  + Pafunty Chebyshev, 1821~1894
    + &gt; 12K "descendents"
    + "Father" of modern Russian mathematics
    + most famous for spelling: Chebychev, Chebysheff, Chebychov, Chebyshow, Tchebychev, Tchebycheff, Tschebyschev, Tschebychef, Tschebyscheff, ...
  + many contributions
    + $\forall\; n < \;\;\; \exists\; \text{ prime } < 2n$
    + probability theory
  + legendary teacher
    + Punctual
    + advocated applied math
    + flowery language
  + to isolate mathematics from practical sciences is to shut the cow away from the bulls
  + famous students
    + Markov
    + Lyapunov

+ Markov Inequality to Chebychev Inequality
  + Markov inequality<br/>
    Probability that non-negative $X$ is $\alpha$ times larger than its mean is $\le \frac{1}{\alpha}$

    \[ \Pr(\text{non-negative } X \ge \alpha \mu) \le \frac{1}{\alpha} \]

  + Chebyshev inequality<br/>
    Probability that any $X$ is $\alpha$ times further from $\mu$ than $\sigma$ is $\le \frac{1}{\alpha^2}$

    \[ \Pr(\text{any } X \ge \alpha \sigma \text{ away from } \mu) \le \frac{1}{\alpha^2} \]

+ Chebyshev's inequality
  + two forms
    + easier to illustrate, understand, remember
    + easier to prove, use
  + $X$: any r.v. (discrete or continuous) w/ finite <span style="color: magenta;"> mean $\mu$</span> and <span style="color: magenta;"> std $\sigma$</span>
  + 1st formulation

    \[ \forall\; \alpha \ge 1 \quad \Pr(|X - \mu| \ge \alpha \sigma) \le \frac{1}{\alpha^2} \]

  + 2nd formulation: $a = \alpha \sigma$, $a$ a value of interest

    \[ \forall\; a \ge \sigma \quad \Pr(|X - \mu| \ge a) \le \frac{\sigma^2}{a^2} \]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/y9yygxbq" ismap target="_blank">
        <img src="img/t10-02.png" style="margin: 0.1em;" alt="Example distribution for Chebyshev's inequality" title="Example distribution for Chebyshev's inequality" width=450>
      </a>
    </div>

  + toward a proof
    + Markov inequality

      \[ \forall\; a \ge \mu \quad \Pr(X \ge a) \le \frac{\mu}{a} \]

    + Chebyshev inequality

      \[\begin{align*}
        \forall\; a \ge \sigma \quad &\Pr(|X - \mu| \ge a) \le \frac{\sigma^2}{a^2} \qquad\quad\qquad\; |X - \mu| \ge a\\
        & \Pr \left((X - \mu)^2 \ge a^2\right) \le \frac{\sigma^2}{a^2} \qquad\qquad (X - \mu)^2 \ge  a^2
      \end{align*}\]

    + applying Markov to $(X - \mu)^2$
  + proof
    + $X$: any random variable
    + $\mu_X = E[X] \quad \sigma_X^2 = Var(X) = E[(X - \mu_X)^2]$

      \[ \Pr(|X - \mu_X| \ge a) \le \frac{\sigma_X^2}{a^2} \]

    + $Y = (X - \mu_X)^2 \to Y \ge 0 \to \mu_Y = E[(X - \mu_X)^2] = \sigma^2_X$

      \[\begin{align*}
        \Pr(|X - \mu_X| \ge a) &= \Pr((X - \mu_X)^2 \ge a^2) = \Pr(Y \ge a^2) \\
        &\le \frac{\mu_Y}{a^2} = \frac{\sigma_X^2}{a^2} \quad (\text{Markov})
      \end{align*}\]

+ Example: citations
  + $X$: \# paper citations
  + $\mu = 8, \text{suppose } \sigma = 5$
  + $\Pr(X \ge 28)?$
  + Markov
    + $\Pr(X \ge a) \le \frac{\mu}{a}$
    + $\Pr(X \ge 28) \le 8/28 \approx 29\%$
  + Chebyshev
    + $\Pr(|X - \mu| \ge a) \le \frac{\sigma^2}{a^2}$
    + $\Pr(X \ge 28) = \Pr(X - \mu \ge 20) \le \Pr(|X - \mu| \ge 20) \le (\frac{\sigma}{20})^2 = (\frac{5}{20})^2 = \frac{1}{16} \approx 6.3\%$
    + $\Pr(X \ge 40,000) = \Pr(X -\mu \ge 39,992)$ $\le \Pr(|X - \mu| \ge 39,992)$ $\le \left( \frac{\sigma}{39,992} \right)^2$ $= \left( \frac{5}{39,992} \right)^2$ $= 1.6 \times 10^{-6} \%$

+ Example: survey responses
  + survey expected to result in $\mu =$ 1M responses w/ $\sigma = 50K$
  + bound $\Pr(0.8M < \# \text{ responses } < 1.2M)$
    + 0.8M = $\mu - 4\sigma$
    + 1.2M = $\mu + 4\sigma$
  + $\Pr( \mu - 4\sigma < X < \mu + 4\sigma) = \Pr(|X - \mu| < 4\sigma) = 1 - \Pr(|X - \mu| \ge 4\sigma) \ge 1 - \frac{1}{16} = \frac{15}{16}$

+ Markov vs. Chebyshev inequalities

  <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;"></th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Formula</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Applies</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Input</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Range</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Deceases</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <th style="text-align:center;">Markov</th>
      <td style="text-align:center;">$\Pr(X \ge a) \le \frac{\mu}{a}$</td>
      <td style="text-align:center;">$X \ge 0$</td>
      <td style="text-align:center;">$\mu$</td>
      <td style="text-align:center;">$a \ge \mu$</td>
      <td style="text-align:center;">Linearity</td>
    </tr>
    <tr>
      <th style="text-align:center;">Chebyshev</th>
      <td style="text-align:center;">$\Pr(|X - \mu| \ge a) \le \frac{\sigma^2}{a^2}$</td>
      <td style="text-align:center;">Any $X$</td>
      <td style="text-align:center;">$\mu$ &amp; $\sigma$</td>
      <td style="text-align:center;">$a \ge \sigma$</td>
      <td style="text-align:center;">Quadratically</td>
    </tr>
    </tbody>
  </table>

+ One-sided Chebyshev inequality
  + [Henry Bottomley](http://www.se16.info/hgb/cheb.htm)
  + Theorem: (Chebyshev inequality - one-sided version) for $t > 0$

    \[ \Pr(X - \mu \ge t) \le \frac{1}{1 + t^2/Var(X)} = \frac{Var{X}}{Var(X) + t^2} \]

  + Proof: [Ref1](http://www.se16.info/hgb/cheb.htm#OTProof) and [Ref2](http://www.se16.info/hgb/cheb2.htm)
    + one of these, loosely based on _Probability and Random Process_ by Grimmett and Stirzaker
    + w/ $a > 0, \forall\, b \ge 0$ 

      \[ \Pr(X \ge a) = \Pr(X+b \ge a + b) \le E\left[ \frac{(X+b)^2}{(a+b)^2} \right] = \frac{\alpha^2+b^2}{(a+b)^2} \]

    + treating $\frac{\sigma^2+b^2}{(a+b)^2}$ as a function of $b$, the minimum occurs at $b = \sigma^2/a$, so 

      \[ \Pr(X \ge a) \le \frac{\sigma^2 + (\sigma^2/a)^2}{(a+\sigma^2/a)^2} = \frac{\sigma^2(a^2+\sigma^2)}{(a^2 + \sigma^2)^2} = \frac{\sigma^2}{\sigma^2 + a^2} \]


+ [Original Slides](https://tinyurl.com/y9yygxbq)


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 10.3 Law of Large Numbers






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 10.4 Moment Generating Functions






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 10.5 Chernoff Bound






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 10.6 Central Limit Theorem






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 10.7 Central Limit Theorem Proof






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 10








## Programming Assignment 10









