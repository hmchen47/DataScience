# Topic 5: Probability Introduction


## 5.1 Probability

+ Why probability?
  + something in life are certain
  + most are a less predictable
    + physicians: illness, medication
    + farmers: rain, diet trends
    + investors: stock price, economy
    + advertisers: views, competition
    + consumers: availability, sale
    + students: food line, grade, parents job, date, game

+ Random phenomena

  <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Give Up?</th>
      <th colspan="3" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Reason intelligently?</th>
    </tr>
    </thead>
    <tbody>
    <tr> <th>Learn</th> <td>range</td> <td>average</td> <td>variability</td> </tr>
    <tr> <th>Infer</th> <td>relations</td> <td>structure</td> <td>change</td> </tr>
    <tr> <td>Predict</td> <td>future</td> <td>likelihood</td> <td>guarantees</td> </tr>
    <tr> <td>Benefit</td> <td>plan</td> <td>create</td> <td>compete</td> </tr>
    </tbody>
  </table>

+ Coming to terms
  + as with sets: need terminology
  + discuss
    + concisely
    + precisely
    + effectively
    + process of generating and observing data
    + individual and collection of observations
    + meaning of probability
  + several approaches
    + intuitive
    + axioms $\gets$ data

+ Experiments
  + probability developed in part to aid science
  + process
    + generate random data
    + observe outcome
  + unified approach
    + applies generally
    + understand
    + analyze
    + generalize
  + experiment
    + biology
    + engineering
    + business
    + sociology
  + out experiments
    + coin - starting simple
    + die - get very complex

+ Outcomes and sample space
  + <span style="color: magenta;">(possible) outcomes</span>: potential experiment results
  + <span style="color: magenta;">sample space</span>: set of possible outcomes, denoted <span style="color: magenta;">$\Omega$</span>

    <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Experiment</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">$\Omega$</th>
      </tr>
      </thead>
      <tbody>
      <tr> <td style="text-align: center;">Coin</td> <td style="text-align: center;">$\{h, t\}$</td> </tr>
      <tr> <td style="text-align: center;">Die</td> <td style="text-align: center;">$\{1, 2, \dots, 6\}$</td> </tr>
      <tr> <td style="text-align: center;">Gender</td> <td style="text-align: center;">$\{m, f\}$</td> </tr>
      <tr> <td style="text-align: center;">Age</td> <td style="text-align: center;">$\Bbb{N}$</td> </tr>
      <tr> <td style="text-align: center;">Temperature</td> <td style="text-align: center;">$\Bbb{R}$</td> </tr>
      </tbody>
    </table>

+ Two sample-space types
  + <span style="color: magenta;">discrete</span>: finite or countable infinite sample space
    + e.g., $\{h, t\}, \{1, 2, \dots, 6\}, \Bbb{N}, \Bbb{R}, \{words\}, \{cities\}, \{people\}$
  + <span style="color: magenta;">continuous</span>: uncountably infinite sample space
    + e.g., $\Bbb{R}, [0, 1], \{\text{temperatures}\}, \underbrace{\{\text{salaries}\}, \{\text{prices}\}}_{\text{upgraded}}$
  + discrete space: easier to understand, visualize, analyze; important; first
  + continuous: important; conceptually harder; later

+ Random outcomes
  + algebra
    + unknown value, denote <span style="color: magenta;">x</span> $\gets$ lower case
    + e.g., $2x - 4=0$, solution
      + before: $x \in \Bbb{R}$
      + after: $x = 2$
  + probability
    + random outcome, denoted by <span style="color: magenta;">$X$</span> $\gets$ upper case
    + experiment
      + $X$: coin flip outcome
      + before: $X \in \Omega$
      + after: $X = \begin{cases} \text{h} & \text{get h} \\ \text{t} & \text{get t} \end{cases}$

+ Probability of an outcome
  + the <span style="color: magenta;">probability</span>, or <span style="color: magenta;">likelihood</span>, of an outcome $x \in \Omega$, denoted <span style="color: magenta;">$P(x)$</span>, or <span style="color: magenta;">$P(X = x)$</span>, is the fraction of time $x$ will occur when experiment is repeated many times
  + fair coin
    + as \# experiment $\to \infty$, fraction of heads (or tails) $\to 1/2$
    + heads w/ probability $1/2$: $P(h) = 1/2 \quad P(X=h) = 1/2$
    + tails w/ probability $1/2$: $P(t) = 1/2 \quad P(X=t) = 1/2$
  + fair die
    + as \# experiments $\to \infty$, fraction of 1's (or 2,...,6) $\to 1/6$
    + 1 w/ probability $1/6$: $P(1) = 1/6 \quad P(X=1) = 1/6$
  + $P(X)$ or $P(X=x)$: probability of x; fraction of times will occur

+ Probability portrait
  + $n$ experiments
  + $x \in \Omega \quad n_x =$ \# times x appeared

    \[ \Pr(x) = \lim_{n \to \infty} \frac{n_x}{n} \]

    \[\begin{array}{ccccc}
      0 \leq n_x \leq &\to& 0 \leq \frac{n_x}{n} \leq 1 &\to& 0 \leq p(x) \leq 1 \\\\
      \displaystyle \sum_{x \in \Omega} n_x = n &\to& \displaystyle\sum_{x \in \Omega} \dfrac{n_x}{n} = 1 &\to& \displaystyle\sum_{x\in \Omega} p(x) = 1
    \end{array}\]

+ Sum of probabilities
  + $\Pr(x)$: the fraction of times outcome $x$ occurs, e.g., $\Pr(h) = 1/2, \Pr(1) = 1/6$
  + viewed over the whole sample space $\to$ a pattern merges
    + coin: $\Pr(h) = 1/2, \Pr(r) = 1/2$
    + die: $\Pr(1) = 1/6, \dots, \Pr(6) = 1/6$
    + rain: $\Pr(\text{rain}) = 10\%, \Pr(\text{no rain}) = 90\%$

+ Probability distribution
  + $\Pr$ mapping outcome in $\Omega$ to nonnegative values that sum to 1
  + Probability distribution function: a function $P: \Omega \to \Bbb{R} \text{ s.t. } P(x) \geq 0$
  + sample space $\Omega$ + distribution $\Pr$ = probability space

    \[ \Omega + \Pr = \text{probability space} \]



### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.2 Distribution Types






### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.3 Events






### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.4 Repeated Experiments






### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.5 Games of Chance






### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.6 Axiomatic Formulation






### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.7 Inequalities






### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 5





## Programming Assignment 5







