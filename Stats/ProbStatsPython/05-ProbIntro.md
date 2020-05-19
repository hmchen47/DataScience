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
  + give up or reasonintelligently

  <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Properties</th>
      <th colspan="3" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Interests</th>
    </tr>
    </thead>
    <tbody>
    <tr> <th>Learn</th> <td>range</td> <td>average</td> <td>variability</td> </tr>
    <tr> <th>Infer</th> <td>relations</td> <td>structure</td> <td>change</td> </tr>
    <tr> <th>Predict</th> <td>future</td> <td>likelihood</td> <td>guarantees</td> </tr>
    <tr> <th>Benefit</th> <td>understand</td> <td>plan</td> <td>build</td> </tr>
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
  + precisely
    + intuitive approach
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
  + generic
    + biology
    + engineering
    + business
    + sociology
  + understand
    + simple examples
    + progress

+ Outcomes and sample space
  + <span style="color: magenta;">outcomes</span>: possible experiment results
  + <span style="color: magenta;">sample space</span>: set of possible outcomes, denoted <span style="color: magenta;">$\Omega, S \text{ or } U$</span>

    <table style="font-family: arial,helvetica,sans-serif; width: 20vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
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
    + e.g., $\{h, t\}, \{1, 2, \dots, 6\}, \Bbb{N}, \Bbb{Z}, \{words\}, \{cities\}, \{people\}$
  + <span style="color: magenta;">continuous</span>: uncountably infinite sample space
    + e.g., $\Bbb{R}, [0, 1], \{\text{temperatures}\}, \underbrace{\{\text{salaries}\}, \{\text{prices}\}}_{\text{upgraded}}$
  + discrete space: easier to understand, visualize, analyze; important; first
  + continuous: important; conceptually harder; later

+ Random outcomes
  + algebra
    + unknown value, denote <span style="color: cyan;">x</span> $\gets$ lower case
    + e.g., $2x - 4=0$, solution
      + before: $x \in \Bbb{R}$
      + after: $x = 2$
  + probability
    + <span style="color: cyan;">random</span> value of outcome, denoted by <span style="color: cyan;">$X$</span> $\gets$ upper case
    + experiment
      + $X$: coin flip outcome
      + before: $X \in \Omega$
      + after: $X = \begin{cases} \text{h} & \text{get h} \\ \text{t} & \text{get t} \end{cases}$

+ Probability of an outcome
  + the <span style="color: magenta;">probability</span>, or <span style="color: magenta;">likelihood</span>, of an outcome $x \in \Omega$, denoted <span style="color: magenta;">$P(x), \Pr(x)$</span>, or <span style="color: magenta;">$P(X = x), \Pr(X = x)$</span>, is the fraction of times $x$ will occur when experiment is repeated many times
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
      0 \leq n_x \leq n &\to& 0 \leq \frac{n_x}{n} \leq 1 &\to& 0 \leq p(x) \leq 1 \\\\
      \displaystyle \sum_{x \in \Omega} n_x = n &\to& \displaystyle\sum_{x \in \Omega} \dfrac{n_x}{n} = 1 &\to& \displaystyle\sum_{x\in \Omega} p(x) = 1
    \end{array}\]

+ Probability distribution function
  + $P(x)$: the fraction of times outcome $x$ occurs, e.g., $\Pr(h) = 1/2, \Pr(1) = 1/6$
  + viewed over the whole sample space $\to$ a pattern merges
    + coin: $\Pr(h) = 1/2, \Pr(t) = 1/2$
    + die: $\Pr(1) = 1/6, \dots, \Pr(6) = 1/6$
    + rain: $\Pr(\text{rain}) = 10\%, \Pr(\text{no rain}) = 90\%$
  + <span style="color: magenta;">Probability distribution function (PDF)</span>: $\Pr$ mapping outcome in $\Omega$ to nonnegative values that sum to 1
  
    \[ \Pr: \Omega \to \Bbb{R} , \; \Pr(x) \geq 0 \text{ s.t. } \sum_{x\in \Omega} \Pr(x) = 1 \]

  + sample space $\Omega$ + distribution $P$ = probability space


+ [Original Slides](https://tinyurl.com/y7p67f6u)


### Problem Sets

0. Which of the following outcomes are random (not certain) when rolling a six-sided dice?<br/>
  a. A real number.<br/>
  b. An even number.<br/>
  c. A positive number.<br/>

  Ans: <span style="color: magenta;">b</span>
  Explanation: The outcome of dice is certainly real and positive, but it may or may not be even, so it is random.

1. Which of the following outcomes are random (not certain) after throwing a six-sided dice?<br/>
  a. Get number  3 <br/>
  b. Get an even number<br/>
  c. Get a positive number<br/>

  Ans: ab<br/>
  Explanation
    + True. We may get e.g. 4 as an outcome, which is not 3.
    + True. We may get e.g. 3 as an outcome, which is not even.
    + False. All outcomes of a six-sided dice are positive.


2. Imagine a single experiment where we flip a coin 6 times, and get “head, tail, head, head, head, head”.

  Which of the following statements hold?

  a. The coin is not fair.<br/>
  b. The coin's "tail" probability is 1/6.<br/>
  c. The sequence "head, tail, head, head, head, head" is an outcome in the sample space.<br/>
  d. The sample space of the experiment is {head, tail}.<br/>

  Ans: <span style="color: magenta;">c</span>
  Explanation
    + False. The outcome is random and the coin may be fair.
    + False. In this experiment 1 out of 6 outcomes was "tail", but the coin's "tail" probability may differ.
    + True. The sample space consists of all sequences of six "head" and "tail", and this is one of them.
    + False. The sample space is a set of tuples  {(head, head, head, head, head, head),(head, head, head, head, head, tail),⋯,(tail, tail, tail, tail, tail, tail)} .


### Lecture Video

<a href="https://tinyurl.com/yctp6qka" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.2 Distribution Types

+ Uniform probability spaces
  + generally, outcomes may have different probability
    + e.g., Rain: P(rain) = 10%, P(no rain) = 190%
  + uniform (equiprobable) spaces: uniform distribution
    + all outcomes are equally likely
    + $\forall\, x \in \Omega \quad \Pr(x) = p$
    + $1 = \sum_{x \in \Omega} \Pr(x) = \sum_{x \in \Omega} p = |\Omega| \cdot p$
    + $p = 1 / |\Omega|$
  + example
    + $\sum_{x \in \{3,5\}} x^2 = 3^2 + 5^2 = 34$
    + $\sum_{x \in \{3,5\}} x = 3 + 5 = 8$
    + $\sum_{x \in \{3,5\}} p = p + p = 2p$
    + $\sum_{x \in \Omega} p = p + p + \cdots + p = |\Omega|$
    + fair coin: $\Pr(h) = P(t) = p \quad 1 = \Pr(h) + \Pr(t) = 2 \cdot p \quad p = 1/2$
  + drastically simplified probability specification
  + uniform spaces: every outcome w/ probability $1/|\Omega|$
  + all you need to know is $|\Omega|$
  + notation: denoted $U$, drawing uniformly, randomly

+ Example: Fair coin
  + $\Omega = \{ \text{heads}, \text{tails}\} = \{h, t\}$
  + $|\Omega| = 2$
  + flip, or toss
  + equally likely: $U \to \Pr(h) = \Pr(t)$
  + $\Pr(h) = \Pr(t) = \frac{1}{|\Omega|} = \frac{1}{2}$

+ Example: Fair die
  + sample space: $\Omega = \{1, 2, 3, 4, 5, 6\}$
  + $|\Omega| = 6$
  + roll
  + equally likely: $U \to \Pr(1) = \cdots = \Pr(6)$
  + $\Pr(1) = \cdots = \Pr(6) = \frac{1}{\Omega} = \frac{1}{6}$

+ Example: deck of cards
  + sample space: $\Omega = \{ \text{cards }\}$
  + $|\Omega| = 52$
  + draw a card
  + equally likely: $U \to \Pr(3C) = \cdots = \Pr(QH) = \frac{1}{\Omega} = \frac{1}{52}$

+ Non-uniform
  + uniform, equiprobable, spaces, e.g., coin, die, cards
  + in nature, nonuniform spaces around, e.g., rain, grades, words, illnesses, web pages, people, ...
  + example: Pie chart
    + usually non-uniform
    + challenge: non-uniform distribution we can remember
  + example: Tetrahedra die
    + 4-sided, pyramid die
    + used in games, D&D
    + in games die equiprobable
    + assumption: different probabilities
    + easy to remember
      + 4 faces
      + $\Pr(1) = .1, \;\Pr(2) = .2 \;\Pr(3) = .3 \;\Pr(4) = .4$
    + conveniently, add to 1
    + probability distribution

+ Do's and Don'ts
  + random notation may be confusing at first
  + which expressions are valid?
    + valid expression
      + $\Pr(x)$ specify $x$, e.g., for $\forall\, x, \Pr(x) = 1/4$
      + $\Pr(X=3)$ w/ fair die: 1/6
      + $\Pr(3) \stackrel{def}{=} P(X=3)$
    + possible, but less common, make sure it's what you mean
      + $\Pr(1 = 3): 0$
      + $\Pr(X)$: random value
    + even less likely, probably wrong
      + $\Pr(x=3)$


+ [original Slides](https://tinyurl.com/y84vwcva)


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







