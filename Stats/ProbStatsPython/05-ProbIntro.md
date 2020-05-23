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

0. An outcome in a uniform probability space has probability 1/10, what is the size of the sample space?<br/>
  a. 5<br/>
  b. 10<br/>
  c. 20<br/>
  d. None of the above<br/>

  Ans: b


1. Which of the following sample spaces are uniform?<br/>
  a. {land,sea} for a randomly point on a globe<br/>
  b. {odd, even} for a random integer from {1,2,. . . ,100}<br/>
  c. {leap year, non-leap year} for a random year before 2019<br/>
  d. {two heads, two tails, one head and one tail} when flipping two fair coins<br/>
  e. {distance to origin} for a random point in {−3, −1, 1, 3} × {−4, −2, 2, 4}<br/>

  Ans: be<br/>
  Explanation
    + False. About 70% of the times will be "sea".
    + True. Half the time we'll observe even, and half odd.
    + False. "non-leap year" is roughly three times more likely.
    + False. "one head and one tail" is more likely.
    + True. There are four distances, and each shared by 4 of the 16 points.


2. Given an uniform probability space $\Omega = \{1,2,3, \dots ,100\}$, what is the probability that the outcome contains the digit 1 (e.g. 19,91)?

  Ans: 19/100<br/>
  Explanation: There are 20 of them. Note that 11 should not be counted twice.



### Lecture Video

<a href="https://tinyurl.com/yctp6qka" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.3 Events

+ Events
  + sometimes: care about one particular outcome; examples,
    + our hourse will win race
    + get ready B+ in class
  + usually: interested in a set of possible outcomes; examples,
    + temperature > 98.6 ℉
    + stock close higher
    + pass the course
  + outcome: single element of $\Omega$
  + event: subset of a space $\Omega$
  + analogy
    + social event: set of people
    + probability event: set of outcomes

+ Die events
  + event: subset of sample space $\Omega$
  + Die: $\Omega = \{1, 2, \dots, 6 \} \supseteq$ Events
  + complement

    <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Name</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$\hspace{3em}$</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Name</th>
      </tr>
      </thead>
      <tbody>
      <tr> <td style="text-align: center;">$\{1, 2, \dots, 6\}$</td> <td style="text-align: center;">$\Omega$ (certain)</td> <td rowspan="5" style="text-align: center;">$\xrightarrow{\text{Complement}}$</td> <td style="text-align: center;">$\{ \}$</td> <td style="text-align: center;">$\varnothing$</td> </tr>
      <tr> <td style="text-align: center;">$\{2, 4, 6\}$</td> <td style="text-align: center;">Even</td> <td style="text-align: center;">$\{1, 3, 5\}$</td> <td style="text-align: center;">Odd</td> </tr>
      <tr> <td style="text-align: center;">$\{1, 4\}$</td> <td style="text-align: center;">Square</td> <td style="text-align: center;">$\{2, 3, 5, 6\}$</td> <td style="text-align: center;">Non square</td> </tr>
      <tr> <td style="text-align: center;">$\{5, 6\}$</td> <td style="text-align: center;">$> 4, \geq 5$</td> <td style="text-align: center;">$\{1, 2, 3, 4\}$</td> <td style="text-align: center;">$\leq 4, < 5$</td> </tr>
      <tr> <td style="text-align: center;">$\{1, 2, 5\}$</td> <td style="text-align: center;">$\{1, 2, 5\}$</td> <td style="text-align: center;">$\{3, 4, 6\}$</td> <td style="text-align: center;">$\{3, 4, 6\}$</td> </tr>
      </tbody>
    </table>

+ Tetrahedral die

  <table style="font-family: arial,helvetica,sans-serif; width: 18.8vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <tbody>
    <tr> <th style="text-align: center; background: seagreen">Face</th> <td style="text-align: center;">1</td> <td style="text-align: center;">2</td> <td style="text-align: center;">3</td> <td style="text-align: center;">4</td> </tr>
    <tr> <th style="text-align: center; background: seagreen">Probability</th> <td style="text-align: center;">.1</td> <td style="text-align: center;">.2</td> <td style="text-align: center;">.3</td> <td style="text-align: center;">.4</td> </tr>
    </tbody>
  </table>

  <table style="font-family: arial,helvetica,sans-serif; width: 30vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr style="font-size: 1.2em;">
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Name</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Set</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">$\Omega$ (certain)</td> <td style="text-align: center;">$\{1, 2, 3, 4\}$</td> </tr>
    <tr> <td style="text-align: center;">Even</td> <td style="text-align: center;">$\{2, 4\}$</td> </tr>
    <tr> <td style="text-align: center;">Prime</td> <td style="text-align: center;">$\{2, 3\}$</td> </tr>
    <tr> <td style="text-align: center;">$\varnothing$ (null)</td> <td style="text-align: center;">$\{\hspace{0.5em} \}$</td> </tr>
    </tbody>
  </table>

+ Event occurrence <br/>
  an event <span style="color: magenta;">occurs</span>, or <span style="color: magenta;">happens</span>, if it contains the observed outcome

    \[ E \;{\color{Magenta}{\text{ occurs }}} \text{ if } X \,{\color{Magenta}{\in}}\, E \]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/y8rpqaq6" ismap target="_blank">
        <img src="img/t05-01.png" style="margin: 0.1em;" alt="Illustration of event occurrence" title="Illustration of event occurrence" width=250>
      </a>
    </div>

    <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Event</th>
        <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Subset</th>
        <th colspan="4" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Outcome</th>
      </tr>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">1</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">2</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">3</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">4</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td style="text-align: center;">$\Omega$ (certain)</td> <td style="text-align: center;">$\{1, 2, 3, 4\}$</td>
        <td style="text-align: center; background-color: darkgreen;">&#9745;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center; background-color: darkgreen;">&#9745;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
      </tr>
      <tr>
        <td style="text-align: center;">Even</td> <td style="text-align: center;">$\{2, 4\}$</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
      </tr>
      <tr>
        <td style="text-align: center;">Prime</td> <td style="text-align: center;">$\{2, 3\}$</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center; background-color: darkgreen;">&#9745;</td> <td style="text-align: center; background-color: darkred;">&#9746;</td>
      </tr>
      <tr>
        <td style="text-align: center;">$\varnothing$ (null)</td> <td style="text-align: center;">$\{\hspace{0.5em}\}$</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkred;">&#9746;</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkred;">&#9746;</td>
      </tr>
      </tbody>
    </table>

+ Event probability
  + $\Pr(E)$: probability of event $E$
  + $\Pr(X \in E)$: probability that $E$ occurs
  + fraction of experiments where $E$ occurs, as \# experiments grows

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/y8rpqaq6" ismap target="_blank">
        <img src="img/t05-02.png" style="margin: 0.1em;" alt="Example of event occurrence w/ Tetrahedral die" title="Example of event occurrence w/ Tetrahedral die" width=450>
      </a>
    </div>

  + task: $\Pr(E)$ = fraction
    + \# experiments 12 $\to \infty$
    + general distributions and events

+ $\Pr(X) \to \Pr(E)$
  + related probability of event to probability of its elements (outcomes)
  + general event $E$
    + \# times E occurs = sum of \# times its elements occur

    \[\begin{align*}
      \Pr(E) &= \text{ fraction of times $E$ occurs} \\
      &= \text{ sum of fraction of times its elements occur} \\
      &= \text{ sum of its element probabilities} \\\\
      \Pr(E) &= \Pr(X \in E) = \sum_{x \in E} \Pr(x)
    \end{align*}\]

  + example: Tetrahedral die
    + \# times Event occurs = sum of \# times 2 and 4 occur
    + Pr(Even) = fraction of times Even occurs = sum of fraction of times 2 and 4 = Pr(2) + Pr(4)

    <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Event</th>
        <th rowspan="2" style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Subset</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">1</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">2</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">3</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">4</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">$\Pr(E)$</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td style="text-align: center;">$\Omega$ (certain)</td> <td style="text-align: center;">$\{1, 2, 3, 4\}$</td>
        <td style="text-align: center; background-color: darkgreen;">&#9745;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center; background-color: darkgreen;">&#9745;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center;">$.1+.2+.3+.4 =1$</td>
      </tr>
      <tr>
        <td style="text-align: center;">Even</td> <td style="text-align: center;">$\{2, 4\}$</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center;">$.2+.4 =.61$</td>
      </tr>
      <tr>
        <td style="text-align: center;">Prime</td> <td style="text-align: center;">$\{2, 3\}$</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkgreen;">&#9745;</td>
        <td style="text-align: center; background-color: darkgreen;">&#9745;</td> <td style="text-align: center; background-color: darkred;">&#9746;</td>
        <td style="text-align: center;">$.2+.3 =.5$</td>
      </tr>
      <tr>
        <td style="text-align: center;">$\varnothing$ (null)</td> <td style="text-align: center;">$\{\hspace{0.5em}\}$</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkred;">&#9746;</td>
        <td style="text-align: center; background-color: darkred;">&#9746;</td> <td style="text-align: center; background-color: darkred;">&#9746;</td>
        <td style="text-align: center;">$0$</td>
      </tr>
      </tbody>
    </table>

+ Uniform spaces
  + equiprobable spaces
  + simple formula for probability of
    + outcome: $\Pr(x) = 1/|\Omega|$
    + event

      \[ \Pr(E) = \sum_{x \in E} \Pr(x) = \sum_{x \in E} \dfrac{1}{|\Omega|} = \dfrac{\sum_{x \in E} 1}{|\Omega|} = \frac{|E|}{|\Omega|} \]

  + example: die
    + $\Omega = \{1, 2, 3, 4, 5 6\} \to |\Omega| = 6$

    <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Event</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">$|Event|$</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">$\Pr(Event) = \frac{|Event|}{6}$</th>
      </tr>
      </thead>
      <tbody>
      <tr> <td style="text-align: center;">Even</td> <td style="text-align: center;">$\{2, 4, 6\}$</td> <td style="text-align: center;">3</td> <td style="text-align: center;">3/6 = 1/2</td> </tr>
      <tr> <td style="text-align: center;">Square</td> <td style="text-align: center;">$\{1, 4\}$</td> <td style="text-align: center;">2</td> <td style="text-align: center;">2/6 = 1/3</td> </tr>
      </tbody>
    </table>

+ Dos and Don'ts
  + Valid expression
    + $\Pr(X \in \text{ Even})$: die - $\Pr(X \in \{2, 4, 6\}= 3/6 = 1/2$
    + $\Pr(\text{ Even }) \,\stackrel{def}{=} \Pr(X \in \text{ Even })$
  + less likely
    + $\Pr(3 \in \text{ Even }) = 0$
      + 3 is a (constant) random variable, so possible
      + the r.v. 3 is never in Even, so probability is 0
    + $\Pr( 4 \in \text{ Even }) = 1$
      + less common
      + check that you meant it
  + unlikely: $\Pr( x \in \text{ Even })$
    + previous anomaly on steriods, what is x?
    + $\forall\, x \in \{3, 7\}, \Pr(x \in \text{ Even })=0$, double check meant it



+ [Original Slides](https://tinyurl.com/y8rpqaq6)


### Problem Sets

0. What is the probability of drawing a Red Ace from a standard deck of cards?<br/>
  a. 1/52<br/>
  b. 2/52<br/>
  c. 4/52<br/>
  d. None of the above<br/>

  Ans: b <br/>
  Explanation: Both "heart ace" and "diamond ace" are "red ace". Since the sample space is uniform, the answer is 2/52.


1. Which of the following holds for every event A?<br/>
  a. $P(A) \geq 0$<br/>
  b. $P(A) \leq1$<br/>
  c. $P(A)+P(A^c)=1$<br/>
  d. $P(A)=P(A^c)$<br/>
  e. $A=\varnothing \implies P(A)=0$<br/>
  f. $P(A)=0 \implies A=\varnothing$<br/>
  
  Ans: abce<br/>
  Explanation
    + True. $0 \leq P(A) \leq 1$.
    + True. Same as above.
    + True.
    + False.
    + True. Note that $A \cap \varnothing = \varnothing, A \cup \varnothing = \varnothing$ for any $A$. $P(A)=P(A \cup \varnothing) = P(A)+P(\varnothing)$, hence $P(\varnothing)=0$.
    + False. Suppose a uniform sample space $\Omega$  has infinite number of elements. Then for some events  A  with finite size (i.e. $|A|$ is finite ), $P(A)=|A|/|\Omega|$.


2. Which of the following always hold for events  A  and  B ?<br/>
  a. $A \subseteq B \implies P(A) \leq P(B)$<br/>
  b. $P(A) \leq P(B) \implies A \subseteq B$<br/>

  Ans: a<br/>
  Explanation:
    + True.
    + False. Let  A  be the event of getting a head from a tossed coin, and B be the event of getting a tail.  P(A)=P(B)=12 , but A⊈B.


3. Which of the following implies  P(S−T)=P(S)−P(T)  for events  S  and  T ?<br/>
  a. $T \subseteq S$<br/>
  b. $T \subset S$<br/>
  c. $S = T$<br/>
  d. $S \subseteq T$<br/>

  Ans: abc<br/>
  Explanation:
    + Note that $P(S−T) = P(S \cup T)−P(T) .
    + When $T \subseteq S$, $T \subset S$, and $S=T$, we have $S \cup T=S$, hence $P(S−T)=P(S)−P(T)$.
    + When $S \subseteq T$, we have $S \cup T=T$, hence $P(S−T)=0$.


4. 50% of UCSD students play soccer, 40% play basketball, and 30% play both. What is the probability that a random UCSD student does not play any of the two games.<br/>
  a. 0<br/>
  b. 0.1<br/>
  c. 0.4<br/>
  d. 0.6<br/>

  Ans: c<br/>
  Explanation: Let $A$ be the event that a student play soccer, $B$ be the event that a student play basketball. Then $A \cap B$ is the event that a student play both. We already know $P(A)=0.5, P(B)=0.4, P(A \cap B)=0.3$, hence $P(A \cup B) = P(A)+P(B)−P(A \cap B)=0.6$. The probability that a random UCSD student does not play any of the two games is $1 − P(A \cup B)=0.4$.


5. Which of the following are events in the sample space $\Omega = \{1,2,3,4,5\}$?<br/>
  a. $\{1,2,3\}$<br/>
  b. $\varnothing$<br/>
  c. $\Omega$<br/>
  d. $\{1\}$<br/>
  e. $\{0,3,4\}$<br/>

  Ans: abcd<br/>
  Explanation
    + True.
    + True.
    + True.
    + True.
    + False. {0,3,4} is not a subset of $\Omega$.


6. For the uniform space  {1,2,…,10} , find:<br/>
  a. $P(\{\text{primes}\})$,<br/>
  b. $P(\{\text{multiples of 3}\})$.<br/>

  Ans: a. (0.4); b. (0.3)<br/>
  Explanation
    + $\{\text{primes}\}=\{2,3,5,7\}$. Its probability is $P(\{\text{primes}\}) = |\text{primes}||\Omega|=2/5$.
    + $\{\text{primes}\}=\{3,6,9\}$. Its probability is $P(\{\text{multiples of 3}\}) = |\text{multiples of 3}||\Omega|=3/10$.


7. A bag contains 5 red and 3 blue balls.<br/>
  a. Pick one ball at random and observe its random color. What is the size of the color sample space.<br/>
  b. What is P(blue)?<br/>
  c. Two balls added to the bag and now P(blue)=0.4. How many of the two balls are blue?<br/>
  d. Two balls are removed from the original bag and now P(blue)=0.5. How many of the two balls were blue?<br/>

  Ans: a. (2); b. (3/8); c. (1); d. (0)<br/>
  Explanation:
    + The sample space is $\{Red ,Blue\}$.
    + $\frac{|\{\text{blue}\}|}{|\Omega|} = 3/8 = 0.375$.
    + If $a$ blues balls are added, $P(\text{blue}) = \frac{3+a}{8+2}=0.4$. Hence $a=1$.
    + If $a$ blues balls are removed, $\frac{3−a}{8−2} = 0.5$. Hence $a=0$.


8. Six balls are numbered 1, 2, 3, 4, 5, and 6. What is the chance that the numbers on three balls, picked simultaneously and randomly, will sum to a multiple of 3?<br/>
  a. 1/3<br/>
  b. 1/4<br/>
  c. 2/5<br/>
  d. 4/15<br/>

  Ans: c<br/>
  Explanation: The number of ways to pick 3 balls is $\tbinom{6}{3}=20$. 8 of them have their sum as a multiple of 3.


9. A standard poker deck has 52 cards, of 13 ranks $\{A,2, \dots ,10,J,Q,K\}$  and 4 suits $\{diamonds,clubs,hearts,spades\}$. What is the probability that a hand of five cards contains:
  a. a queen of hearts,
  b. at least one queen,
  c. at least one heart?

  Ans: a. (5/52); b. (0.3412); c. (0.7785)<br/>
  Explanation:
    + The probability that there's no queen of hearts is $\tbinom{51}{5}/\tbinom{52}{5}=47/52$. Thus the probability that there's a queen of hearts is $1−47/52=5/52$.
    + Similar to above, probability that there's no queen is $\tbinom{48}{5}/\tbinom{52}{5}=0.65884$. Thus the probability that there's a queen is $1−0.65884=0.34115$.
    + Following the same principle, probability that there's a hearts is $1−\tbinom{39}{5}/\tbinom{52}{5}=0.7785$.



### Lecture Video

<a href="https://tinyurl.com/ya6hral5" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.4 Repeated Experiments

+ Composite experiments
  + experiments often consist of several parts
    + student: major, year, GPA
    + Advertisement: product, audience, cost
  + still viewed as a single experiment $\to$ outcomes more complex
    + 3-tuple: (CS, senior, 3.8), (book, teenage, $9.99)
    + sample space: Cartesian product
  + goal: combine simple, analyze complex

+ Independent repetitions
  + repetition
    + all experiments of same type
    + examples: daily temperatures, daily stock price, coin flips, die rolls, card draws
  + independent
    + different components unrelated
    + examples: first coin heads
      + second coin 50% heads / tails (independent)
      + second coin more likely heads (dependent)

+ Example: two coins
  + fair, independent flips
  + outcomes
    + coin 1 = {h, t}, coin 2 = {h, t}
    + $\Omega = \{hh, ht, th, tt\} = \{h, t\}^2$
  + probability
    + coin 1: $\Pr(\{h\}) = 1/2, \;\Pr(\{t\}) = 1/2$
    + coin 2: $\Pr(\{h\}) = 1/2, \;\Pr(\{t\}) = 1/2$
    + $\Pr(\{h, h\}) = 1/4, \;\Pr(\{h, t\}) = 1/4, \;\Pr(\{t, h\}) = 1/4, \;\Pr(\{t, t\}) = 1/4$
    + Cartesian power: $|\Omega| = 2^2 = 4$
    + 1 coin w/ $U \to$ 2 coins w/ $U$

    \[ \Pr(\text{hh}) = \Pr(\text{ht}) = \Pr(\text{th}) = \Pr(\text{tt}) = 1/|\Omega| = 1/4 \]

+ Example: two dice
  + fair, independent experiments
  + outcome: 
    + dice 1 = {1, 2, ..., 6}, dice 2 ={1, 2, ..., 6}
    + two dice: $\Omega = \{11, 12, \dots, 66\} = \{1, \dots, 6\} \to |\Omega| = 6^2 = 36$
  + probability
    + die 1: $\Pr(1) = \Pr(2) = \cdots = \Pr{6} =1/6$
    + die 2: $\Pr(1) = \Pr(2) = \cdots = \Pr{6} =1/6$
    + two dice: $\Pr(11) = \Pr(12) = \cdots = \Pr(66) = 1/36$
    + 1 die w/ $U \to$ 2 dice w/ $U$
    + $\Pr(11) = \Pr(12) = \Pr(21) = \cdots = \Pr(66) = 1/|\Omega| = 1/36$

+ Events
  + probability of event

    \[ \Pr(E) = \Pr(X \in E) = \sum_{x \in E} \Pr(x) \]

  + uniform space: $U \to \Pr(E) = |E|/|\Omega|$
  + example: 2 coins
    + $|\Omega| = 2^2 = 4$
    + $\Pr(\text{ Different outcomes }) = \Pr(\{\text{hh, tt} \}) = 2/|\Omega| = 2/4 = 1/2$
    + $\Pr(\text{ At least one h}) = \Pr(\{\text{ ht, th, hh }\}) = 3/|\Omega| = 3/4$
  + example: 3 coins
    + $|\Omega| = 2^3 = 8$
    + $\Pr(\text{ Alternating }) = \Pr(\{\text{ hth, tht }\}) = 2/8 = 1/4$
    + $\Pr(\text{ odd # h's }) = \Pr(\{\text{ htt, tht, tth, hhh }\}) = 4/8 = 1/2$

+ Sampling
  + many sources of randomness, e\.g., coin, die, ...
  + often sample (select) physical objects
    + patients in a study
    + customers at a restaurant
    + products for quality control
    + visitors to web pages
    + cards from a deck
    + balls from a urn
  + two sampling types: with or without replacement

+ Replacement
  + sequentially select physical objects
  + with replacement
    + <span style="color: cyan;">replace</span> (reuse) selected element
    + outcomes <span style="color: cyan;">can</span> repeat
    + experiments often <span style="color: cyan;">independent</span>
    + e.g., coins, dice
  + without replacement
    + <span style="color: cyan;">do not replace</span> (reuse) selected element
    + outcomes <span style="color: cyan;">cannot</span> repeat
    + experiments <span style="color: cyan;">dependent</span>
    + cf. difference largest for small $\Omega$

+ Sampling (selection) With / Without differences
  + __with__ replacement
    + repeat as if from scratch
    + same element able to be selected again
    + example: coin, die
  + __without__ replacement
    + repeat w/ smaller set
    + same element unable to be selected again
    + example: cards, people

+ Example: balls in a jar with replacement
  + activity
    + 1 yellow and 1 blue ball in a jar
    + pick one ball and then replace it
    + pick again
  + selection with replacement
  + second selection - from <span style="color: magenta;">original</span> set
  + outcome: $\Omega = \{\text{ yy }, \,\text{ yb }, \,\text{ by }, \,\text{ bb } \}$
  + probabilities
    + $|\Omega| = 4$
    + uniform space: $U \to \Pr(\text{ yy }) = \Pr(\text{ yb }) = \Pr(\text{ by }) = \Pr(\text{ bb }) = 1/4$

+ Example: balls in a jar without replacement
  + activity
    + 1 yellow and 1 blue ball in a jar
    + pick one ball and do not replace it
    + pick again
  + outcomes: $\Omega = \{\text{ yb }, \,\text{ by } \}$
  + selection without replacement
  + second selection - from a <span style="color: magenta;">subset</span>
  + probabilities
    + $\Pr(\text{ yy }) = \Pr(\text{ bb }) = 0, \;\Pr(\text{ yb }) = \Pr(\text{ by }) = 1/2$
    + $|\Omega| = 2$

+ Example: drawing cards
  + six cards: $\Omega = \{1, 2, 3, 4, 5, 6\}$
  + activity
    + draw one
    + w/o replacement, draw a second
  + outcomes
    + $\Omega = \{12, \dots, 16, 21, \dots, 26, \dots, 65\} =(6]^{\underline{2}} \gets$ 2-permutations of {1,2, ..., 6}
    + $|\Omega| = 6^{\underline{2}} = 6 \cdot 5 = 30$
  + probabilities
    + $i = j$ unable to happen
    + $i\neq j, \; \Pr(i, j) = 1/6 \cdot 1/5 = 1/30$
    + $\Pr(i, j) = 1/|\Omega| = 1/30$

+ Replacement summary
  + 2 selections

    <table style="font-family: arial,helvetica,sans-serif; width: 58vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;"></th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Original</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">with replacement</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">without replacement</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td style="text-align: center;">Description</td>
        <td style="text-align: center;"></td>
        <td style="text-align: center;">outcomes can repeat</td>
        <td style="text-align: center;">outcomes cannot repeat</td>
      </tr>
      <tr>
        <td style="text-align: center;">Sample space</td>
        <td style="text-align: center;">$\Omega_1$</td>
        <td style="text-align: center;">$\{(x, y): x, y \in \Omega_1 \} = \Omega_1^2$</td>
        <td style="text-align: center;">$\{(x, y): x, y \in \Omega_1, x \neq y \} = \Omega_1^{\underline{2}}$</td>
      </tr>
      <tr>
        <td style="text-align: center;">| Sample space |</td>
        <td style="text-align: center;">$|\Omega_1|$</td>
        <td style="text-align: center;">$|\Omega_1^2|$</td>
        <td style="text-align: center;">$|\Omega_1^{\underline{2}}| = |\Omega_1| \cdot (|\Omega_1| - 1)$</td>
      </tr>
      </tbody>
    </table>

  + probabilities if original sample space uniform

    <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;"></th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Original</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">with replacement</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">without replacement</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td style="text-align: center;">Uniformity</td>
        <td style="text-align: center;">Uniform</td>
        <td style="text-align: center;">Uniform</td>
        <td style="text-align: center;">Uniform</td>
      </tr>
      <tr>
        <td style="text-align: center;">$\Pr(\text{ element })$</td>
        <td style="text-align: center;">$1 / \Omega_1$</td>
        <td style="text-align: center;">$1 / |\Omega_1|^2$</td>
        <td style="text-align: center;">$1 / \left(|\Omega_1| \cdot (|\Omega_1| - 1) \right)$</td>
      </tr>
      </tbody>
    </table>

+ Order not matter
  + order matters: card $\Pr(5) = 1/6 \to \Pr(3) = 1/5$, stock $10 \to 50 \neq 50 \to 10$
  + sometimes order not matter: voting order in elections
  + when order not matter
    + tuple of outcome $\to$ set of outcomes
    + (2, 5), (5, 2) $\to$ {2, 5} $\implies$ Event {(2, 5), {5, 2)}}
    + (4, 4) $\to$ {4, 4}
  + $\to$: order matter, easier to calculate
  + $\gets$ order not matter, considering w/ & w/o replacement

+ order consideration with replacement
  + 2 cards $\in \{1, \dots, 6\}$ w/ replacement
  + equivalently, 2 dice
  + order
    + $\gets$, matter:
      + $\Pr(1, 1) = \Pr(1, 2) = \cdots = \Pr(6, 5) = \Pr(6, 6) = 1/36$
      + remaining uniform
    + $\to$, not matter
      + $\Pr(\{1, 2\}) = \Pr(\{(1, 2), (2,1)\}) = \Pr(1, 2) + \Pr(2, 1) = 2/36$
      + $\Pr(\{1, 1\}) = \Pr(1, 1) = 1/36$
      + not uniform
  + sanity check
    + $\sum$ Probabilities = 1
    + e.g., $\binom{6}{2} \cdot \frac{2}{36} + \binom{6}{1} \cdot \frac{1}{36} = \frac{5}{6} + \frac{1}{6} = 1$

+ Order consideration w/o replacement
  + 2 cards $\in \{1, \dots, 6\}$ sequentially w/o replacement
  + order:
    + $\gets$, matter:
      + $i \neq j, \,\Pr(i, j) = 1/6 \cdot 1/5 = 1/30$
      + remaining unifrom
    + $\to$, not matter:
      + $\Pr(\{1, 2\}) = \Pr(\{(1, 2), (2, 1)\}) = \Pr(1, 2) + \Pr(2, 1) = 2/30$
      + $\{1, 1\}$ not happened
      + remaining uniform
  + sanity check
    + probabilities sum to 1
    + $\binom{[6]}{2} \cdot \frac{2}{30} = 1$
  + alternative calculation
    + select both cards simultaneously
    + $\Omega = \{\{1, 2\}, \{1, 3\}, \dots, \{5, 6\}\} = \binom{[6]}{2}$
    + $|\Omega| = \binom{[6]}{2} = 15$, uniform space
    + $\Pr(\{1, 2\}) = 1/15$
    + sequential simultaneous same

+ Example: poker hand probabilities
  + deck: 52 cards, hand: 5 cards
  + $\Omega = \{\text{ possible hands } \} = \{\{2\heartsuit, 4\heartsuit, J\heartsuit, 4\heartsuit, 9\heartsuit\}, \{4\diamondsuit, 5\heartsuit, 6\clubsuit, 7\clubsuit, 8\spadesuit\}, \dots\}$
  
    \[\begin{align*}
      & \;{\color{Grey}{\approx 50 \cdot 50 \cdot 10 \cdot 100 = 2.5 \text{ M }}} \\
      |\Omega| = \binom{52}{5} = \frac{52!}{5! \cdot 47!} = \frac{52 \cdot 51 \cdot 50 \cdot 49 \cdot 48}{5 \cdot 4 \cdot 3 \cdot 2 \cdot 1} &= 2,598,960 \approx 2.6 \text{ million}
    \end{align*}\]
  + all hands equally likely $\to$ equiprobable
  + $\Pr(\{2\spadesuit, 4\clubsuit, A\diamondsuit, A\heartsuit, A\spadesuit\}) \stackrel{U}{\approx} 1/2.6 \text{ M }$



+ [Original Slides](https://tinyurl.com/yape5urh)


### Problem Sets

0. Which of the following are independent repetitive experiments?<br/>
  a. card draws with replacement<br/>
  b. card draws without replacement<br/>
  c. neither<br/>

  Ans: a<br/>
  Explanation: Card draws with replacement will be independent repetitive experiments.


1. You have two fair coins. If you flip a head with the first coin, what is the probability of flipping a head with the second?<br/>
  a. \(1\)<br/>
  b. \(0\)<br/>
  c. \(\frac{1}{2}\)<br/>
  d. \(\frac{1}{4}\)<br/>

  Ans: c<br/>
  Explanation: Flipping the first and the second coin are independent.


2. A bag has 3 red and 3 green apples. You start by randomly selecting one red apple from the bag. Which of the following has the highest probability?<br/>
  a. Select another red apple after replacing the first.<br/>
  b. Select another red apple without replacing the first.<br/>
  c. Select a green apple after replacing the first red apple.<br/>
  d. Select a green apple without replacing the first red apple.<br/>

  Ans: d<br/>
  Explanation:
    + After replacement, there are 3 green and 3 red apples. The probability to select a red one is $\frac{1}{2}$.
    + Without replacement, there are 3 green and 2 red apples. The probability to select a red one is $\frac{2}{5}$ .
    + After replacement, there are 3 green and 3 red apples. The probability to select a red one is \frac{1}{2}$.
    + Without replacement, there are 3 green and 2 red apples. The probability to select a red one is \frac{3}{5}$.


3. Roll two fair and distinguishable six-sided dice. What is the probability that the outcome of the second die is strictly greater than the the first?

  Ans: 5/12<br/>
  Explanation: The outcomes that give rise to this event are $(1,2),(1,3), \dots,(1,6), (2,3),(2,4), \dots,(2,6),(3,4),\dots,(3,6),(4,5),(4,6),(5,6)$, where in each ordered pair, the numbers correspond to the first and second die respectively. It can be easily verified that there are  15  such outcomes, each of probability 1/36, thus 15×1/36=5/12.


4. 5 Engineers and 3 artists align at random along line. What is the probability that the first and last are Engineers?<br/>
  a. 3/14<br/>
  b. 8/16<br/>
  c. 9/14<br/>
  d. 5/14<br/>

  Ans: <span style="color: magenta;">d</span><br/>
  Explanation
    + The 5 engineers and 3 artists can line up in $\tbinom{8}{3}$ ways, each corresponding to one choice of the the 3 artist locations out of the total 8. Note that all these arrangements are equally likely, hence this space is uniform.
    + If the first and last are engineers, then there are 6 locations left, and the number of ways to choose the locations of the artists is $\tbinom{6}{3}$.
    + Hence the probability that the first and last are engineers is $\tbinom{6}{3}/\tbinom{8}{3}=6⋅5⋅4/8⋅7⋅6=5/14$.
    + In the sequential probability section we will see an easier way to find the answer.


5. Which of the following are distributed uniformly when rolling two fair dice?<br/>
  a. The outcome (number) of the first die, e.g., "4", or "6",<br/>
  b. The ordered pair of outcomes, e.g., \((1,3)\), \((3,1)\), or \((6,6)\),<br/>
  c. The unordered pair of outcomes, e.g., \(\{1,3\}\) or \(\{6,6\}\),<br/>
  d. the difference between the first and second outcomes, e.g. \((1,3)\to-2\),<br/>
  e. the difference between the first and second outcomes mod 6, e.g. \((1,3)\to(-2)_6=4\),<br/>
  f. The absolute value of difference\((1,3)\to|1-3|=2\),<br/>
  g. The lower of the two outcomes, e.g. \((1,3)\to1\),<br/>
  h. the sum of the two outcomes, e.g. \((1,3)\to4\).<br/>

  Ans: abe<br/>
  Explanation
    + True.
    + True.
    + False. $P(\{1,3\})=236 \neq P(\{6,6\})=136$.
    + False. $P(\text{difference is 0})=\frac{6}{36} \neq P(\text{difference is 1})=\frac{5}{36}$.
    + True.
    + False. $P(\text{absolute difference is 0})=\frac{6}{36} \neq P(\text{absolute difference is 1})=\frac{10}{36}$.
    + False.  



### Lecture Video

<a href="https://tinyurl.com/yb4x68hz" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.5 Games of Chance

+ Roulette
  + basics
    + history
      + where: France
      + time: 17th century
      + who: Blaise Pascal
      + letter wheel
    + wheel: integers from 1 to 36 and 0
    + spin wheel and a ball
    + bet on where ball will land
  + bets: place one token bet and return
    + number: 36
    + red, black, even, odd, low, high: 2
    + 1-12, 13-24, 25-36, $(x)_3 =0, (x)_3 = 1, (x)_3 = 2$: 3

+ Probabilities
  + individual outcomes
    + $\Omega = \{0, 1, \dots, 36\} \to |\Omega| = 37$
    + $\Pr(0) = \Pr(1) = \cdots = \Pr(36) = \frac{1}{|\Omega|} = \frac{1}{37} \gets$  should be 1/36 but 1/37 because of 0
  + events
    + Even = {2, 4,..., 36} $\to |\text{Even}| = 18 \to \Pr(\text{Even}) = \frac{|\text{Even}|}{|\Omega|} = \frac{18}{37} \gets$ should be half but slightly smaller because of 0
    + $|\text{Odd}| = |\text{Red}| = |\text{Black}| = |\text{1 to 18}| = |\text{19 to 36}| = 18 \to \Pr(\dots) = 18/37$
    + $|\text{1 to 12}| = \cdots = |(x)_3 = 0| = 12 \to \Pr(\dots) = 12/37$
  
+ Win or lose?
  + how much can you expect to make on Roulette?
    + one game: random
    + many games: more predictable
  + expected profit or loss
  + simplicity: each bet $1
  + evaluate two bet types
    + single-number bets
      + always bet on single number, e.g., 6
      + \# games: $n \gg 1$
      + bet: 1 each game $\to$ total n
      + get: correct $\approx \frac{n}{37}$ games $\to$ 36 each $\to$ total = $\frac{36}{37}n$
      + gain: $\frac{36}{37}n - n = -\frac{1}{37}n$
      + lose: &cent;2.7/bet = 2.7% $\gets$ house edge (advantage)
    + bet red
      + always bet red
      + \# games: $n \gg 1$
      + bet: 1 each game $\to$ total = n
      + get: correct $\frac{18}{37}n$ games $\to$ 2 each $\to$ total = $\frac{36}{37}n$
      + gain: $\frac{36}{37}n - n = -\frac{1}{37}n$
      + house edge $\approx 2.7\%$

+ House edge in other games

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/ybolxett" ismap target="_blank">
      <img src="img/t05-03.png" style="margin: 0.1em;" alt="House edge and sigma of various games" title="House edge and sigma of various games" width=350>
    </a>
  </div>
  


+ [Original Slides](https://tinyurl.com/ybolxett)


### Problem Sets

0. What is the probability that two cards drawn from a standard deck without replacement have the same rank?<br/>
  a. 1/13<br/>
  b. 1/17<br/>
  c. 2/52<br/>
  d. None of the above<br/>

  Ans: b<br/>
  Explanation:
    + There are 13 different ranks.
    + The number of ways to two cards from a specific rank is (4 choose 2).
    + Hence the probability is 13 * (4 choose 2) / (52 choose 2) = 3 / 51 = 1/17


1. What is the probability that a random four-card hand consists of a single suit?<br/>
  a. $\frac{4}{52}$<br/>
  b. $\frac{13}{52}$<br/>
  c. $\tbinom{13}{4}/\tbinom{52}{4}$<br/>
  d. $\tbinom{4}{1}⋅\tbinom{13}{4}/(\tbinom{52}{4}$<br/>

  Ans: d<br/>
  Explanation: There are $\tbinom{4}{1}$ ways to choose the suit (e.g. hearts) and $\tbinom{13}{4}$ ways to draw 4 cards from this suit. Yet the total number of ways to draw 4 cards is $\tbinom{52}{4}$. Hence, the probability is $4 \tbinom{13}{4} / \tbinom{52}{4}$.


2. Find the probability that a five-card hand contains:<br/>
  a. the ace of diamonds,<br/>
  b. at least an ace,<br/>
  c. at least a diamond.<br/>

  Ans: a. (0.09615); b. (0.3412); c. (0.7785)<br/>
  Explanation
    + The number of hands containing the ace of diamonds is $\tbinom{51}{4}$, corresponding to the choice of the remaining 4 cards from the other 51. Hence the probability is $\tbinom{51}{4}/\tbinom{52}{5}=5/52$.
    + The number of ways to draw 5 cards without any ace is $\tbinom{48}{5}$. By the complement rule, the answer is $1−\tbinom{48}{5}/\tbinom{52}{5}=0.3412$.
    + The number of ways to draw 5 cards without any diamond is $\tbinom{39}{5} \cdot 1 − \tbinom{39}{5}/\tbinom{52}{5}=0.7785$.


3. Five cards are dealt from a poker deck. What is the probability of:<br/>
  a. three-of-a-kind (three cards of one rank and two cards of two other ranks),<br/>
  b. two pairs (two pairs of same-rank cards),<br/>
  c. one pair (a pair of same-rank cards, and three cards of three different ranks).<br/>

  Ans: a. (0.021128); b. (0.047539); c. ()<br/>
  <span style="color: magenta;">Explanation</span><br/>
    + We deal with the 3 cards of the same rank first, and then the 2 remaining cards with different ranks. The are 13 ranks. The number of ways to get 3 cards of a particular rank, (e.g. ace) is $\tbinom{4}{3}=4$. In total the number of ways to get 3 cards of the same rank is $13 \cdot 4$. The remaining 2 cards cannot have the same rank as the one we choose the first step, so there are 12 cards left. Since 4 suits can be chosen for each card, the number of ways in total is  \tbinom{12}{2}⋅4^22$. The answer is $13 \cdot 4 \cdot \tbinom{12}{2} \cdot 42/ \tbinom{52}{5}=0.0211$.
    + We first deal with the rank of the 2 pairs, and then the one left over. There are $\tbinom{13}{2}$ ways to choose 2 ranks out of 13. The number of ways get 2 cards of a probability of a particular rank, (e.g. ace), is $\tbinom{4}{2}$. We do the same for both the pair, so the total number of ways is $\tbinom{13}{2} \cdot \tbinom{4}{2}^2$,  For the one left over, there are 11 ranks left that can be chosen, and 4 suit can be chosen for each rank. The number of ways is $11 \cdot 4$. The answer is $\tbinom{13}{2} \cdot \tbinom{4}{2}^2 \cdot 11 \cdot 4/ \tbinom{52}{5}=0.0475$.
    + We first deal with the rank of the pair, and then the three left over. There are $\tbinom{13}{1}=13$ ways to choose a ranks out of 13. The number of ways get 2 cards of a probability of a particular rank, (e.g. ace), is $\tbinom{4}{2}$. The total number of ways is $13 \cdot \tbinom{4}{2}$, For the one left over, there are 12 ranks left that can be chosen, and 4 suit can be chosen for each rank. The number of ways is $\tbinom{12}{3} \cdot 43$. The answer is $13 \cdot \tbinom{4}{2} \cdot \tbinom{12}{3} \cdot 43/ \tbinom{52}{5}=0.4226$.
    + [5-CARD POKER HANDS](https://tinyurl.com/6mh5em)


4. A 52-card deck is randomly split into four 13-card hands. Find the probability that:<br/>
  a. each hand has an ace, <br/>
  b. one hand has all four aces.<br/>

  Ans: a. (0.105498); b. (0.002641)<br/>
  Explanation
    + There are $4!$ ways to assign 4 aces to 4 hands. There are $\binom{48}{12,12,12,12}$  ways to assign the remaining 48 cards equally to 4 hands (12 for each). The answer is $4! \binom{48}{12,12,12,12}/\tbinom{52}{13,13,13,13}=0.1055$.
    + There are 4 ways to assign all 4 aces to one hand. There are $\tbinom{48}{9,13,13,13}$ ways to assign the remaining 48 cards to 4 hands (9 for the one which gets 4 aces, and 13 for the others). The answer is $4 \tbinom{48}{9,13,13,13}/\tbinom{52}{13,13,13,13}=0.0106$.


5. Assume that in blackjack, an ace is always worth 11, all face cards (Jack, Queen, King) are worth 10, and all number cards are worth the number they show. Given a shuffled deck of 52 cards:<br/>
  a. What is the probability that you draw 2 cards and they sum 21?<br/>
  b. What is the probability that you draw 2 cards and they sum 10?<br/>
  c. Suppose you have drawn two cards: 10 of clubs and 4 of hearts. You now draw a third card from the remaining 50. What is the probability that the sum of all three cards is strictly larger than 21?<br/>

  Ans: a. (0.048265); b. (<span style="color: magenta;">0.04072398</span>); c. (27/50)<br/>
  Explanation
    + The possible combinations that sum to 21 are (A, 10), (A, J), (A, Q), (A, K). The number of them is 16⋅4. The answer is $16⋅4/\tbinom{(52}{2}=0.0483$.
    + The possible combinations that sum to 10 are (2, 8), (3, 7), (4, 6), (5, 5). The number of them is $3 \cdot 4^2 + \tbinom{4}{2}$. The answer is $(3⋅4⋅4+ \tbinom{4}{2})/\tbinom{52}{2}=0.0407$.
    + To exceed 21, the third card belongs to { A, 8, 9, 10, J, Q, K }. As one 10 was drawn, $4 \cdot 7−1=27$ choices are left. The answer is $27/50=0.54$.


6. Three dice are rolled. What is the probability that the three outcomes<br/>
  a. contain at least a `1', e.g., 5,1,2,<br/>
  b. are all distinct, e.g., 3,2,5,<br/>
  c. in the order rolled, form an increasing consecutive sequence, e.g., 2,3,4.<br/>
  d. can be arranged to form a consecutive sequence, e.g., 3,2,4 that can form 2,3,4?<br/>

  Ans: a. (0.421296); b. (5/9); c. (0.01852); d. (1/9)<br/>
  Explanation:
    + Each roll differs from "1" with probability 5/6, hence "1" does not appear in any roll with probability $(5/6)^3$ , because they are independent, and "1" does appear with probability $1−(5/6)^3=0.4213$.
    + There are $6^3$ possible ordered outcomes. By the "Partial Permutations" lecture, there are $6^{\underline{3}}$ distinct-element sequences of length 3 over $1 \dots 6$. Hence the probability is $6^{\underline{3}}/6^3=0.5556$.
    + There are four increasing sequences (starting with 1...4), hence $4/6^3=0.0185$.
    + Adding the permutations to part 2, the number of sequences is $4 \cdot 3!$. Hence, the answer is $4 \cdot 3!/6^3=1/9=0.1111$.


7. An instructor assigns 10 problems and says that the final exam will consist of a random selection of 5 of them. If a student knows how to solve 7 of the problems, what is the probability that he or she will answer correctly<br/>
  a. all 5 problems,<br/>
  b. at least 4 problems?<br/>

  Ans: a. (1/12); b. (<span style="color: magenta>1/2</span>)<br/>
  Explanation
    + The student answers all 5 correctly in the event that all 5 questions appear from the 7 questions that he/show knows to solve. Thus the probability is $\tbinom{7}{5}/\tbinom{10}{5}=0.083333$.
    + The student answers at least 4 correctly at least 4 questions appear from the 7 questions that he/show knows to solve. Thus the probability is $\tbinom{7}{4} \times \tbinom{3}{1}/ \tbinom{10}{5}+\tbinom{7}{5}/\tbinom{10}{5}=0.50$.
    + [Slader](https://tinyurl.com/yajz23pj)


8. Let $X$ be the number of draws from a deck, without replacement, till an ace is observed. For example for draws Q, 2, A, $X=3$. Find:<br/>
  a. $P(X = 10)$,<br/>
  b. $P(X = 50)$,<br/>
  c. $P(X < 10)$?<br/>

  Ans: a. (0.0424047); b. (0); c. (0.54415)<br/>
  <span style="color: magenta;">Explanation</span>
    + This equals the probability that one randomly inserts four aces into the deck, of which one falls exactly at the 10th position, and the other three in the following $42$ positions. This probability is then given by $\tbinom{42}{3}/\tbinom{52}{4}=328/7735$.
    + The probability is zero since this is impossible.
    + Consider the probability of its complement, which equals the probability that one randomly inserts four aces into the deck, and all of them fall in the last 43 positions. The probability of this happening is $\tbinom{43}{4}/\tbinom{52}{4}=3526/7735$ and thus the probability is $1−3526/7735=4209/7735$.



### Lecture Video

<a href="https://tinyurl.com/ycdy6shj" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.6 Axiomatic Formulation

+ Three axioms
  + Non-negative: 
  
    \[ \forall\, A, \Pr(A) \geq 0 \]

  + Unitary: 
  
    \[ \Pr(\Omega) = 1 \]

  + Addition rule:

    \[ A, B \text{ disjoint} \implies P(A \cup B) = P(A) + P(B) \]

    + generalization

      \[ A_1, A_2, \dots \text{ disjoint} \implies P(A_1 \cup A_2 \cdots) = P(A_1) + P(A_2) + \cdots \]

    + applied to countable unions only

+ Building results
  + complement rule
  + subtraction, nested sets, general sets
  + inclusion-exclusion

+ Complement rule
  + complement rule for counting: $|A^c| = |\Omega| - |A|$
  + proof:
    + $A \cup A^c = \Omega$
    + $\Pr(A) + \Pr(A^c) = \Pr(A \cup A^c) = \Pr(\Omega) = 1$
  + complement rule for probability: $\Pr(A^c) = 1 - \Pr(A)$

+ Subtraction rule - nested sets
  + complement rule: $A \subseteq \Omega, \Pr(A^c) = 1 - \Pr(A) \text{ s.t. } \Pr(\Omega - A) = \Pr(\Omega) - \Pr(A)$
  + generalize: $A \subseteq B \implies \Pr(B-A) = \Pr(B) - \Pr(A)$
  + proof:
    + $A \subseteq B \implies B = A \cup (B-A)$
    + $\Pr(B) = \Pr(A \cup (B-A)) = \Pr(A) + \Pr(B-A)$
  + subtraction rule for nested sets: $\Pr(B-A) = \Pr(B) - \Pr(A)$

+ Subtraction rule - general sets
  + Nested: $A \subseteq B \;\text{ s.t. }\; \Pr(B-A) = \Pr(B) - \Pr(A)$
  + general: $\forall\, A, B \;\text{ s.t. }\; \Pr(B-A) = \Pr(B) - \Pr(A \cap B)$
  + proof:
    + $B - A = B - (A \cap B),\; A \cap B \subseteq B$
    + $\Pr(B-A) = \Pr(B - (A \cap B)) = \Pr(B) - \Pr(A \cap B)$

+ Inclusion-exclusion
  + $\Pr(A \cup B) = \Pr(A) + \Pr(B) - \Pr(A \cap B)$
  + proof:
    + $A \cup B = A \cup (B-A)$
    + $\Pr(A \cup B) = \Pr(A \cup B-A) = \Pr(A) + \Pr(B-A) = \Pr(A) + \Pr(B) - \Pr(A \cap B)$

+ More sets
  + two sets: $\Pr(A \cup B) = \Pr(A) + \Pr(B) - \Pr(A \cap B)$
  + three sets: 
  
    \[\begin{align*}
      \Pr(A \cup B \cup C) &= \Pr(A) + \Pr(B) + \Pr(C) \\
      &\hspace{1.0em} - \Pr(A \cap B) - \Pr(A \cap B) - \Pr(B \cap C) \\
      &\hspace{1.0em} + \Pr(A \cap B \cap C)
    \end{align*}\]

  + n sets

    \[ \Pr\left(\bigcup_{i=1}^n A_i \right) = \sum_{1 \leq i \leq n} \Pr(A_i) - \sum_{1\leq i < j \leq n} \Pr(A_i \cap A_j) + \cdots + (-1)^{n-1} \Pr\left( \bigcap_{i=1}^n A_i \right) \]


+ [Original Slides](https://tinyurl.com/yafgbra7)


### Problem Sets

0. Does $P(A)=0$ imply that A is the empty set?<br/>
  a. Yes<br/>
  b. Not necessarily<br/>

  Ans: b<br/>
  Explanation: It is possible that $P(A)=0$ for a non-empty set A.


1. For any three events A, B, and C, we have $P(B)=$<br/>
  a. $P(A \cap B)+P(B \cap C)+P(B \cap A^c \cap C^c)$<br/>
  b. $P(A \cap B)+P(B \cap C)−P(A \cap B \cap C)+P(B \cap A^c \cap C^c)$<br/>
  c. $P(A^c \cap C^c)+P(A \cap B)+P(B \cap C)$<br/>
  d. $P(\Omega)−P(A)−P(C)+P(A \cap B \cap C)$<br/>

  Ans: b<br/>
  Explanation
    + False. It is $P(B)+P(A \cap B \cap C)$.
    + True.
    + False. This includes the events outside of the three circles.
    + False. Same as above.


2. Under which of the following probability assignments does $S=\{a_1,a_2,a_3\}$ become a probability space?<br/>
  a. $P(a_1)=0.2, P(a_2)=0.3, P(a_3)=0.4$<br/>
  b. $P(a_1)=0.2, P(a_2)=0.3, P(a_3)=0.5$<br/>
  c. $P(a_1)=0.3, P(a_2)=−0.2, P(a_3)=0.9$<br/>
  d. $P(a_1)=0.2, P(a_2)=0, P(a_3)=0.8$<br/>

  Ans: bd<br/>
  Explanation: Two necessary conditions:
    1. The probability  P  of the events satisfies  0≤P≤1 .
    2. All  P s sum up to 1.


3. Which of the following always holds?<br/>
  a. $A \subseteq B \implies P(A) < P(B)$,<br/>
  b. $A \subseteq B \implies P(A) \leq P(B)$,<br/>
  c. $A \subset B \implies P(A) < P(B)$,<br/>
  d. $A \subset B \implies P(A) \leq P(B)$.<br/>

  Ans: bc<br/>
  Explanation: The only tricky part may be the third. Note that because elements may have 0 probabilities, non-empty events may also have zero probability. Hence  A  may be a strict subset of  B  and yet have the same probability. For example, if the sample space is $\{a,b\}$ and $P(a)=1$ while $P(b)=0$, then $P(\{a\})=P(\{a,b\})$.


4. Which of the following statements are true?<br/>
  a. If $P(E)=0$ for event $E$, then $E=\varnothing$.<br/>
  b. If $E=\varnothing$, then $P(E)=0$.<br/>
  c. If $E1 \cup E2=\Omega $, then $P(E1)+P(E2)=1$.<br/>
  d. If $P(E1)+P(E2)=1$, then $E1 \cup E2=\Omega $.<br/>
  e. If $E1 \uplus E2=\Omega $, then $P(E1)+P(E2)=1$.<br/>
  f. If $P(E1)+P(E2)=1$, then $E1 \cup E2=\Omega $.<br/>

  Ans: be<br/>
  Explanation
    + False. $E$ is not necessary to be $\varnothing$.
    + True.
    + False. Let $\Omega=\{1,2,3\},E_1=\{1,2\},E_2=\{2,3\}$. $E_1 \cup E_2=\Omega$, but $P(E_1)+P(E_2)=4/3$.
    + False. Let $\Omega=\{1,2,3\},E_1=\{1,2\},E_2=\{1\}$.  $P(E_1)+P(E_2)=1$, but $E1 \cup E2 \neq \Omega$.
    + True.
    + False. Same as option 4.


5. Suppose A, B are events that $P(A)=0.65$, $P(B)=0.5$ and $P(A \cap B)=0.25$. What are the following probabilities?<br/>
  a. $P(A^c)$<br/>
  b. $P(B^c)$<br/>
  c. $P(A \cup B)$<br/>
  d. $P(A−B)$<br/>
  e. $P(B−A)$<br/>
  f. $P(A \nabla B)$<br/>
  g. $P((A\cup B)^c)$<br/>

  Ans: a.(0.35); b. (0.5); c(0.9); d. (0.4); e. (0.25); f. (0.65); g. (0.1)<br/>
  Explanation
    + $P(A^c)=1−P(A)=0.35$.
    + $P(B^c)=1−P(B)=0.5$.
    + $P(A \cup B)=P(A)+P(B)−P(A \cap B)=0.9$.
    + $P(A−B)=P(A\cup B)−P(B)=0.4$.
    + $P(B−A)=P(A \cup B)−P(A)=0.25$.
    + $P(A \nabla B)=P(A \cup B)−P(A \cap B)=0.65$.
    + $P((A \cup B)^c)=1−P(A \cup B)=0.1$.


6. Let $P$ be a probability function on $S=\{a_1,a_2,a_3\}$. Find $P(a_1)$ if:<br/>
  a. $P({a_2,a_3})=3P(a_1)$<br/>
  b. $P(a_1)=2P(a_2)=3P(a_3)$<br/>

  Ans: a. (0.25); b. (6/11)<br/>
  Explanation
    + We have $P(\{a_2,a_3\})=3P(a_1)$ and $P(a_1)+P(\{a_2,a_3\})=1$. Solving the equations we have $P(a_1)=0.25$.
    + We have $P(a_1)=2P(a_2)=3P(a_3)$ and $P(a_1)+P(a_2)+P(a_3)=1$. Solving the equations we have $P(a_1)=6/11$.


7. Let $X$ be distributed over $\Omega=\{1,2,\dots,100\}$ with $P(X=i)=\frac{i}{k}$ for some integer $k$. Find:<br/>
  a. $k$<br/>
  b. $|E|$ where $E={x|x \in \Omega, x \text{ is multiples of 3}\}$,<br/>
  c. $P(E)$ .<br/>

  Ans: a. (5050); b. (33); c. (0.333267)<br/>
  Explanation:
    + $K=1+2+…+100=(1+100) \cdot 100/2=5050$.
    + $E=\{3,6,9,…,96,99\}$, hence $|E|=33$.
    + $(3+6+...+96+99) / 5050 = 102 \cdot 33/ (2 \cdot 5050) = 1683/5050 = 0.333267$.  Note that, as could be expected, this probability is very close to $1/3$.


8. Consider a die where the probability of rolling $1,2,3,4,5$ and $6$ are in the ratio $1:2:3:4:5:6$. What is the probability that when this die is rolled twice, the sum is $7$?

  Ans: 0.126984<br/>
  Explanation: Let $p$ be the probability of rolling a 1, then for $i=1,2,3,\dots,6$, the probability of rolling $i$ is $i \cdot p$. These probabilities sum to  $21p$, which must be 1, hence $p=1/21$. Hence The probability that the sum is 7 is $2 \cdot (1 \cdot 6+2 \cdot 5+3 \cdot 4)/(21 \cdot 21)=56/21^2=8/63$.


9. Jack solves a Math problem with probability 0.4, and Rose solves it with probability 0.5. What is probability that at least one of them can solve the problem?<br/>
  a. 0.7<br/>
  b. 0.9<br/>
  c. 0.6<br/>
  d. Not enough information<br/>

  Ans: d<br/>
  Explanation: Let A be the event that Jack solves the problem, B be the event that Rose solves the problem. $P(A \cup B)=P(A)+P(B)−P(A \cap B)$, but $P(A \cap B)$  is missed here.



### Lecture Video

<a href="https://tinyurl.com/yafgbra7" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 5.7 Inequalities

+ Probability of null event
  + $\Pr(\Omega) = 1$
  + what about $\Pr(\varnothing)?$
    + $\Omega = \varnothing \cup \Omega \to \Pr(\Omega) = \Pr(\varnothing \cup \Omega) = \Pr(\varnothing) + \Pr(\Omega)$
    + $\Pr(\varnothing) = 0$
  + the null event has nil probability

+ Probability value range
  + $0 \leq \Pr(A) \leq 1$
  + $\Pr(A) \geq 0 \quad$ show: $\Pr(A) \leq 1$
    + $A \cup A^c = \Omega$
    + $\Pr(A) \leq \Pr(A) + \Pr(A^c) = \Pr(A \cup A^c) = \Pr(\Omega) = 1$
    + $\Pr(A) \leq 1$
  + probability always between 0 and 1

+ Subset
  + $A \subseteq B \to \Pr(A) \leq \Pr(B)$
  + proof
    + $A \subseteq B \to B = A \cup (B - A)$
    + $\Pr(B) = \Pr(A \cup (B - A)) = \Pr(A) + \Pr(B-A) \geq \Pr(A)$

+ Union
  + $\max(\Pr(A), \Pr(B)) \leq \Pr(A \cup B) \leq \Pr(A) + \Pr(B)$
  + proof
    + left $\leq$: $A, B \subseteq A \cup B \to \Pr(A), \Pr(B) \leq \Pr(A \cup B)$
    + right $\leq$: $\Pr(A \cup B) = \Pr(A) + \Pr(B) - \Pr(A \cap B)$
  + Union bound: $\Pr(A \cup B) \leq \Pr(A) + \Pr(B) \to$ very useful

+ The Linda problem
  + statement: Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations.
  + which is more probable?
    1. Linda is a bank teller
    2. Linda is a bank teller and is active in the feminist movement
  + A. Tversky and D. Kahneman, "Extensional vs. Intuitive Reasoning: the Conjunction Fallacy in Probability Judgment", Psychological Review, 1983
  + survey: 88 UBC students answered 85% bank teller & activist more likely
    + B - bank teller
    + A - active in feminist movement
  + $\Pr(B)$ vs. $\Pr(B \cap A) \quad B \supseteq (B \cap A): \Pr(B) \geq \Pr(B \cap A)$
  + irrational concept of probability

+ Related questions
  + several conjunction fallacy problems
  + Björn Borg
    + preeminent tennis player of late '70's
    + 6 French opens, Wimbledon '76-'80
  + 1980: suppose Björn reaches the '81 Wimbledon Finals
    + more likely?
      + A. Björn will lose first set
      + B. Björn will lose first set but win match
    + Survey: 72% choose B $\to$ again $B \subseteq A \to \Pr(B) \leq \Pr(A)$
  + 1981: reach final, won first set, lost match to John McEnroe
  + 1983: age 26, retired

+ The ultimate reward
  + T & K: many probability-preception studies
  + 1996: Tversky passed away
  + 2002: Kahneman won Economics Nobel Prize
  + "for integrating insights from psychological research into economic science, especially regarding human judgment and decision-making under certainty"

+ Criticism
  + multiple choice and ranking questions often disjoint
    + coffee or tea
    + tea hot or cold
    + rank Mon, Tue, or Wed
  + often, when not explicitly disjoint, we still interpret them as such
    + with milk or with milk and sugar $\implies$ milk only or milk and sugar
    + meet Monday or Monday and Thuesday $\implies$ Mon only or Mon and Tue
  + students may have similarly interpreted the Linda question
    + teller or teller and activist $\implies$ teller only or teller and activist
    + simply thought Linda more likely active in movement than not

+ Bottom line
  + humans only moderately good at estimating probability


+ [Original Slides](https://tinyurl.com/ybgs9muc)


### Problem Sets

0. Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. Which is more probable?<br/>
  a. Linda is a bank teller<br/>
  b. Linda is a bank teller and is active in the feminist movement<br/>

  Ans: a<br/>
  Explanation: It is more probable that Linda is a bank teller than Linda is both a bank teller and an activist.


1. Which of the following holds for all events A and B

  1.1 in any probability space:<br/>
  
  a. $A \supseteq B \implies P(A) \geq P(B)$<br/>
  b. $P(A) \geq P(B) \implies A \supseteq B$<br/>
  c. $|A| \geq |B| \implies P(A) \geq P(B)$<br/>
  d. $P(A) \geq P(B) \implies |A| \geq |B|$<br/>

  1.2 in any uniform probability space:

  a. $A \supseteq B \implies P(A) \geq P(B)$<br/>
  b. $P(A) \geq P(B) \implies A \supseteq B$<br/>
  c. $|A| \geq |B| \implies P(A) \geq P(B)$<br/>
  d. $P(A) \geq P(B) \implies |A| \geq |B|$<br/>
  
  Ans: 1.1 (xacd, a); 1.2 (xabcd, acd)<br/>
  Explanation
    + 1.1
      + $A supseteq B \implies P(A)=P(B)+P(A∖B) \geq P(B)$.
      + $A$ and $B$ can be nonempty and disjoint with $P(A) \geq P(B)$, then $A$ does not contain $B$.
      + $B$ can be a singleton with higher probability than a set $A$ with two elements.
      + Similar counter-example to 3.
    + 1.2
      + Follows from the result for general spaces.
      + Similar counter-example to part a.
      + I uniform sample spaces $S$, for any event $E$, $P(E)=|E|/|S|$, hence $|A| \geq |B| \implies P(A) \geq P(B)$.
      + Again, follows since for any event $E$, $P(E)=|E|/|S|$.


2. Let $\Omega$ be any sample space, and $A,B$ are subsets of $\Omega$ . Which of the following statements are always true?<br/>
  a. If $|A|+|B| \geq |\Omega |$, then $P(A \cup B)=1$<br/>
  b. If $|A|+|B| \geq |\Omega |$, then $P(A)+P(B) \geq 1$<br/>
  c. If $P(A)+P(B)>1$, then $A \cap B \neq \varnothing$<br/>
  d. If $P(A)+P(B)>1$, then $P(A \cup B)=1$<br/>

  Ans: xbc, c<br/>
  Explanation: Let  Ω={1,2,3} , and  P(1)=P(2)=0.1,P(3)=0.8 .
    + False. Let $A=B=\{1,2\}$. $|A|+|B|=4>|Ω|$, but $P(A \cup B)=0.2$.
    + False. Let $A=B=\{1,2\}$. $|A|+|B|=4>|Ω|$, but $P(A)+P(B)=0.4$.
    + True.
    + False. Let $A=B=\{3\}$. $P(A)+P(B)=1.6>1$, but $P(A \cup B)=0.8$.



### Lecture Video

<a href="https://tinyurl.com/yb96hxf6" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 5

+ [Original Lecture NB](https://tinyurl.com/yd39fhof)

+ [Local Lecture NB](src/Topic05_Lecture.ipynb)

+ [Local Lecture Python code](src/Topic05_Lecture.py)

+ principal tool for visualization: [Matplotlib](https://matplotlib.org/#)

+ fundamental package for statistics in Python: [Numpy](http://www.numpy.org/)



## Programming Assignment 5

+ [Original HW NB](https://tinyurl.com/ycup4khk)

+ [Local HW NB](src/HW_Topic05.ipynb)

+ [Local HW Python code](src/HW_Topic05.py)


1. Which of the following is the approximately correct output for: `largest_face( [7,4], 2)`<br/>
  a. 0.84846<br/>
  b. 0.87541<br/>
  c. 0.10714<br/>
  d. 0.42236<br/>

  Ans: c<br/>
  
  ```python
  def largest_face(f, x_max):
    # inputs: f is a list of integers and x_max is an integer
    # output: a variable of type 'float'

    if x_max > max(f) or x_max <= 0:
        return 0
    P1 = 1
    P2 = 1
    for face in f:
        P1 *= (min(1, x_max / face))
        P2 *= (min(1, (x_max - 1) / face))

    return P1 - P2
  ```


2. Which of the following is the correct output for: `face_sum([2,4,4,6], 10)`<br/>
  a. 0.15625<br/>
  b. 0.58962<br/>
  c. 0.04515<br/>
  d. 0.52158<br/>

  Ans: a<br/>

  ```python
  def face_sum(m, s):
    # inputs: m is list of integers and s is an integer
    # output: a variable of type 'float'
    
    num_all = 1;
    for num_m in m :
        num_all *= num_m
    
    return len(constrained_compositions(s, m)) / num_all
  ```






