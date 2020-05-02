# Topic 1: Introduction to Probability and Statistics

## 1.1 Introduction to Probability and Statistics

### Lecture Notes

+ Why should you care about prob&stat?
  + a powerful tool to deal w/ uncertainty
  + example: Navigation software
    + Certainty: find the _shortest_ route from a to b
    + Uncertainty: find the _fastest_ rout from a to b
  + example: Search engine
    + Certainty: find _all_ web pages that contain the word "Trump", "Hillary"
    + Uncertainty: find the 10 _most relevant_ pages for the query "Trump, Hillary debate"
      + not about certain words appeared
      + about what pages really most relevant up to date
  + example: Insurance company
    + Certainty: if a person w/ life insurance dies, the insurance company has to pay the family $X
    + Uncertainty: what is the minimal life insurance premium such that _the probability_ that the life insurance company will be bankrupt in 10 years is smaller than 1%?

+ What will learn in the course?
  + Navigation and search engine problems are advanced, in this class you will learn the foundations
  + Solve basic problems of reasoning under uncertainty
  + Examples
    + if you flip a coin 100 times, what is the probability of getting at most 10 "heads"?
    + what is the probability of getting a "4 of a kind" hand in poker?

+ Computer science examples
  + If you want to hash 1,000,000 elements and can allow 5 indirections for only 10 elements, how big does the table need to be?
  + Suppose that the expected time between failures for a router in one year.  What is the probability that the router will fail during the first month?

+ Some don't belief in statistics
  
  > I don't believe in statistics.  There are too many factors that can't be measured.  You can't measure a ballplayer's heart. - Red Auerbach, basketball coach

+ Summary
  + Uncertainty is all around us
  + Probability and Statistics provide a rational way to deal w/ uncertainty


+ [Original Slides](https://tinyurl.com/y75qx6vl)


### Problem Set

+ Probability and statistics help us understand, analyze, and utilize random phenomena.

1. Which of the following are best solved using probability and statistics?<br/>
  a. Predicting the number of rainy days in April.<br/>
  b. Approximating the closing price of IBM stock tomorrow.<br/>
  c. Estimating your potential winnings in a game of Blackjack.<br/>
  d. Guessing the winner of the next World Cup.<br/>

  Ans: abcd<br/>
  Explanation: All these events are uncertain, and can be addressed by various aspects of probability and statistics, some of which we will encounter in this course.

2. What are probability and statistics useful for?<br/>
  a. Quantifying uncertainty.<br/>
  a. Finding exact solutions to mathematical equations.<br/>
  c. Making predictions about the future.<br/>

  Ans: ac<br/>
  Explanation
  + True. Just note that some random phenomena can be better quantified than others.
  + False. There is no uncertainty here.
  + True. Of course, the accuracy may depend on what we predict, and how far into the future.


### Lecture Video

<a href="https://tinyurl.com/y7nx6ec2" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 1.2 What is Probability Theory

### Lecture Notes

+ What is Probability Theory?
  + Probability theory: a __mathematical__ framework for computing the probability of complex events
  + Assumption: __we know the probability of the basic events.__
  + the precise meaning of "probability" and "event"?
  + giving precise definitions later in the class
  + relying on common sense at first

+ A simple (?) question
  + flipping a fair coin w/ equal probabilities of "heads" or "tails"
  + what does the statement mean?
    + flipping the coin $k$ times, $\exists\, k \gg 1$, say $k = 10,000$
  + the number of "heads" is __about__ $\frac{k}{2} = \frac{10,000}{2} = 5,000$
  + what does __about__ mean?

+ Simulating coin flips
  + using the pseudo random number generators in `numpy` to simulate the coin flips
  + instead of "Heads" and "tails", using $x_i = 0$ or $x_i = -1$
  + considering the sum $S_{10000} = x_1 + x_2 + \cdots + x_{10000}$
  + the number of heads is about 5,000 $\implies S_{10000} \approx 0$
  + varying the number of coin flips, dented by $k$
  + demo: generate $ n \times k$ coin flips w/ values $\pm 1$

    ```python
    import numpy as np
    # Generate the sum of k coin flips, repeat that n times
    def generate_counts(k=10000, n=100):
      X = 2*(np.random.rand(k, n) > 0.5) -1 # generate a kxn matrix of +-1 random numbers
      S = sum(X, axis=0)
      return S
    ```

  + the sum $S_{10000}$ is not __exactly__ 0, it is only __close to__ 0
  + using __probability theory__, calculate __how small__ is $|S_k|$
  + later, its wll be shown that the probability that $|S_k| \geq 4\sqrt{k}$ is smaller than $2 \times 10^{-8} = 0.000002\%$

+ Demo: simulation

  ```python
  from math import sqrt  
  figure(figsize=[13, 3.5])
  for j in range(2, 5):
    k = 10*8j
    counts = generate_counts(k=k, n=100)
    subplot(130 + j - 1)
    hist(counts, bin=10);
    d = 4*sqrt(k)
    plot([-d, -d], [0, 30], 'r')
    plot([+d, +d], [0, 30], 'r')
    grid()
    title('{0:d} flips, bound=+-{1:6.1f}'.format(k, d))

  figure(figsize=[13, 3.5])
  for j in range(2, 5):
    k = 10*8j
    counts = generate_counts(k=k, n=100)
    subplot(130 + j - 1)
    hist(counts, bin=10);
    xlim([-k, k])
    d = 4*sqrt(k)
    plot([-d, -d], [0, 30], 'r')
    plot([+d, +d], [0, 30], 'r')
    grid()
    title('{0:d} flips, bound=+-{1:6.1f}'.format(k, d))
  ```

+ Summary
  + executing some experiments summing $k$ random numbers: $S_k = x_1, x_2 + \cdots + x_k$
  + the probability of the value of $x_i, \; i = 1, 2, \dots, k$

    \[ p(x_i) = \begin{cases} 1/2 & \text{for } x_i = -1 \\ 1/2 & \text{for } x_i = +1 \end{cases} \]

  + experiments show that the sum $S_k$ is (almost) always in the range $[-4\sqrt{k}, 4\sqrt{k}]$

    \[\begin{align*}
      k \to \infty &\text{ s.t. }\frac{4\sqrt{k}}{k} = \frac{4}{\sqrt{k}} \to 0 \\
      \therefore\; k \to \infty &\text{ s.t. } \frac{S_k}{k} \to 0
    \end{align*}\]

+ Math interpretation
  + math involved in __proving__ (a precise version of) the statements above
  + in most cases, __approximating__ probabilities using simulations (Monte-Carlo simulations)
  + calculating the probabilities is better because
    + providing a precise answer
    + much faster than Monte-Carlo simulations


+ [Original Slides](https://tinyurl.com/ya5gx8z7)



### Lecture Video

<a href="https://tinyurl.com/ydz88by5" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 1.3 What is Statistics

### Lecture Notes

+ What is statistics?
  + probability theory: computing probabilities of complex events given the underlying base distribution
  + statistics: 
    + the opposite direction of probability
    + given __data__ generated by a __stochastic process__
    + __inferring__ properties of the stochastic process

+ Example: deciding whether a coin is based
  + probability: discussing the distribution of the number of heads when flipping a fair coin many times
  + statistics:
    + flipping a coin 1000 times and get 570 heads
    + able to conclude that the coin is biased (not fair)?
    + what to conclude if getting 507 heads?

+ The logic of Statistical Inference
  + suppose that the coin is fair
  + using __probability theory__ to compute the probabilities of getting at least 570 (507) heads
  + probability is very small $\implies$ __reject__ <span style="color: red;">with confidence</span> the hypothesis that the coin is fair

+ Calculating the answer
  + modeling the coin flip

    \[ x_i = \begin{cases} -1 & \text{for tail} \\ +1 &\text{for head} \end{cases} \]

  + observing the sum $S_k = \sum_{i=1}^k x_i$. here $k = 1000$
  + 570 heads out of 1000 flips
    + 570 heads: $S_{1000} = 570 -430 = 140$ 
    + the value 140 is very unlikely that $|S_k| > 4\sqrt{k} \approx 126.5$

    $\therefore$ it is very unlikely that the coin is unbiased.
  + 507 heads out of 1000 flips

    \[ 507 \text{ heads } = 493 \text{ tails }\implies S_{1000} = 14, \quad 14 \ll 126.5 \]

    $\therefore$ we cannot conclude that the coin is biased.
  + Conclusion
    + The probability that an unbiased coin would generate a sequence w/ 570 or more heads is extremely small $\implies$ able to conclude, <span style="color: red;">with high confidence</span>, that the coin is biased
    + on the other hand, $|S_{1000}| \geq 14$ is quite likely $\implies$ getting 507 heads does not provide evidence that the coin is biased

+ Real-world examples
  + why should we care whether a coin is fair?
    + this is a valid critique
    + examples where knowing a coin is biased or not is required
  + Case I: Polls
    + suppose elections will take place in a few days and we want to know how people plan to vote
    + suppose there are to parties: D and R
    + take a survey for __all__ voters $\to$ expensive
    + using a poll instead: call up a small random selected set of people
    + call $n$ people at random and count the number of D votes
    + can you say <span style="color: red">with confidence</span> that there are more D votes, or more R votes?
    + mathematically equivalent to flipping a biased coin
    + asking whether you can say <span style="color: red">with confidence</span> that is is biased towards 'heads' or towards 'tails'
  + Case 2: A/B testing
    + a common practice when optimizing a web page to perform A/B tests
    + A/B referred to two alternative designs for the page
    + to see which design users prefer by randomly presenting design A or design B
    + measuring how long the user stayed on a page, or whether the user clicked on a advertisement
    + to decide, <span style="color: red">with confidence</span>, which of the two designs is better
    + similar to making decision <span style="color: red">with confidence</span> on whether 'heads' is more probably than 'tails' or vice versa

+ Summary
  + statistics is about analyzing real-world data drawing conclusion
  + examples including
    + using polls to estimate public opinion
    + performing A/B tests to design web pages
    + estimating the rate of global warming
    + deciding whether a medical procedure effective


+ [Origin Slides](https://tinyurl.com/y8logskb)


### Problem Set

+ If we flip a coin a thousand times and get 507 heads, can we conclude with certainty that the coin is unbiased?

  Ans: No<br/>
  Explanation: Regardless of the bias of the coin (except when it is always heads or always tails), we can get 507 heads, hence we cannot deduce the bias with certainty. We will see how likely this outcome is later on.

1. In rolling a fair  6-sided die  1,200  times, roughly how many times would you expect to see a 2 ?<br/>
  a.  200 <br/>
  a.  600 <br/>
  a.  1,000 <br/>
  a. Not enough information is given<br/>

  Ans: a<br/>
  Explanation: The number 2 will appear in roughly one sixth of the flips, namely $1200 \times 1/6=200$ times.


2. A coin is tossed  1000  times and turns up heads  700  times. Is the coin biased?<br/>
  a. With high confidence, yes.<br/>
  b. Unclear.<br/>

  Ans: a<br/>
  Explanation: The probability that an unbiased coin would generate  700  heads is small. Hence we can be pretty confident that it is biased. How confident, we will see later on.


3. Which of the following describe the differences between probability and statistics?<br/>
  a. Probability predicts what will happen. Statistics, at least in part, uses what has already happened.<br/>
  b. Probability requires existing data. Statistics requires underlying models.<br/>
  a. Probability and statistics are two words describing the same thing.<br/>

  Ans: a<br/>
  Explanation
  + True. Probability analyzes given models. Statistics helps us understand the what we observe.
  + False. Probability requires underlying models. Statistics requires existing data.
  + False. As you can see from the previous two parts, they are different.



### Lecture Video

<a href="https://tinyurl.com/y9uvlvfn" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 1.4 A Puzzle

### Lecture Notes




### Problem Set




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 1.5 History of Probability and Statistics

### Lecture Notes




### Problem Set




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 1](./01-Intro.md#)




## Programming Assignment 1




