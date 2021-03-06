# Topic10: Inequalities and Limit Theorems


## 10.1 Markov Inequality


0. A mob of 30 meerkats has an average height of 10”, and 10 of them are 30” tall. According to Markov's Inequality this is:<br/>
  a. Possible<br/>
  b. Impossible<br/>

  Ans: 


1. Which of the following are correct versions of Markov's Inequality for a nonnegative random variable  X :<br/>
  a. $P(X \ge \alpha \mu) \le \frac{1}{\alpha}$<br/>
  b. $P(X \ge \alpha \mu ) \le \mu \alpha$<br/>
  c. $P(X \ge \mu) \le \frac{1}{\alpha}$<br/>
  d. $P(X \ge \alpha) \le \frac{\mu}{\alpha}$<br/>

  Ans: 


2. Markov variations

  Upper bound $P(X \ge 3)$ when $X \ge 2$ and $E[X]=2.5$.

  Ans: 


3. a. In a town of 30 families, the average annual family income is `$80,000`. What is the largest number of families that can have income at least `$100,000` according to Markov’s Inequality? (Note: The annual family income can be any non-negative number.)

  b. In the same town of 30 families, the average household size is 2.5. What is the largest number of families that can have at least 4 members according to Markov’s Inequality? (Note the household size can be any positive integer.)

  Ans: 


## 10.2 Chebyshev Inequalities

0. Which of the following is correct about Chebyshev's inequality?<br/>
  a. It only applies to non-negative distribution<br/>
  b. It only applies to discrete distribution<br/>
  c. It only applies to continuous distribution<br/>
  d. None of the above<br/>

  Ans: 


1. Apply Chebyshev's Inequality to lower bound $P(0 < X < 4)$ when $E[X]=2$ and $E[X^2]=5$.

  Ans: 


2. The average number of spelling errors on a page is  5  and the standard deviation is  2 . What is the probability of more than  20  mistakes on a page?<br/>
  a. no greater than  1% <br/>
  b. no greater than  2% <br/>
  c. no greater than  5% <br/>
  d. no greater than  10%<br/>

  Ans: 


3. Let $X\sim \text{Exponential}(1)$. For $P(X \ge 4)$, evaluate:<br/>
  a. Markov's inequality,<br/>
  b. Chebyshev's inequality,<br/>
  c. the exact value.<br/>

  Ans: 

4. A gardener has new tomato plants sprouting up in her garden. Their expected height is 8”, with standard deviation of 1". Which of the following lower bounds the probability that a plant will be between 6" and 10" tall?<br/>
  a. 10%<br/>
  b. 25%<br/>
  c. 50%<br/>
  d. 75%<br/>
  
  Ans: 


5. If $E[X]=15$, $P(X \le 11)=0.2$, and $P(X \ge 19)=0.3$, which of the following is _impossible_?<br/>
  a. $Var(X) \le 7$<br/>
  b. $Var(X) \le 8$<br/>
  c. $Var(X) > 8$<br/>
  d. $Var(X) > 7$<br/>

  Ans: 


## 10.3 Law of Large Numbers

0. You have two fair coins, and you toss the pair 10,000 times (so you get 10,000 outcome pairs). Roughly how many pairs will not show any tails?<br/>
  a. 0<br/>
  b. 1250<br/>
  c. 2500<br/>
  d. 5000<br/>

  Ans: 


1. In plain terms, the Weak Law of Large Numbers states that as the number of experiments approaches infinity, the difference between the sample mean and the distribution mean can be as small as possible. (True/False)

  Ans: 


2. Given n iid random varibles $X_1,X_2, \dots ,X_n$ with mean $\mu$, standard deviation $\alpha < \infty$ , and the sample mean $S_n = \frac{1}{n} \sum_{i=1}^n X_i$, is it true that $\lim_{n\to\infty} E[(S_n−\mu)^2]=0$? (True/False)

  Ans: 


3. The height of a person is a random variable with variance $\le 5 \text{ inches }^2$. According to Mr. Chebyshev, how many people do we need to sample to ensure that the sample mean is at most 1 inch away from the distribution mean with probability $\ge 95\%$?

  Ans: 


4. For $i=1,2,\dots ,n$, let $X_i \sim U(0,4)$, $Y_i \sim N(2,4)$, and they are independent. Calculate,<br/>
  a. $E(X_i)$<br/>
  b. $V(X_i)$<br/>
  c. $E(Y_i)$<br/>
  d. $V(Y_i)$<br/>

  Find the limit in probability of when $n \to \infty$<br/>
  e. $\frac 1 n \sum_{i=1}^n (X_i+Y_i)$<br/>
  f. $\frac 1 n \sum_{i=1}^n (X_iY_i)$<br/>

  Ans: 


5. a. Flip a fair coin $n$ times and let $X_n$ be the number of heads. Is it true that $P(|X_n − \frac n 2|>1000)<0.99$? (True/False)

   b. Does the result above contradict with the WLLW? (Yes/No)

  Ans: 


## 10.4 Moment Generating Functions

0. If $M(t)$ is a moment generating function, then what is $M(0)$?<br/>
  a. 0<br/>
  b. 1<br/>
  c. infinity<br/>
  d. depends on the distribution

  Ans: 


1. If $X$ has moment generating function $M_X(t)=(1−3t)^{−1}$, what is $Var(X)$?<br/>
  a. 6<br/>
  b. 9<br/>
  c. 12<br/>

  Ans: 
  

2. Let $M_X(t)$ be the MGF of $X$. Which of the following hold for all $X$ and $Y$?<br/>
  a. $M_{X}(0)=1$<br/>
  b. $M_{X}(t) \ge 0$ for all $t$<br/>
  c. $M_{3X+2}(t)=e^{2t}⋅M_X(3t)$<br/>
  d. $M_{X+Y}(t)=M_X(t)M_Y(t)$<br/>

  Ans: 
  


3. If $X$ is a non-negative continuous random variable with moment generating function

  \[ M_X(t)=\frac{1}{(1-2t)^2},\quad t<\frac{1}{2} \]

  a. $E[X]$<br/>
  b. $Var(X)$<br/>

  Ans: 
  


4. Let $X_1, X_2, \dots$  be independent $B_{1/2}$ random variables, and let $M \sim P_4$, namely Poisson with mean $4$. Which of the following is the MGF of $X_1+X_2+ \cdots +X_M$?<br/>
  a. $e^{2(1+e^t)}e^{−4}$<br/>
  b. $e^{1+e^t}e^{−2}$<br/>
  c. $\frac{1+e^t}{2}$<br/>
  d. $\frac{1+e^2t}{2}$<br/>

  Ans: 
  


5. Let $X$ be a random variable with MGF $M_X(t)=\frac13 e^{−t}+\frac16+\frac12 e^{2t}$. What is $P(X \le 1)$?

  Ans: 
  


6. Let $M_X(t)$ be an MGF, which of the following are valid MGF's?<br/>
  a. $M_X(2t)M_X(7t)$<br/>
  b. $e^{−5t}M_X(t)$<br/>
  c. $3M_X(t)$<br/>

  Ans: 
  


7. If $M_X(t)=e^{−5(1−e^t)}$,<br/>
  a. find $Var(X)$.<br/>
  b. $P(X = 3)$<br/>

  Ans: 
  


8. Find the MGF of $(X_1+X_2+X_3+X_4)/3$ where each $X_i$ is an independent $B_{1/2}$ random variable?<br/>
  a. $((1+e^{t/3})/2)^4$<br/>
  b. $((1+e^t)/2)^4$<br/>
  c. $((2/3+e^t/3))^4$<br/>
  d. $((2/3+e^t/3/3))^4$<br/>

  Ans: 


## 10.5 Chernoff Bound

0. If we want to apply Chernoff bound to other distributions, the formulas are going to be different from Chernoff bound on binomial distributions. Because different distributions have the different moment generating functions. (True/False)

  Ans: 


1. You toss a fair coin $1000$ times and take a step forward if the coin lands head and a step backward if it lands tail. Upper bound the probability that you end up $\ge 100$ steps from your starting point (in either direction) using Chernoff bound (after the final simplification as in the slides).

  Ans: 



2. A coin is equally likely to be either $B_{1/3}$ or $B_{2/3}$. To figure out the bias, we toss the coin $99$ times and declare $B_{1/3}$ if the number of heads is less than $49.5$ and $B_{2/3}$ otherwise. Bound the error probability using the Chernoff bound derived in lecture video (in its final form, after simplifcation).

  Ans: 


## 10.6 Central Limit Theorem

0. Let X be a random variable with $\mu = 10$ and $\sigma = 4$. If X is sampled 100 times, what is the approximate probability that the sample mean of these 100 observations is less than 9?<br/>
  a. 0.002<br/>
  b. 0.004<br/>
  c. 0.006<br/>
  d. None of the above<br/>

  Ans: 


1. For $i \ge 1$, let $X_i \sim G_{1/2}$ be distributed Geometrically with parameter $1/2$.

  Define $Y_n=\frac{1}{\sqrt{n}}\sum_{i=1}^n (X_i-2)$

  Approximate $P(−1 \le Y_n \le 2)$  with large enough $n$.

  Ans: 


2. A class has 100 students. Each student's score is a random variable with mean $85$ and standard deviation $40$. Use the CLT to approximate the probability that the class average score is below $80$.

  Ans: 


3. The time between consecutive shuttle arrivals is known to be exponentially distributed with mean 10 minutes. You arrive at the shuttle stop at a uniformly-distributed time.

  a. What is the probability that you wait for less than 9 minutes?  
  
  b. Assume that you took the shuttle once a day during the past 30 days. What is the approximate probability, according to the CLT, that your average wait time was less than 9 minutes?

  Ans: 


## 10.7 Central Limit Theorem Proof

0. Suppose that X, Y, and $(X+Y)/\sqrt{2}$ all share the same probability density function f. What could f be?<br/>
  a. Uniform over [0,1]<br/>
  b. Exponential with parameter 2<br/>
  c. Normal with mean 0<br/>
  d. Normal with mean 1<br/>

  Ans: 



# Topic11: Statistics, Parameter Estimation and Confidence Interval
  

## 11.1 Statistics

0. Recall a statistic is a single value calculated from the sample. Which of the following is a statistic?<br/>
  a. sample max<br/>
  b. sample mean<br/>
  c. sample median<br/>
  d. all of the above<br/>

  Ans: 


1. $225$ iPhones go on sale on black Friday, and 100 customers are in line to buy them. If the random number of iPhones that each customer wishes to buy is distributed Poisson with mean 2, approximate the probability that all 100 customers get their desired number of iPhones?

  Ans: 
  


2. The number of years a Bulldog lives is a random variable with mean 9 and standard deviation 3, while for Chihuahuas, the mean is 15 and the standard deviation is 4. Approximate the probability the that in a kennel of 100 Bulldogs and 100 Chihuahuas, the average Chihuahua lives at least 7 years longer than the average Bulldog.

  Ans: 



## 11.2 Mean and Variance Estimation

0. A distribution has mean 5 and variance 10. If we collect a sample by making 20 independent observations, what is the variance of the sample mean?<br/>
  a. 2<br/>
  b. 1/2<br/>
  c. 1/4<br/>
  d. 1/40<br/>

  Ans: 
  

1. If an estimator is unbiased, then<br/>
  a. its value is always the value of the parameter,<br/>
  b. its expected value is always the value of the parameter,<br/>
  c. it variance is the same as the variance of the parameter.<br/>

  Ans: 
  


2. If $\{X_1, \dots, X_n\}$ are the observed values of $n$ sample items, which of the following are unbiased estimators for distribution mean?<br/>
  a. $X_1$<br/>
  b. $\frac{1}{n}\sum_{i=1}^n X_i$<br/>
  c. $\sqrt{\frac{1}{n}\sum_{i=1}^n X_i^2}$<br/>

  Ans: 
  


3. As the sample size $n$ grows, the sample mean estimates the distribution mean better. Because<br/>
  a. its bias decreases,<br/>
  b. its variance decreases,<br/>
  c. none of the above.<br/>

  Ans: 


4. A sample of size $n$ has sample mean $20.20$. After adding a new observed value $21$, the sample mean increases to $20.25$. What is $n$?

  Ans: 



5. To estimate the average alcohol consumption of UCSD students, we take three random samples of 40, 45 and 50 students respectively, and their sample means turn out to be 3.15, 3.20 and 2.76 pints per week respectively. What is the sample mean of the collection of all three samples?

  Ans: 
  


6. Let $X_1,X_2, \dots,X_n$ be independent samples from a distribution with pdf $f_X(x)=\frac{1}{\theta^2}xe^{-\frac{x}{\theta}} (X\ge 0)$. Which of the following is an unbiased estimator for $θ$?<br/>
  a. $\overline{X}$<br/>
  b. $\frac{\overline{X}}{2}$<br/>
  c. $\frac{\overline{X}}{3}$<br/>
  d. $\frac{\overline{X}}{6}$<br/>

  Ans: 
  


7. For $i \in \{1,\dots,n\}$, let $X_i \sim U(0,W)$ independently of each other, and let $M_n = \max_{i \in \{1, \dots,n\}}X_i$. For what value of $c$ is $c \cdot M_n$ an unbiased mean estimator?<br/>
  a. $\frac{n+1}{2n}$<br/>
  b. $\frac{n}{2(n−1)}$<br/>
  c. $\frac{2n+1}{4n}$<br/>
  d. $\frac{2n}{4n−1}$<br/>

  Ans: 


8. Let $X$ be distributed $Poisson(λ)$. Which of the following is an unbiased estimator for $λ^2$.<br/>
  a. $X^2$<br/>
  b. $X^2−X$<br/>
  c. $2X^2−X$<br/>
  d. $3X^2−2X$<br/>

  Ans: 



## 11.3 Variance Estimation

0. As an estimator for distribution variance, the "raw" sample variance is<br/>
  a. biased<br/>
  b. unbiased<br/>

  Ans: 


1. Let $\overline{X}_n$ and $S_n^2$ be the sample mean and the sample variance of $\{X_1, \dots, X_n\}$. Let $\overline{X}_{n+1}$ and $S^2_{n+1}$ be the sample mean and the sample variance of $\{X_1, \dots, X_n, \overline{X}_n\}$. Which of the following hold

  1) for sample means,<br/>
  a. $\overline{X}_n > \overline{X}_{n+1}$<br/>
  b. $\overline{X}_n < \overline{X}_{n+1}$<br/>
  c. $\overline{X}_n = \overline{X}_{n+1}$<br/>

  2) for sample variances?<br/>
  a. $S^2_n > S^2_{n+1}$<br/>
  b. $S^2_n < S^2_{n+1}$<br/>
  c. $S^2_n = S^2_{n+1}$<br/>

  Ans: 


2. Consider the following array of $m \times n$ random variables $X_{11}, X_{12}, \cdots, X_{1n}, \cdots,$ $X_{i1}, X_{i2}, \cdots, X_{in}, \cdots,$ $X_{m1}, X_{m2}, \cdots, X_{mn}$. For $i = 1, \cdots, m$, let $\overline{X}_i$ be the sample mean of $\{X_{i1}, X_{i2}, \cdots, X_{in}\}$, and $\overline{S}^2$ be the "raw" sample variance of $\{\overline{X}_1, \overline{X}_2, \cdots,\overline{X}_m\}$. If $\forall i, j, Var(X_{ij}) = \sigma^2$, what is $E[\overline{S}^2]$?<br/>
  a. $\frac{n−1}{n} \sigma^2$<br/>
  b. $\frac{m−1}{m} \sigma^2$<br/>
  c. $\frac{1}{n} \sigma^2$<br/>
  d. $\frac{1}{m} \sigma^2$<br/>
  e. $\frac{n−1}{mn} \sigma^2$<br/>
  f. $\frac{m−1}{mn} \sigma^2$<br/>

  Ans: 
  


3. If all the observations in a sample increase by 5<br/>
  a. the sample mean increases by 5,<br/>
  b. the sample mean stays the same,<br/>
  c. the sample variance increases by 5,<br/>
  d. the sample variance stays the same.<br/>

  Ans: 


## 11.4 Unbiased Variance Estimation

0. Compared to the distribution variance, the expectation of the biased "raw" sample variance is<br/>
  a. always larger<br/>
  b. always smaller<br/>
  c. always equal<br/>
  d. could be any of the above<br/>

  Ans: 


1. As the sample size $n$ grows, the effect of the Bessel correction<br/>
  a. becomes larger,<br/>
  b. becomes smaller,<br/>
  c. stays the same<br/>

  Ans: 
  


2. According to the U.S. Department of Agriculture, ten to twenty earthworms per cubic foot is a sign of healthy soil. The soil of a garden is checked by digging 8 holes, each of one-cubic-foot, and counting the earthworms, and the following counts are found: 5, 25, 15, 10, 7, 12, 16, 20. Use the unbiased estimators discussed in the video to estimate<br/>
  a. the true mean,<br/>
  b. the true variance.<  br/>

  Ans: 


## 11.5 Estimating Standard Deviation

0. There is an unbiased estimator for standard deviation for general distributions. (True/False)

  Ans: 



## 11.6 Confidence Interval

0. The margin of error of confidence interval with 100% confidence level will be<br/>
  a. Zero<br/>
  b. One standard deviation<br/>
  c. Infinity<br/>
  d. None of the above<br/>

  Ans: 


1. Which of the following will increase the length of the confidence interval?<br/>
  a. Increase confidence level<br/>
  b. Decrease confidence level<br/>
  c. Increase sample size<br/>
  d. Decrease sample size<br/>

  Ans: 
  


2. A psychologist estimates the standard deviation of a driver's reaction time to be 0.05 seconds. How large a sample of measurements must be taken to derive a confidence interval for the mean with margin of error at most 0.01 second, and confidence level 95%?

  Ans: 
  


3. A sample of size $n=25$ with the population standard deviation $\sigma=3$, compute the margin of error of a 90% confidence interval for the mean $\mu$.

  Ans: 


## 11.7 Confidence Interval - Sigma Unknown

0. A confidence interval of mean has confidence level 95%. It means<br/><br/>
  a. The confidence interval includes distribution mean with probability 95%<br/>
  b. 95% of the observations in the sample fall into this interval<br/>
  c. If we take a new sample point, it falls into this interval 95% of the time<br/>
  d. Non of the above<br/>

  Ans: 
  

1. Student's t-distribution can be used to form confidence intervals only when the samples are normal distributed. (True/False)

  Ans: 


2. To find the average SAT verbal score in a class, six students are sampled and their scores are 560, 610, 500, 470, 660, and 640. Assuming that students' SAT verbal scores follow normal distribution, what is the upper limit for the confidence interval of the distribution mean with confidence level 90%?

  Ans: 


3. a. What is the critical $t$ for a 92% confidence interval with a sample size 10?<br/>
   b. What is the critical $t$ for a 92% confidence interval with a sample size 1000?<br/>
   c. What is the critical $z$ for a 92% confidence interval?<br/>

  Ans: 



# Topic 12: Regression and PCA


## 12.1 Review of Linear Algebra

0. Which is NOT true of an orthonormal basis?	<br/>
  a. All of the vectors in the set are orthogonal to each other. The norm of each vector is 1.<br/>
  b. The standard basis in $\Bbb{R}^3$, $e_1=(1,0,0), e_2=(0,1,0), e_3=(0,0,1)$, is orthonormal.<br/>
  c. A vector in the set cannot be a scalar multiple of another vector in the set.<br/>
  d. An orthonormal basis can contain infinitely many vectors for any vector space.<br/>

  Ans: 


1. What is the length of $\vec{u}$ such that $\vec{u} = \frac{\vec{v}}{\parallel \vec{v}\parallel}$, $\vec{v} =(2,3,7)$?<br/>
  a. 1 <br/>
  b. 3.61 <br/>
  c. 7.84 <br/>
  d. 62<br/>

  Ans: 


2. If every vector in an orthonormal basis is orthogonal to each other, this implies that there can be one and only one vector for each dimension of the vector space in this set. (True/False)

  Ans: 


3. An inner product, such as the dot product, always uses two vectors as operands and produces a scalar number as the result. (True/False)

  Ans: 


4. If vectors $\vec{a}$ and $\vec{b}$ are orthogonal, then what is the value of $\vec{a} \cdot \vec{b}$?<br/>
  a. 0 <br/>
  b. 1 <br/>
  c. 2 <br/>
  d. 90<br/>

  Ans: 


## 12.2 Matrix Notation and Operations

0. Select the correct statement about matrices from the following:<br/>
  a. A matrix cannot be divided by a scalar, and a scalar cannot be divided by a matrix<br/>
  b. A matrix can be divided by a scalar, but a scalar cannot be divided by a matrix<br/>
  c. A matrix cannot be divided by a scalar, but a scalar can be divided by a matrix<br/>
  d. A matrix can be divided by a scalar, and a scalar can be divided by a matrix<br/>

  Ans: 


1. A $m \times n$ matrix can be added with a $n \times m$ matrix, but they cannot be multiplied. (Assume $m \ne n$) (True/False)

  Ans: 


2. Let $\vec{a} =(1,0,0)$, $\vec{b} =(0,1,0)$, and $\vec{c} =(0,0,1)$. This is the standard basis that spans $\Bbb{R}^3$. Answer the following questions about this set of vectors:

  1) $\vec{a} +\vec{b} =?$<br/>
    <span style="padding-left: 1em">a.</span> (1,1)<br/>
    <span style="padding-left: 1em">b.</span> (0,0,1)<br/>
    <span style="padding-left: 1em">c.</span> (1,1,1)<br/>
    <span style="padding-left: 1em">d.</span> (1,1,0)<br/>

  2) $\vec{a} \cdot \vec{b} =?$<br/>
    <span style="padding-left: 1em">a.</span> (0,0,0)<br/>
    <span style="padding-left: 1em">b.</span> 0<br/>
    <span style="padding-left: 1em">c.</span> (1,1,0)<br/>
    <span style="padding-left: 1em">d.</span> 2<br/>

  3) $(\vec{a} \cdot \vec{b} )\vec{c}$ =?<br/>
    <span style="padding-left: 1em">a.</span> (0,0,0)<br/>
    <span style="padding-left: 1em">b.</span> 0<br/>
    <span style="padding-left: 1em">c.</span> 1<br/>
    <span style="padding-left: 1em">d.</span> (0,0,1)<br/>

  4) $−\vec{c} =?$<br/>
    <span style="padding-left: 1em">a.</span> (0,0,1)<br/>
    <span style="padding-left: 1em">b.</span> (0,0,−1)<br/>
    <span style="padding-left: 1em">c.</span> (1,0,0)<br/>
    <span style="padding-left: 1em">d.</span> Vectors cannot be negative<br/>

  5) $\parallel \vec{a} \parallel = \sqrt{\vec{a} \cdot \vec{a}}$<br/>
    <span style="padding-left: 1em">a.</span> True<br/>
    <span style="padding-left: 1em">b.</span> False<br/>
    <span style="padding-left: 1em">c.</span> This notation is meaningless<br/>

  6) $\parallel \vec{a} \parallel +\parallel \vec{b} \parallel = ?$<br/>
    <span style="padding-left: 1em">a.</span> $\parallel \vec{c} \parallel$<br/>
    <span style="padding-left: 1em">b.</span> 1 <br/>
    <span style="padding-left: 1em">c.</span> 2 <br/>
    <span style="padding-left: 1em">d.</span> (1,1,0) <br/>

  Ans: 


3. Given a matrix, $A = \begin{bmatrix} 4 \ 1 \\ 1 \ 9 \end{bmatrix}$, find $(4A)^{-1}$.<br/>
  a. $(4A)^{-1} = \begin{bmatrix} 1 \ \, {- \frac{1}{9}} \\ \!\!\!\! \, {- \frac{1}{9}} \ \hspace{.3cm} \frac{4}{9} \end{bmatrix}$<br/>
  b. $(4A)^{-1} = \begin{bmatrix} \frac{9}{140} \ \ {\,-\frac{1}{140}} \\ \!\!\!\!\!\!\! \,- \frac{1}{140} \ \hspace{.4cm} \frac{1}{35} \end{bmatrix}$<br/>
  c. $(4A)^{-1} = \begin{bmatrix} \frac{1}{16} \ \hspace{.5cm} \! \frac{1}{4} \\  \frac{1}{4} \ \hspace{.4cm} \frac{1}{36} \end{bmatrix}$<br/>
  d. $(4A)^{-1} = \begin{bmatrix} \frac{1}{36} \ \hspace{.1cm} {-\frac{1}{4}} \\ \!\!\!\!  \, {-\frac{1}{4}} \ \hspace{.4cm} \frac{1}{16} \end{bmatrix}$<br/>

  Ans: 


4. Given the matrix $A$ below, answer the following questions:

  \[ A = \begin{bmatrix} a_{11} \ a_{12} \\ a_{21} \ a_{22} \end{bmatrix} \]

  1) $4A+4A=$?<br/>
    <span style="padding-left: 1em">a.</span> $4A$<br/>
    <span style="padding-left: 1em">b.</span> $8A$<br/>
    <span style="padding-left: 1em">c.</span> $16A$<br/>
    <span style="padding-left: 1em">d.</span> Cannot add two matrices of the same dimension<br/>

  2) $A-2 = \begin{bmatrix} {a_{11}\!-2} \ {a_{12}\!-2} \\ {a_{21}\!-2} \ {a_{22}\!-2} \end{bmatrix}$ (True/False)
  
  3) $A^{−1}= \frac{1}{A}$ (True/False)
  
  4) $(A^\top)I=$?<br/>
    <span style="padding-left: 1em">a.</span> $A$ <br/>
    <span style="padding-left: 1em">b.</span> $A^\top$ <br/>
    <span style="padding-left: 1em">c.</span> $1$ <br/>
    <span style="padding-left: 1em">d.</span> $A^{−1}$ <br/>

  5) $(A^\top)^\top=A$ (True/False)

  Ans: 


5. Recall, from linear algebra, that the determinate of a matrix, $A = \begin{bmatrix} a_{11} \ a_{12} \\ a_{21} \ a_{22} \end{bmatrix}$, is equal to $(a_{11}a_{22}-a_{12}a_{21})$. If this determinant is equal to $0$, what does that indicate about the matrix, $A$?<br/>
  a. The difference of the norms of the column vectors is $0$<br/>
  b. The matrix $A$ has no transpose<br/>
  c. The matrix $A$ has no inverse<br/>
  d. This is an identity matrix<br/>

  Ans: 
  


6. If $A = \begin{bmatrix} a_{11} \ a_{12} \\ a_{21} \ a_{22} \end{bmatrix}$ and $B = \begin{bmatrix} b_{11} \ b_{12} \\ b_{21} \ b_{22} \end{bmatrix}$, then $BA=$?<br/>
  a. $C$<br/>
  b. $C^{−1}$<br/>
  c. $C^\top$<br/>
  d. None of the above<br/>

  Ans: 


7. Only square matrices have inverses. (True/False)

  Ans: 


## 12.3 Solving a System of Linear Equations

0. In the matrix equation, ${\bf A \vec{w}=\vec{b}}$,what does the matrix, A, contain?<br/>
  a. The x-values of two points<br/>
  b. The slope and y-intercept of the line connecting two points<br/>
  c. The y-values of two points<br/>
  d. The slope of the line connecting two points<br/>

  Ans: 


## 12.4 Linear Regression

0. If your data set contains 10 colinear points, meaning they are all points on the same line, should you use a linear regression to find that line? (yes/no)

  Ans: 


1. When a system has more dimensions than points, it is called an “overdetermined system”. (True/False)

  Ans: 


2. The purpose of linear regression is to find a line that most closely matches a set of data with multiple data points. (True/ False)

  Ans: 


3. Given points $p_1=(2,3)$ and $p_2=(3,0)$, and the equation ${\bf A \vec{w}} = \vec{b}$ answer the following:

  1) Find the coefficient matrix, $A$.<br/>
    <span style="padding-left: 1em;">a.</span> $A = \begin{bmatrix} 1 \ 1 \\ 2 \ 3 \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">b.</span> $A = \begin{bmatrix} 1 \ 2 \\ 1 \ 3 \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">c.</span> $A = \begin{bmatrix} 1 \ 3 \\ 1 \ 2 \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">d.</span> $A = \begin{bmatrix} 1 \ 3 \\ 2 \ 1 \end{bmatrix}$<br/>

  2) Find the dependent variable vector, $\vec{b}$.<br/>
    <span style="padding-left: 1em;">a.</span> $\vec{b} = \begin{bmatrix} 9 \\ {-3} \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">b.</span> $\vec{b} = \begin{bmatrix} 3 \\ { 0} \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">c.</span> $\vec{b} = \begin{bmatrix} 9 \\ {-1} \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">d.</span> $\vec{b} = \begin{bmatrix} 6 \\ { 2} \end{bmatrix}$<br/>

  3) Solve for the parameter vector, $\vec{w}$.<br/>
    <span style="padding-left: 1em;">a.</span> $\vec{w} = \begin{bmatrix} 9 \\ -3 \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">b.</span> $\vec{w} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">c.</span> $\vec{w} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$<br/>
    <span style="padding-left: 1em;">d.</span> $\vec{w} = \begin{bmatrix} -3 \\ 1 \end{bmatrix}$<br/>

  4) Give the equation for the line connecting $p_1$ and $p_2$.<br/>
    <span style="padding-left: 1em;">a.</span> $y = 3x + 9$<br/>
    <span style="padding-left: 1em;">b.</span> $y = x - 3$<br/>
    <span style="padding-left: 1em;">c.</span> $y = -3x+3$<br/>
    <span style="padding-left: 1em;">d.</span> $y = -3x+9$<br/>

  Ans: 


4. The parameter vector, $\vec{w} \in \Bbb{R}^2$, represents the slope and Y-intercept of a line in the 2-D plane. (True/False)

  Ans: 

  
5. Why do we want to minimize the square difference from a point to the line instead of the actual difference when using the least squares method?<br/>
  a. It’s more accurate to minimize the larger value<br/>
  b. We could minimize the actual difference as well<br/>
  c. We want to ensure the value is positive because it is a distance<br/>
  d. We want to ensure that far away points are weighted more heavily than nearby points<br/>

  Ans: 


## 12.5 Polynomial Regression


## 12.6 Regression Towards the Mean


## 12.7 Principle Component Analysis



# Topic13: Hypothesis Testing


## 13.1 Hypothesis Test - Introduction

0. If we fail to reject the null hypothesis, does it mean that the null hypothesis is correct?<br/>
  a. Yes, it must be correct.<br/>
  b. No, we just don't have enough evidence to reject it.<br/>

  Ans: 


1. The distribution of the test statistic T depends on<br/>
  a. Null hypothesis $H_0$,<br/>
  b. Alternative hypothesis $H_A$,<br/>
  c. Observed data t,<br/>
  d. None of above.<br/>

  Ans: 


2. The null hypothesis says that  Z  follows normal distribution $N(0,\sigma^2)$. If the null hypothesis is correct, which of the following is the most unlikely event?<br/>
  a. $Z \in [−\sigma, \sigma]$<br/>
  b. $Z \notin [−2\sigma,2\sigma]$<br/>
  c. $Z \ge \sigma$<br/>

  Ans: 


## 13.2 Hypothesis Testing - p-Values

0. If the statistic T is observed to be t, the p-value is<br/>
  a. The probability that T=t<br/>
  b. The probability under the null hypothesis that T=t<br/>
  c. The probability under the null hypothesis that T=t or is further towards the alternative hypothesis<br/>
  d. None of the above<br/>

  Ans: 


1. One- and two-sided tests

  We know the male students' height is approximately normal, and has standard deviation 4 inches. In a sample of 10 male students, the mean height is 68 inches. Calculate the p value corresponding to the following null hypotheses.

  1) $H_0$: The average height of male students in this college is 70 inches.<br/>
     <span style="padding-left: 1.2em;">$H_1$: The average height of male students in this college is not 70 inches.</span>

  2) $H_0$: The average height of male students in this college is at least 70 inches.<br/>
    <span style="padding-left: 1.2em;"> $H_1$: The average height of male students in this college less than 70 inches.</span>

  Ans: 
  


2. The null hypothesis says that 20% of college students are left-handed, while the alternative hypothesis says that less than 20% of college students are left-handed. If we took a sample of 20 college students and let $X$ be the number of lefties in the sample. Calculate the p values if

  1) $X=1$

  2) $X=2$
  
  Ans: 
  

3. The null hypothesis states that a random variable follows the standard normal distribution, while the alternative hypothesis states that the random variable has negative mean. Which of the following shaded areas represents the  p  value when the observed outcome is  z ?

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y3ceuemh" ismap target="_blank">
      <img src="img/t13-12.fig1.png" style="margin: 0.1em;" alt="Option of Q3 in T13.2: fig1" title="Option of Q3 in T13.2: fig1" height=130>
      <img src="img/t13-12.fig2.png" style="margin: 0.1em;" alt="Option of Q3 in T13.2: fig2" title="Option of Q3 in T13.2: fig2" height=130>
      <img src="img/t13-12.fig3.png" style="margin: 0.1em;" alt="Option of Q3 in T13.2: fig3" title="Option of Q3 in T13.2: fig3" height=130>
      <img src="img/t13-12.fig4.png" style="margin: 0.1em;" alt="Option of Q3 in T13.2: fig4" title="Option of Q3 in T13.2: fig4" height=130>
    </a>
  </div>

  Ans: 


4. In the following problem we discuss the test comparing two distribution means with the same variance. Assume $X \sim \mathcal{N}(\mu_1, \sigma^2)$, $Y \sim \mathcal{N}(\mu_2, \sigma^2)$, and they are independent.

  1) What is the variance of $X−Y$?<br/>
    a. $\sigma^2$<br/>
    b. $2\sigma^2$<br/>
    c. $\sigma^2/2$<br/>
 
  2) If $\overline{X}$ is the sample mean of $n$ independent random observations of $X$ and  $\overline{Y}$ is the sample mean of $n$ independent random observations of $Y$, what distribution does $\overline{X}-\overline{Y}$ follow?<br/>
    a. $\mathcal{N}(\mu_1-\mu_2,\frac{\sigma^2}{n})$<br/>
    b. $\mathcal{N}(\mu_1-\mu_2,\frac{\sigma^2}{2n})$<br/>
    c. $\mathcal{N}(\mu_1-\mu_2,\frac{2\sigma^2}{n})$<br/>
  
  3) we now want to test the null hypothesis $H_0$

    + $H_0$: In college, the average GPA of men is equal to the average GPA of women.<br/>
    + $H_1$: In college, the average GPA of men is different from the average GPA of women.<br/>

  A sample of 10 men's GPA in college has sample mean 2.9, and a sample of 10 women's GPA has sample mean 3.1. We also know the GPAs of men and women have the same standard deviation 0.2. Calculate the p value.

  Ans: 


## 13.3 Lady Tasting Tea

0. If instead of preparing 4 cups milk-first ,and 4 tea-first, each cup is prepared randomly, with equal probability of milk- and tea-first. Which of the following most closely approximates the p value of correctly guessing all 8 milk/tea orders?<br/>
  a. 0.014<br/>
  b. 0.010<br/>
  c. 0.004<br/>
  d. 0.001<br/>

  Ans: 


1. Continuing the poll question, what is the smallest number of milk-tea cups the lady must guess correctly to reject the null hypothesis with significance level $\alpha=5\%$.

  Ans: 


2. For a real $−1 \le \alpha \le 1$, define $f_\alpha(x)=2\alpha x+1− \alpha$.

  It is easy to see that $f_\alpha$ is non-negative and integrates to 1, namely is a distribution, over $[0,1]$.

  1) Consider the null hypothesis that $\alpha=0$, namely $f_\alpha$ is uniform, and the alternative hypothesis that $\alpha > 0$. Given a single sample, 0.8, from $f_\alpha$, find the  $p$-value.

  2) Find the lowest outcome for rejecting the null hypothesis with 5% significance level.
  
  Ans: 



3. An old scale displays a weight that is uniformly distributed between the real weight  ±10  lbs. For example for a person with weight 100 lbs, the scale will show a weight uniformly distributed between 90 and 110.

  Consider the null hypothesis that a person weighs 100 lbs, and the alternative hypothesis that the weight is lower.

  1) What is the $p$-value (in percentage) of 91?
  
  2) What is the $p$-value (in percentage) of 90?

  3) What is the highest weight in lbs for which we can reject the null hypothesis with significance level 10%?
  

  Ans: 


## 13.4 Hypothesis Testing - Z and T Tests

0. We first calculate the p-value of a sample under a t-test. We then receive additional information about the distribution variance and calculate the p-value again under a z-test. Which of the following do you think will happen?<br/>
  a. The p value will increase.<br/>
  b. The p value will decrease.<br/>
  c. It could be both.<br/>

  Ans: 


1. This is the T-test version of Q5 in section 13.2.

  We now want to test the null hypothesis $H_0$

  + $H_0$: In college, the average GPA of men is equal to the average GPA of women.
  + $H_1$: In college, the average GPA of men is different from the average GPA of women.

  A sample of 10 men's GPA in college has sample mean 2.9, and a sample of 10 women's GPA has sample mean 3.1. We also know the GPAs of men and women have the same  estimated standard deviation  0.2. Calculate the p value.
  
  Ans: 


2. The null hypothesis says that a sprinter's reaction time follows a normal distribution with mean at most 0.150 seconds. Six measurements of a sprinter's reaction time show 0.152, 0.154, 0.166, 0.147, 0.161, and 0.159 seconds. What is the p value?

  Ans: 





# 14. Final Exam

+ [Standard Normal Probability & t-distribution critical values](https://tinyurl.com/y33hyqwu)

1. Statistical Reasoning

  Which type of reasoning can be used for each of the following statements? (Bayesian Reasoning / Frequentist Reasoning)

  a. Based on a recent survey, the fraction of the population that prefer sweet to sour is between 73 and 76 percent.<br/>
  b. There is a 20% chance of rain tomorrow.<br/>
  c. I will bet you `20$` to `1$` that my football team would win tomorrow's match.<br/>
  d. The chance that two random people have the same birthday is at least 1/365.<br/>
  
  Ans: 


2. Which of the following statements hold for all sets A and B?<br/>
  a. $B−A=B\cap A^c$<br/>
  b. $A \times B \subseteq A \cup B$<br/>
  c. $(A \Delta B)−B= \varnothing$<br/>
  d. $(A\cup B)−(A\cap B)=(A−B) \cup (B−A)$<br/>
  
  Ans: 



3. A bag contains 5 red balls and 5 blue balls. Three balls are drawn randomly without replacement. Find:<br/>
  a. the probability that all 3 balls have the same color, <br/>
  b. the conditional probability that we drew at least one blue ball given that we drew at least one red ball.<br/>

  Ans: 


4. Students who party before an exam are twice as likely to fail as those who don't party (and presumably study). If 20% of the students partied before the exam, what fraction of the students who failed went partying?

  Ans: 



5. Random variables X and Y are distributed according to

  \[ \begin{array}{c|ccc}
          X\setminus Y & 1 & 2 & 3\\
          \hline
          1 & 0.12 & 0.08 & 0.20\\
          2 & 0.18 & 0.12 & 0.30
        \end{array}
  \]
  
  and $Z=\max\{X,Y\}$. Evaluate:

  a. X and Y are independent, (Yes/No) <br/>
  b. $P(Y \ne 3)$, <br/>
  c. $P(X < Y)$, <br/>
  d. $E[Z]$, <br/>
  e. $V[Z]$. <br/>

  Ans: 



6. X follows normal distribution $N(\mu,\sigma^2)$ whose pdf satisfies $max_x f(x)=0.0997356$ and cdf satisfies $F(−1)+F(7)=1$. Determine <br/>
  a. $\mu$, <br/>
  b. $\sigma,$ <br/>
  c. $P(X \le 0).$ <br/>

  Ans: 



7. A hen lays eight eggs weighing 60, 56, 61, 68, 51, 53, 69, and 54 grams, respectively. Use the unbiased estimators discussed in class to estimate the weight distribution's

  a. mean,<br/>
  b. variance.<br/>

  Ans: 


8. A biologist would like to estimate the average life span of an insect species. She knows that the insect's life span has standard deviation of 1.5 days. According to Chebyshev's Inequality, how large a sample should she choose to be at least 95% certain that the sample average is accurate to within ±0.2 days?

  Ans: 


9. Suppose that an underlying distribution is approximately normal but with unknown variance. You would like to test $H_0:\mu=50$ vs. $H_1:\mu<50$. Calculate the p-value for the following 6 observations: 48.9, 50.1, 46.4, 47.2, 50.7, 48.0.<br/>
  a. less than 0.01<br/>
  b. between 0.01 and 0.025<br/>
  c. between 0.025 and 0.05<br/>
  d. between 0.05 and 0.1<br/>
  e. more than 0.1<br/>
  
  Ans: 


10. 20% of the items on a production line are defective. Randomly inspect items, and let X1 be the number of inspections till the first defective item is observed, and X5 be the number of inspections till the fifth defective item is observed. In both cases, X1 and X5 include the defective item itself (e.g. if the items are {good,good,defective},X1 is 3 ). Calculate

  a. E(X5),<br/>
  b. V(X5),<br/>
  c. E(X5|X1=4),<br/>
  d. V(X5|X1=4).<br/>

  Ans: 
  


11. (For Fun) Model Selection

  A $k$-piece-constant function is define by $k−1$ thresholds $−100<t_1<t_2<\cdots<t_{k−1}<100$ and $k$ values $a_1,a_2, \dots, a_k$. Let

  \[ f(x) = \left\{ \begin{array} \\  a_1, \quad -100 \le x \lt t_1, \\ a_2, \quad t_1 \le x \lt t_2, \\ \vdots \\ a_i, \quad t_{i - 1} \le x \lt t_i, \\ \vdots \\ a_k, \quad t_{k - 1} \le x \le 100. \end{array} \right. \]

  be a $k$-piece-constant function. Suppose you are given $n$ data points $((x_1,y_1),\dots,(x_n,y_n))$ each of which is generated in the following way:

  1. first, $x$ is drawn according to the uniform distribution over the range  [−100,100] .
  2. second $y$ is chosen to be $f(x)+\omega$  where $\omega$ is drawn according to the normal distribution $N(0,\sigma)$ 

  You partition the data into a training set and a test set of equal sizes. For each $j=1,2,\dots$ you find the $j$-piece-constant function $g_j$ that minimizes the root-mean-square-error (RMSE) on the training set. Denote by $train(j)$ the RMSE on the training set and by  test(j)  the RMSE on the test set.

  Which of the following statements is correct?

  a. $train(j)$ is a monotonically non-increasing function.<br/>
  b. $test(j)$ is a monotonically non-increasing function.<br/>
  c. $test(j)$ has a minimum close to $j=k$<br/>
  d. $train(j)$ has a minimum close to $j=k$<br/>
  e. if $j>n/2$, $train(j)=0$<br/>

  Ans: 


