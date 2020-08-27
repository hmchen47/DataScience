
# 14. Final Exam


## 14.1 Preparing for the Final Exam

+ [Standard Normal Probability & t-distribution critical values](https://tinyurl.com/y33hyqwu)

1. Statistical Reasoning

  Which type of reasoning can be used for each of the following statements? (Bayesian Reasoning / Frequentist Reasoning)

  a. Based on a recent survey, the fraction of the population that prefer sweet to sour is between 73 and 76 percent.<br/>
  b. There is a 20% chance of rain tomorrow.<br/>
  c. I will bet you `20$` to `1$` that my football team would win tomorrow's match.<br/>
  d. The chance that two random people have the same birthday is at least 1/365.<br/>
  
  Ans: a. (Frequentist); b. (<font style="color: cyan;">Bayesian</font>, xFrequentist); c. (Bayesian); d. (<font style="color: cyan;">Frequentist</font>, xBayesian) <br/>
  Explanation:
    + [Frequentist and Bayesian statistics — the comparison](https://tinyurl.com/y5gjrre4)
      + Frequentists don’t attach probabilities to hypotheses or to any fixed but unknown values in general.
        + However, I know that its value is fixed (not a random one). Therefore, I cannot assign probabilities to the mean being equal to a certain value, or being less/greater than some other value. The most I can do is collect data from a sample of the population and estimate its mean as the value which is most consistent with the data.
      + Bayesians view probabilities as a more general concept. As a Bayesian, you can use probabilities to represent the uncertainty in any event or hypothesis.
        + I agree that the mean is a fixed and unknown value, but I see no problem in representing the uncertainty probabilistically. I will do so by defining a probability distribution over the possible values of the mean and use sample data to update this distribution.
    + [Bayesian and frequentist reasoning in plain English](https://tinyurl.com/y2rpszxc)
      + Frequentist Reasoning
        + I can hear the phone beeping. I also have a mental model which helps me identify the area from which the sound is coming. Therefore, upon hearing the beep, I infer the area of my home I must search to locate the phone.
        + Sampling is infinite and decision rules can be sharp. Data are a repeatable random sample - there is a frequency. Underlying parameters are fixed i.e. they remain constant during this repeatable sampling process.
        + The frequentist is asked to write reports. He has a big black book of rules. If the situation he is asked to make a report on is covered by his rulebook, he can follow the rules and write a report so carefully worded that it is wrong, at worst, one time in 100 (or one time in 20, or one time in whatever the specification for his report says).
      + Bayesian Reasoning
        + I can hear the phone beeping. Now, apart from a mental model which helps me identify the area from which the sound is coming from, I also know the locations where I have misplaced the phone in the past. So, I combine my inferences using the beeps and my prior information about the locations I have misplaced the phone in the past to identify an area I must search to locate the phone.
        + Unknown quantities are treated probabilistically and the state of the world can always be updated. Data are observed from the realised sample. Parameters are unknown and described probabilistically. It is the data which are fixed.
        + The Bayesian is asked to make bets, which may include anything from which fly will crawl up a wall faster to which medicine will save most lives, or which prisoners should go to jail. 
    + [Are you a Bayesian or a Frequentist? (Or Bayesian Statistics 101)](https://tinyurl.com/y57asbsn)
      + One of the basic differences of Bayesian and Frequentists is how they treat the parameters.
      + frequentist statistics: the best (maximum likelihood) estimate for p is p=1014, i.e., $p \approx 0.714$. In this case, the probability of two heads is $0.7142 \approx 0.51$ and it makes sense to bet for the event.
      + Bayesian approach: p is not a value, it's a distribution.
    + [Bayesian vs frequentist Interpretations of Probability](https://tinyurl.com/y498bmtt)
      + frequentist approach
        + the only sense in which probabilities have meaning is as the limiting value of the number of successes in a sequence of trials,
      + Bayesian approach
        + interpret probability distributions as quantifying our uncertainty about the worl
        + meaningfully talk about probability distributions of parameters, since even though the parameter is fixed, our knowledge of its true value may be limited


2. Which of the following statements hold for all sets A and B?<br/>
  a. $B−A=B\cap A^c$<br/>
  b. $A \times B \subseteq A \cup B$<br/>
  c. $(A \Delta B)−B= \varnothing$<br/>
  d. $(A\cup B)−(A\cap B)=(A−B) \cup (B−A)$<br/>
  
  Ans: ad<br/>
  Explanation
    + True.
    + False. Let $A=\{1\},B=\{2\}$. Then $A \times B=\{(1,2)\},A \cup B=\{1,2\}$.
    + False. $(A\Delta B) - B = A - B$.
    + True.


3. A bag contains 5 red balls and 5 blue balls. Three balls are drawn randomly without replacement. Find:<br/>
  a. the probability that all 3 balls have the same color, <br/>
  b. the conditional probability that we drew at least one blue ball given that we drew at least one red ball.<br/>

  Ans: a. (1/6); b. (<font style="color: cyan;">10/11</font>, x5/11)<br/>
  Explanation
    + $P(\text{same color}) = \frac{\binom{5}{3} + \binom{5}{3}}{\binom{10}{3}} = \frac{1}{6}$
    + $P(\text{a least one Blue , at least one Red}) = \frac{\binom{5}{2} \binom{5}{1} + \binom{5}{1} \binom{5}{2}}{\binom{10}{3}}$, $P(\text{at least one Red}) = 1 - \frac{\binom{5}{3}}{\binom{10}{3}}$, $P(\text{a least one Blue | at least one Red})$ $= \frac{P(\text{a least one Blue , at least one Red})}{P(\text{at least one Red})}$ $= \frac{\binom{5}{2} \binom{5}{1} + \binom{5}{1} \binom{5}{2}}{\binom{5}{2} \binom{5}{1} + \binom{5}{1} \binom{5}{2} + \binom{5}{3}}$ $= \frac{10}{11}$


4. Students who party before an exam are twice as likely to fail as those who don't party (and presumably study). If 20% of the students partied before the exam, what fraction of the students who failed went partying?

  Ans: <font style="color: cyan;">1/3</font>, x2/3, x2/15<br/>
  Explanation:
    + Let $F$ be the event that a student fail, $P$ be the event that a student partied, $NP$ be the event that a student did not party. We know that $P(F|P) = 2P(F|NP), P(P) = 0.2, P(NP) = 0.8$. $P(P|F) = \frac{P(F|P) P(P)}{P(F)}$ $= \frac{P(F|P)P(P)}{P(F|P)P(P) + P(F|NP)P(NP)}$ $= \frac{1}{3}$
    + [using Bayes’ Rule to calculate conditional probability](https://tinyurl.com/y3ck6llt)


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

  Ans: a. (yes); b. (0.5); c. (0.58); d. (2.38); e. (0.4756)<br/>
  Explanation:
    + $P(XY) = P(X)P(Y)$
    + $P(Y \ne 3) = 0.12 + 0.08 + 0.18 + 0.12 = 0.5$
    + $P(X \lt Y) = 0.08 + 0.2 + 0.3 = 0.58$
    + $P(Z = z) = \begin{cases} 0.12, \quad z = 1, \\ 0.38, \quad z = 2, \\ 0.5, \quad z = 3. \end{cases}$, $E(Z) = \sum_{z = 1}^{3} z \cdot P(Z = z) = 2.38$
    + $V(Z) = \sum_{z = 1}^{3} (z - E(Z))^2 \cdot P(Z = z) = 0.4756$


6. X follows normal distribution $N(\mu,\sigma^2)$ whose pdf satisfies $max_x f(x)=0.0997356$ and cdf satisfies $F(−1)+F(7)=1$. Determine <br/>
  a. $\mu$, <br/>
  b. $\sigma,$ <br/>
  c. $P(X \le 0).$ <br/>

  Ans: a. (3); b. (4); c. (<font style="color: cyan;">0.2266</font>, x0.7734)<br/>
  Explanation:
    + As $F(−1)+F(7)=1$, $−1$ and $7$ are symmetric with respect to $\mu$, hence $\mu=3$.
    + $\frac{1}{\sqrt{2\pi \sigma^2}} = 0.0997356$. hence, $\sigma = 4$
    + $P(X \le 0) = F(0) = \Phi(\frac{0 - 3}{4}) = 0.2266$


7. A hen lays eight eggs weighing 60, 56, 61, 68, 51, 53, 69, and 54 grams, respectively. Use the unbiased estimators discussed in class to estimate the weight distribution's

  a. mean,<br/>
  b. variance.<br/>

  Ans: a. (59); b. (45.7143)<br/>
  Explanation:
    + $\hat{\mu} = \frac{1}{8}\sum_{i = 1}^{8} x_i$
    + $S^2 = \frac{1}{7}\sum_{i = 1}^{8} (x_i - \hat{\mu})^2$


8. A biologist would like to estimate the average life span of an insect species. She knows that the insect's life span has standard deviation of 1.5 days. According to Chebyshev's Inequality, how large a sample should she choose to be at least 95% certain that the sample average is accurate to within ±0.2 days?

  Ans: <font style="color: cyan;">1125</font>, x750<br/>
  Explanation: $P(| X - \mu| \ge 0.2 ) = \frac{\sigma^2}{0.2^2} \ge 0.05$. As $\sigma^2 = \frac{1.5^2}{N}$, where $N$ is the number of samples, we have $N \ge 1125$


9. Suppose that an underlying distribution is approximately normal but with unknown variance. You would like to test $H_0:\mu=50$ vs. $H_1:\mu<50$. Calculate the p-value for the following 6 observations: 48.9, 50.1, 46.4, 47.2, 50.7, 48.0.<br/>
  a. less than 0.01<br/>
  b. between 0.01 and 0.025<br/>
  c. between 0.025 and 0.05<br/>
  d. between 0.05 and 0.1<br/>
  e. more than 0.1<br/>
  
  Ans: c<br/>
  Explanation: The sample mean $\overline{X} = 48.55$ , and the sample variance is $S^2=2.78$. Hence the T-test statistics is $T = \frac{\overline{X} - \mu}{S / \sqrt{n}} = -2.13$, where $n=6$. The p values is $P_{H_0}(\overline{X} \le \mu) = F_{n - 1}(T) = 0.0432$


10. 20% of the items on a production line are defective. Randomly inspect items, and let X1 be the number of inspections till the first defective item is observed, and X5 be the number of inspections till the fifth defective item is observed. In both cases, X1 and X5 include the defective item itself (e.g. if the items are {good,good,defective},X1 is 3 ). Calculate

  a. E(X5),<br/>
  b. V(X5),<br/>
  c. E(X5|X1=4),<br/>
  d. V(X5|X1=4).<br/>

  Ans: a. (25); b. (100); c. (<font style="color: cyan;">24</font>, x20), d. (80)<br/>
  Explanation:
    + $E(X_5) = \frac{n}{p} = 25$
    + $E(X_5) = \frac{n(1 - p)}{p^2} = 100$
    + eometric distribution is memoryless. $E(X_5 | X_1 = 4) = E(X_4 + 4) = 24$
    + $V(X_5 | X_1 = 4) = V(X_4 + 4) = V(X_4) = 80$


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

  Ans: <font style="color: cyan;">ace</font>, x1c<br/>


## 14.2 Final Exam Instructions

All assignments (including the Final Exam) must be submitted by the course end date. We do not recommend that you wait until the course end date to attempt the exam. Each learner has 4 hour and 30 minutes to complete the final exam. Please see the section below for information on disability or other special requests.

Note: It is your responsibility to review and follow all the rules for the exam. All rules in the Online Proctoring Rules for Learners will apply for the final exam and a breach of any of the rules can and will result in a score of 0; however, for this course's final exam the following will be allowed during the exam.

Items allowed:

+ Pencil(s)
+ Pen(s)
+ Three Blank/Clean Sheets Scratch Paper (Letter size).
+ One Standard Sheet of Notes (front and back, typed or handwritten).
+ Physical Scientific Calculator 

If you have any questions on these allowances, please post a message on the discussion board below.
Online websites during the exam are NOT allowed by the software.

Violating any of edX's rules for the exam will result in an automatic score of 0. 

### Final Exam Instructions

As a reminder: [Onboarding quiz](https://tinyurl.com/yxr8f8sq) is designed to help you ensure your system is compatible with the online proctoring system. Take the Onboarding quiz well in advance of the deadline to ensure your system is compatible. 

Check your edX Dashboard for your deadline to upgrade and be ID-verified. Please make sure your ID verified status has not expired. Confirm this by checking your verification status on your [edX Dashboard](https://tinyurl.com/ohp47kp).

__Please note__: Only those in the Verified track may complete Onboarding and be allowed to complete proctored assessments during this course, which are required for MicroMasters in Data Science Program.


#### Plan your time wisely

Please ensure you have sufficient time to complete the assessment from start to finish, including time for the software to download. It should take no more than 30 minutes total. 

This Onboarding quiz does not count as part of your grade for this course. Working through the steps in this Onboarding quiz will help you confirm that you have the following:

+ a compatible computer system (must be a modern version of Windows or Mac laptop or desktop operating system)
+ a consistent internet connection
+ a private test room (environment) in which to complete the assessment

Read the rules and system requirements at the following links: 

+ [Online Proctoring Rules for Learners](https://tinyurl.com/y2vqvwj3)
+ [Proctoring System Requirements](https://tinyurl.com/y4vyvrlt)


### Quick Tips

+ Onboarding must be completed at least 5 days prior to the proctored exam.
+ You may only take the Onboarding quiz once.
+ Once you have completed the final, remain connected to the internet until the recordings are completely uploaded to Verificient.


### Accessibility and Accommodation Requests

If you are a learner with disabilities, or need extra time or assistance during a proctored assessment, please request it as soon as possible. It could take a few days to receive and respond to requests, so please do not wait. Please see the Final Exam Disability Request Discussion Board. This is just the practice exam, so don't worry (hint: there's only 3 questions). However, when we get to the final exam, if you have a disability that means you need extra time that falls under UCSD's guidelines (see below), you will be able to request support. 

"A disability may be temporary or permanent in nature, can occur at anytime during a student’s college career and may impact a student’s ability to complete day to day activities such as hearing, seeing, standing, sitting, writing, thinking, or interacting with others. This is also known as a current functional limitation. Disabilities can occur in the following areas: psychological, psychiatric, learning, attention, chronic health, physical, vision, hearing, and acquired brain injury." (p.2, Utilizing Disability Services at UC, San Diego: A Guide for Students, Faculty, and Staff). 


### Ending the Quiz

When you have completed the questions, and if the exam timer has not reached 00:00, click the "End My Exam" button on the exam timer bar.

Once the exam is finished, remain connected to the internet until the recordings are completely uploaded to Verificient. Confirm that you want to quit the ProctorTrack application when you are prompted. If you encountered any problems while completing the Onboarding exam please contact edX Support via the contact us form or Verificient. Contact Verificient by phone at +1 (844) 753-2020.



## 14.3 Final Exam







