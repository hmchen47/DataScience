# Topic 3: Counting


## 3.1 Counting

+ Overview
  + Sets are often created via simple operations on existing sets: 
    + unions
    + intersection
    + Cartesian products
  + objective: learn how to determine the sizes of such sets
  + Goal: avoid counting
  + The bijection method

+ Functions
  + a <span style="color: magenta;">function f from A to B</span>, denote <span style="color: magenta;">$f: A \to B$</span>, associates w/ every $a\in A$ and <span style="color: magenta;">image</span> $f(a) \in B$
  + e.g., $f: \{1, 2, 3\} \to \{a, b, c, d\} \textrm{ s.t. } f(1) = b, f(2) = a, f(3) = d$

+ One-to-one mapping
  + $f: A \to B$ is <span style="color: magenta;">1-1</span>, or <span style="color: magenta;">injective</span>, if different elements have different images
  + Definition: (injective) $\forall\mid a, a^\prime \in A, a \neq a^\prime \to f(a) \neq f(a^\prime) \text{ and } f(a) = f(a^\prime) \to a = a^\prime$
    + e.g: $A = \{a, b, c\}, f(a) \neq f(b), f(a) \neq f(c), f(b) \neq f(c)$
  + $f: A \to B$ is <span style="color: magenta;">not 1-1</span> if $\exists\mid a \neq a^\prime \in A, f(a) = f(a^\prime)$
    + e.g. $f(b) = f(c)$

+ Set size
  + the number of elements in a set S is called its <span style="color: magenta;">size</span>, or <span style="color: magenta;">cardinality</span>, and denoted <span style="color: magenta;">$|S|$</span> or <span style="color: magenta;">$\#S$</span>
  + $n$-set: set of size $n$
  + examples
    + bits: $|\{0, 1\}| = 2$
    + coin: $|\{\text{heads}, \text{tails}\}| = 2$
    + die: $|\{1, 2, 3, ,4 ,5 ,6\}| = 6$
    + digits: $|\{0, 1, \dots, 9\}| = 10$
    + letters: $|\{a, \dots, z\}| = 26$
    + empty set: $|\varnothing| = 0$
    + integers: $|\mathbb{Z}| = |\mathbb{N}| = |\mathbb{P}| = \infty \to$ countable infinite $\aleph_0$
    + Rreals: $|\mathbb{R}| = \infty \to$ uncountably infinite $\aleph$

+ Integer intervals
  + $m \leq n$: $\{m, \dots, n\} = \{\text{integers from } m \text{ to } n, \text{inclusive}\}$, e.g., $\{3, \dots, 5\} = \{3, 4, 5\}$
  + size: $| \{m, \dots, n\} \mid n-m+1$
  + examples
    + $|\{5, \dots, 5\}| = |\{5\}| = 1 = 5 - 5 + 1$
    + $|\{1, \dots, 3\}| = |\{1, 2, 3\}| = 3 = 3 - 1 + 1$

+ Integer multiples
  + Definition: (integer multiples) $_d(n] = \{ 1 \leq i \leq n: d \mid i\}$
  + remark: 
    + $(n] = [n] = \{ 1, \dots, n\}$
    + size: $|\mid _d(n] \mid| = \lfloor n/d \rfloor$
  + examples
    + $_3(8] = \{3, 6\} = \{1\cdot 3, 2\cdots 3\}, \quad _3(9] = \{3, 6, 9\} = \{1 \cdot 3, 2 \cdot 3, 3 \cdot 3\}$
    + $|\mid _3(8]\mid| = \lfloor 8/3 \rfloor = 2, \quad|\mid _3(9]\mid| = \lfloor 9/3 \rfloor = 3$

+ Set size in Python
  + size: `len`, e.g., `print(len({-1, 1})) # 2`
  + sum: `sum`, e.g., `print(sum({-1, 1})) # 0`
  + minimum: `min`, e.g., `print(min({-1, 1})) # -1`
  + maximum: `max`, e.g., `print(max({-1, 1})) # 1`
  + loops: `for <var> in <set>`
    + example

      ```python
      A = {1, 2, 3}; print(len(A))    # 3
      num = 0 
      for i in A:
          num += 1
      print(num)  # 3
      ```


+ [Original Slides](https://tinyurl.com/yaa4etch)


### Problem Sets

0. The Python definition `A = set(range(1,10))` implies that A has size<br/>
  a. 2<br/>
  b. 9<br/>
  c. 10<br/>
  d. 11<br/>

  Ans: b<br/>
  Explanation: A has size 9 as the elements are 1 to 9.


1. (Perfect squares) A square of an integer, for example, 0, 1, 4 and 9, is called a _perfect square_. How many perfect squares are $\leq 100$?

  Ans: 11 <br/>
  Explanation: The perfect squares $\leq 100$ are $0^2, 1^2, 2^2, \dots, 10^2$. Hence there are 11.


2. Which of the following sets are finite?<br/>
  a. Weeks in a year<br/>
  b. Students at UCSD<br/>
  c. Odd primes<br/>
  d. Positive integer divisors of 30<br/>

  Ans: abd<br/>
  Explanation
    + True.
    + True. Despite appearances, luckily, UCSD has only a finite number of students.
    + False.
    + True. It is {1,2,3,5,6,10,15,30}.


3. Which of the following sets are finite?<br/>
  a. $\{ x \in \mathbb{Z} \mid x^2 \leq 10\}$<br/>
  b. $\{ x \in \mathbb{Z} \mid x^3 \leq 10\}$<br/>
  c. $\{ x \in \mathbb{N} \mid x^3 \leq 10\}$<br/>
  d. $\{ x \in \mathbb{R} \mid x^2 \leq 10\}$<br/>
  e. $\{ x \in \mathbb{R} \mid x^3 = 10\}$<br/>

  Ans: ace<br/>
  Explanation
    + True. It is $\{-3, -2, \dots, 3\}$.
    + False. It is $\{x \in \mathbb{Z} \mid x \leq 2\}$.
    + True. It is  {0,1,2}.
    + False. It is  $\{x \in \mathbb{R} \mid -\sqrt{10} \leq x \leq \sqrt{10}\}$ .
    + True. It is  $\{\sqrt[3]{10}\}$.



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.2 Disjoint Unions

+ Disjoint unions
  + a union of disjoint sets is called a <span style="color: magenta;">disjoint union</span>
    + e.g., $|A| = 2, |B| = 3, A \cap B = \varnothing \to |A \cup B | = 2 + 3 = 5$
  + for disjoint union sets, the size of the union is the sum of the size for each set
    + $|A \cup B | = |A| + |B|$
  + addition rule: `+`
    + numerous applications & implications
    + reason: $\cup \approx +$
  + example: kids play
    + class w/ 2 boys and 4 girls
    + \# students = ? $\implies \cup \to + \to$ \# students = 2 + 3 = 5
  + example: jar w/ marbles
    + 1 blue, 2 green, 3 red
    + \# marbles = ? $\implies \cup$ of 3 sets $\to$ twice $\to$ \# marbles = 1 + 2 + 3 = 6

+ Complements
  + Quintessential disjoint sets: $A$ and $A^c$
    + $A \cup A^c = \Omega$
    + $|\Omega| = |A \cup A^c| = |A| + |A^c|$
  + subtraction (or complement) rule: `-`
    + $|A^c| = |\Omega| - |A|$
    + reason: set difference $\approx -$
  + examples
    + $D = \{ i \in [6]: 3|i\} = \{3, 6\} \to |D| = 2$
    + $D^c = \{i \in [6]: 3 \nmid i\} = \{1, 2, 4, 5\} \to |D^c| = 4$
    + $\Omega = [6] = \{1, \dots, 6\} \text{ s.t. } |D^c| = 4 = 6 - 2 = |\Omega| - |D|$

+ Think outside the circle
  + handy for large or complex sets
  + $|A^c| = |\Omega| - |A| \to |A| = |\Omega| - |A^c|$
  + examples - numbers
    + $A = \{ i \in [100]: 3 \nmid i\} = \{1, 2, 4, 5, 7, \dots, 100\}$ and $\Omega = \{1, \dots, 100\}$
    + $A^c = \{i \in [100] : 3 | i\} = \{3, 6, 9, \dots, 999\} \text{ s.t. } |A^c| = 33$
    + $|A| = |\Omega| - |A^c| = 100 - 33 = 67$
  + Days = {M, Tu, W, Th, F, Sa, Su}
    + \# weekdays? |{1, 2, 3, 4, 5}| = 5
    + \# weekend?   |Days| - |Weekend| = 7 - 5 = 2
  + letters
    + vowels = {a, e, i, o ,u}
    + \# consonants?  26 - 5 = 21
    + facetious question: word containing all 5 vowels, in order?

+ General subtraction rule
  + $\exists\, A \cup B \text{ s.t. } |B - A| = |B| - |A|$
  + proof: $A \cup B \to B = A \cup (B-A) \to |B| = |A| + |B-A| \text{ s.t. } |B-A| = |B| - |A|$


+ [Original Slides](https://tinyurl.com/ybtgvfus)


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.3 General Unions





### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.4 Cartesian Products





### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.5 Cartesian Powers





### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.6 Variations





### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.7 Trees





### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Matplotlib and Numpy.random Notebook





## Lecture Notebook 3






## Programming Assignment 3





