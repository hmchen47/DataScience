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
  + example: Days
    + Days = {M, Tu, W, Th, F, Sa, Su}
    + \# weekdays? |{1, 2, 3, 4, 5}| = 5
    + \# weekend?   |Days| - |Weekend| = 7 - 5 = 2
  + example: letters
    + vowels = {a, e, i, o ,u}
    + \# consonants?  26 - 5 = 21
    + facetious question: word containing all 5 vowels, in order?

+ General subtraction rule
  + $\exists\, A, B \in \Omega \text{ s.t. } |B - A| = |B| - |A|$
  + proof: $\exists\, A \cup B \to B = A \cup (B-A) \to |B| = |A| + |B-A| \text{ s.t. } |B-A| = |B| - |A|$


+ [Original Slides](https://tinyurl.com/ybtgvfus)


### Problem Sets

0. We saw that the size of a union of two disjoint sets is the sum of their sizes.

  If the two sets are not necessarily disjoint, then the size of their union is:<br/>

  a. At least the sum of the set sizes<br/>
  b. At most the sum of the set sizes<br/>
  c. Could be smaller, same, or larger than the sum of the set sizes.<br/>

  Ans: b<br/>
  Explanation: at most the sum of the sizes as some elements may be in both sets, and adding the sizes counts these elements twice.


1. Which of the following are finite for every finite set  A  and an infinite set  B ?<br/>
  a. $A \cap B$<br/>
  b. $A \cup B$<br/>
  c. $A - B$<br/>
  d. $B - A$<br/>
  e. $A \Delta B$<br/>

  Ans: ac


2. Which of the following pairs A and B satisfy $|A \cup B| = |A| + |B|$?<br/>
  a. $\{1,2\}$ and $\{0,5\}$<br/>
  b. $\{1,2\}$ and $\{2,3\}$<br/>
  c. $\{i \in \mathbb{Z} : |i| \leq 3\}$ and $\{i \in \mathbb{Z} : 2 \leq |i| \leq 5\}$<br/>
  d. {English words starting with the letter 'a'}  and  {English words ending with the letter 'a'}<br/>

  Ans: a<br/>
  Explanation: Explanation $|A \cup B| = |A| + |B|$ holds when $A$ and $B$ are disjoint.


3. $|A \cup B \cup C| = |A| + |B| + |C|$ whenever: (True/False)

  a. $A$ and $B$ are disjoint and $B$ and $C$ are disjoint,<br/>
  b. $A$ and $B$ are disjoint, $B$ and $C$ are disjoint, and $A$ and $C$ are disjoint.<br/>

  Ans: a. (False); b. (True)<br/>
  Explanation: 
  + a. False. Let  A=C={1}  and  B={2} . Then  A  and  B  are disjoint,  B  and  C  are disjoint. But  |A∪B∪C|=2 , while  |A|+|B|+|C|=3 .
  + b. True. Since  A  and  C  are disjoint, and  B  and  C  are disjoint, we must have that  A∪B  and  C  are disjoint. Hence  |A∪B∪C|=|A∪B|+|C| . Since  A  and  B  are disjoint, we have  |A∪B|=|A|+|B| . Hence  |A∪B∪C|=|A|+|B|+|C| .


4. (Non perfect-squares) Recall that a square of an integer, for example, 1, 4 and 9, is called a perfect square. How many integers between 1, and 100, inclusive, are not perfect squares?

  Ans: 90 <br/>
  Explanation: The perfect squares between 1 and 100 are $1^2, 2^2, \dots, 10^2$. Hence there are 10. By the complement rule, 100-10=90 integers between 1 and 100 are not perfect squares.



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.3 General Unions

+ Hairy problem
  + A friend claims she can determine the size of any set instantly and exactly.
  + She can determine the exact # of hairs on your head.
  + Can you ask her some questions to be fairly certain if she tells the truth?

+ Simple solution
  + subtraction rule
    + Ask her how many hairs you have.
      + Remove a small number of hairs, say 8.
    + Ask how many hairs you have now.
      + Difference between her answers should be # hairs removed (8).
  + Can you ask a single question?
    + Ask just how many hairs you removed. $\to$ 8 hairs
  + Zero-knowledge proofs:
    + Prove identity w/o revealing password.

+ General unions
  + disjoint A and B: $|A \cup B| = |A| + |B| \to$ size of union = sum of sizes
  + in general: $|A \cup B| \neq |A| + |B|$
    + e.g., $|\{a\} \cup \{a\}| = |\{a\}| = 1 \neq 2 = |\{a\}| + |\{a\}|$
  + __Principle of Inclusion-Exclusion (PIE)__: $|A \cup B| = |A| + |B| - |A \cap B|$

+ Example: Divisibility by 2 numbers
  + $D_{2 \vee 3} = \{i \in [100]: 2 \mid i \vee 3 \mid i\} = \{2, 3, 4, 6, 8, \dots, 100\} \to |D_{2 \vee 3} | = ?$
  + derivation
    + $D_2 = \{ i \in [100]: 2 \mid i\} = \{2, 4, 6, \dots, 100\}$
    + $D_3 = \{i \in [100]: 3 \mid 3\} = \{3, 6, 9, \dots, 99\}$
    + $D_{2 \vee 3} = D_2 \cup D_3 \text{ w/ PIE s.t. } |D_2 \cup D_3| = |D_2| + |D_3| - |D_2 \cap D_3|$
    + $D_2 \cap D_3 = \{i \in [100]: 2 \mid i \wedge 3 \mid i\}$
    + $\therefore\, |D_{2 \vee 3} = |D_2| + |D_3| - |D_2 \cap D_3| = 50 + 33 - 16 = 67$

+ Multiple sets
  + two sets: $|A \cup B| = |A| + |B| - |A \cap B|$
  + 3 sets: $|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |B \cap C| - |C \cap A| + |A \cap B \cap C|$
  + n sets:

    \[ \left|\bigcup_{i=1}^n A_i \right| = \sum_{1 \leq i \leq n} |A_i| - \sum_{1 \leq i < j \leq n} |A_i \cap A_j| + \cdots + (-1)^{n-1} \left| \bigcap_{i=1}^n A_i \right| \]

+ Example: Polyglots
  + 8 students in class, each speaks C, R, or $\Pi$thon $\to |C \cup R \cup \Pi| = 8$
  + each language spoken by 5 students $\to |C| = |R| = |\Pi| = 5$
  + every language _pair_ is spoken by 3 students $\to |C \cap R| = |C \cap \Pi| = |R \cap \Pi| = 3$
  + how many students speak all 3 languages? $\to |C \cap R \cap \Pi| = ?$

    \[\begin{align*}
      |C \cap R \cup \Pi| & = |C| + |R| + |\Pi| - |C \cap R| - |R \cap \Pi| - |\Pi \cap C| + |C \cap R \cap \Pi| \\
      8 & = 5 + 5 +5 - 3 - 3 - 3 + ? \\
      \therefore\; ? & = 8 - 5 - 5 - 5 + 3 + 3 \\
        &= 2
    \end{align*}\]

+ Sanity checks
  + compare PIE to some expected outcomes
  + $A, B$ disjoint: $|A \cup B| = |A| + |B| - |A \cap B| = |A| + |B|$
  + equal sets: $|A \cup A| = |A| + |A| - |A \cap A| = |A|$
  + general union

    \[ \max \{|A|, |B|\} \underbrace{\leq}_{= \iff \\ \text{nested}} |A \cup B| \underbrace{\leq}_{= \iff \\ \text{disjoint}} |A| + |B| \]


+ [Original Slides](https://tinyurl.com/y9hqcdsn)


### Problem Sets

0. When does $|A \cup B| = |A| + |B|$?<br/>
  a. When at least one of A and B is empty<br/>
  b. When A and B are disjoint<br/>
  c. Both of the above.<br/>

  Ans: <span style="color: magenta;">c</span><br/>
  Explanation: Both of the above. If A and B are disjoint then either by the disjoint union rule or by inclusion exclusion, the size of the union is the sum of the sizes. That implies that if one of the sets is empty the same holds.

1. In a high school graduation exam, 80% of examinees passed the English exam, 85% passed the math exam, and 75% passed both. If 40 examinees failed both subjects, what what the total number of examinees?

  Ans: 100<br/>
  Explanation: By inclusion exclusion (applied to fractions), 80+85-75=90% of the students passed at least one topic. Therefore  10% failed both topics. It follows that 40 students correspond to 10%, hence 400 students took the exam.

2. How many integers in  {1,2,…,100}  do not contain the digit  6 ?

  Ans: 81<br/>
  Explanation:<br/>
    Let $U_6 = \{6,16,…,96\}$ be the sets of integers between 1 and 100 whose units digit is 6, and let $T_6 = \{60,61,…,69\}$ be the corresponding set for the tens digit.<br/>
    The set of integers between 1 and 100 containing 6 is $U_6 \cup T_6$, and by inclusion-exclusion, its size is $|U_6 \cup T_6| = |U_6|+|T_6|-|U_6 \cap T_6|=10+10-1=19$.<br/>
    Hence 100-19=81 integers between 1 and 100 do not contain the digit 6.<br/>

3. Of 100 foreign journalists who can speak Chinese, English or French at a press conference:

  + 60 speak Chinese.
  + 65 speak English.
  + 60 speak French.
  + 35 speak both Chinese and English.
  + 25 speak both Chinese and French.
  + 35 speak both English and French.

  How many journalists speak exactly<br/>
  a. one language<br/>
  b. two languages<br/>
  c. three languages<br/>

  Ans: a. (25); b. (65); c. (10)<br/>
  Explanation: By the Principle of Inclusion and Exclusion for three sets, $100=|A \cup B \cup C| =|A|+|B|+|C|-|A\cap B|-|A\cap C|-|B\cap C|+|A\cap B \cap C|=90+|A\cap B\cap C|$.


4. $|A \cup B|=|A|+|B|$ when<br/>
  a. A and B are disjoint,<br/>
  b. A is the complement of B,<br/>
  c. A and B do not intersect,<br/>
  d. At least one of A and B is empty.<br/>

  Ans: abcd<br/>
  Explanation: It holds whenever  A  and  B  are disjoint.


5. The following equation is incorrect. What needs to be added to make it correct?

  \[ |A \cup B∪C| = |A| + |B| + |C| - |A\cap B| - |A \cap C| - |B \cap C| \]

  a. $-|A \cap B \cap C|$<br>
  b. $+|A \cap B \cap C|$<br>
  c. $+3|A \cap B \cap C|$<br>

  Ans: b


6. In a high school graduation exam 70% of examinees passed the English exam, 76% passed the math exam, and 66% passed both. If 40 examinees failed in both subjects, what is the total number of examinees?

  Ans: 200 <br/>
  Explanation: By the inclusion-exclusion principle, the percentage of students who passed at least one of the two exams is 70+76-66=80. Therefore 20% have failed both subjects. Hence the total number of examinees is 40 / 20% = 200.



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.4 Cartesian Products

+ Cartesian products
  + the size of a Cartesian product = the product of the set sizes
  + product rule: $|A \times B| = |A| \times |B|$
  + examples<br/>
    $|\{a, b\}| = 2 \qquad |\{1, 2, 3\}| = 3$

    \[ \{a, b\} \times \{1, 2, 3\} = \begin{Bmatrix} (a, 1) & (a, 2) & (a, 3) \\ (b, 1) & (b, 2) & (b, 3) \end{Bmatrix} \]

    $\therefore\, |\{a, b\} \times \{1, 2, 3\}| = 3 + 3 = 2 \times 3 = 6$
  + example: Table

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/ycrand47" ismap target="_blank">
        <img src="img/t03-01.png" style="margin: 0.1em;" alt="Table and Cartesian product" title="Table and Cartesian product" width=250>
      </a>
    </div>

+ Rules of Cartesian products
  + 2 sets: $A \times B = \{(a, b): a \in A, b \in B\} \to \text{ rectangle } \implies |A \times B| = |A| \times |B|$
  + 3 sets: 
  
    \[ A \times B \times C = \{(a, bb, c): a \in A, b \in B, c \in C\} \to \text{ 'cuboid' }\\ \implies |A \times B \times C| = |A| \times |B| \times |C|\]

  + example: Dandy dresser
    + 3 shirts, 2 pants, and 5 pairs of shoes
    + how many outfits can he have?
      + Outfits = {(shirt, pants, shoes)} = {shirt} $\times$ {pants} $\times$ {shoes} $\gets$ Cartesian product
      + |{Outfits}| = |{shirt}| $\times$ |{pants}| $\times$ |{shoes}| = 3 $\times$ 3 $\times$ 5 = 30

+ Cartesian product for $n$ sets
  + for $n$  sets, $|A_1 \times A_2 \cdots \times A_n| = |A_1| \times \dots \times |A_n|$
  + example: Subway
    + how many sandwiches can Subway make?
    + Beard = {Wheat, Italian}
    + Meat = {Turkey, Ham, Chicken, Beacon, Beef}
    + Cheese = {American, Monterey, Cheddar}
    + Veggie = {Cucumbers, Lettuce, Spinach, Onions}
    + Sauce = {Ranch, Mustard, Mayonnaise}
    + Sandwiches = Bread $\times$ Meat $ \times$ Cheese $\times$ Veggie $\times$ Sauce

    \[\begin{align*} 
      \therefore\; |\text{Sand's}| &= |\text{Bread}| \times |\text{Meat}| \times |\text{Cheese}| \times |\text{Veggie}| \times |\text{Sauce}| \\
       &= 2 \times 5 \times 3 \times  4 \times 3 = 360
    \end{align*}\]


+ [Original Slides](https://tinyurl.com/ycrand47)


### Problem Sets

0. Do AxB and BxA have the same size for any sets A and B?

  Ans: Yes<br>
  Explanation: Yes, just like a rectangle has the same area if you rotate it 90 degrees


1. (Finite X infinite) If  A  is finite and  B  is infinite then  A×B  can be:<br/>
  a. empty<br/>
  b. nonempty finite<br/>
  c. infinite<br/>

  Ans: ac<br/>
  Explanation: If A is empty then A×\\B , otherwise it is infinite.


2. (Order matters) Which of the following ensures that $A \times B = B \times A$?<br/>
  a. $A = B$<br/>
  b. $A = \varnothing$<br/>
  c. $B = \varnothing$<br/>
  d. $A \cap B = \varnothing$<br/>
  e. $|A| = |B|$<br/>

  Ans: abc<br/>
  Explanation
    + True.
    + True. A×B=B×A=∅ .
    + True. Same as above.
    + False. Let  A={1}, B={2}. Then A×B≠B×A .
    + False. Same as above.


3. (Cartesian product shape) Taking the geometric view of Cartesian products, if A and B are real intervals of positive length in $\mathbb{R}$, then A×B is a:<br/>
  a. line,<br/>
  b. rectangle,<br/>
  c. circle,<br/>
  d. triangle,<br/>
  e. none of above.<br/>

  Ans: b<br/>
  Explanation: Let A=[a,b), B=[c,d). Then $ A \times B = \{(x,y) \mid a \leq x < b, c \leq y < d, x\in \mathbb{R}, y \in \mathbb{R}\}$, which is a rectangle.


4. (Divisors) How many positive divisors does 2016 have?

  Ans: 36 <br/>
  Explanation: 2016=25⋅32⋅7. Hence any positive divisor of 2016 can factored as $2^x \cdot 3^y \cdot 7^z$, where $x \in \{0,1,...,5\},  y \in \{0,1,2\}$ and $z \in \{0,1\}$. By the product rule, there are 6×3×2=36 divisors.


### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.5 Cartesian Powers

+ Cartesian powers of a set
  + Cartesian product of a set w/ itself is a <span style="color: magenta;">Cartesian power</span>
  + Cartesian square: $A^2 = A \times A$
  + $n$-th Cartesian power: $A^n \stackrel{\text{def}}{=} \underbrace{A \times A \times \cdots \times A}_{n}$
  
    \[ |A^n| = |A \times A \times A \times \cdots \times A| = |A| \times |A| \times \cdots \times |A| = |A|^n \]

  + example: California license plates
    + till 1904: no registration
    + 1905 ~1912: various registration formats, one-time $2 fee
    + 1912: 6 digits; $\leq 6$ digits $\implies 10^6$ if all OK
    + 1956: 3 letters + 3 digits; $26^3 \times 10^3 \approx 17.6$ m
    + 1969: 1 digit + 3 letters + 3 digits; $26^3 \times 10^4 \approx 176$ m

+ Binary strings
  + n-bit string: $\{0, 1\}^n = \{\text{ length-n binary strings } \}$

    <table style="font-family: arial,helvetica,sans-serif; width: 30vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:5%;">n</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:30%;">Strings</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Size</th>
      </tr>
      </thead>
      <tbody>
      <tr> <th style="text-align: center;">0</th> <td style="text-align: center;">$\{0, 1\}^0$</td> <td style="text-align: center;">$\Lambda$</td> <td style="text-align: center;">$1$</td></tr>
      <tr> <th style="text-align: center;">1</th> <td style="text-align: center;">$\{0, 1\}^1$</td> <td style="text-align: center;">$0, 1$</td> <td style="text-align: center;">$2$</td></tr>
      <tr> <th style="text-align: center;">1</th> <td style="text-align: center;">$\{0, 1\}^2$</td> <td style="text-align: center;">$00, 01, 10, 11$</td> <td style="text-align: center;">$4$</td></tr>
      <tr> <th style="text-align: center;">3</th> <td style="text-align: center;">$\{0, 1\}^3$</td> <td style="text-align: center;">$000, 001, 011, 010, \\100, 110, 101, 111$</td> <td style="text-align: center;">$8$</td></tr>
      <tr> <th style="text-align: center;">$\dots$</th> <td style="text-align: center;">$\dots$</td> <td style="text-align: center;">$\dots$</td> <td style="text-align: center;">$\dots$</td></tr>
      <tr> <th style="text-align: center;">n</th> <td style="text-align: center;">$\{0, 1\}^n$</td> <td style="text-align: center;">$0\dots 0, \dots, 1\dots 1$</td> <td style="text-align: center;">$2^n$</td> </tr>
      </tbody>
    </table>

  + size of n-bit string: $|\{0, 1\}^n| = |\{0, 1\}|^n = 2^n$

+ Subsets
  + the <span style="color: magenta;">power set</span> of S, denoted <span style="color: magenta;">$\mathbb{P}(S)$</span>, is the collection of all subsets of S
  + $\mathbb{P}(\{a, b\}) = \{\{\}, \{a\}, \{b\}, \{a, b\}\}$
  + 1-1 correspondence btw $\mathbb{P}(S)$ (subset of $S$) and $\{0, 1\}^{|S|}$ (binary strings of length $|S|$): mapping $\mathbb{P}(\{a, b\})$ to $\{0, 1\}^2$
  + $|\mathbb{P}(S)| = ?$

      \[ \left|\mathbb{P}(S)\right| = \left| \{0, 1\}^{|S|} \right| = 2^{|S|} \]

  + the size of the power set = the power of the set size

+ Functions
  + a <span style="color: magenta;">function from A to B</span> maps every elements $a \in A$ to an element $f(a) \in B$
  + define a function $f:\; $ by specifying $f(a), \;\forall\, a \in A$
  + example:
    + $f$ from {1, 2, 3} to {p, u}: specifying f(1), f(2), f(3) $\to$ f(1)=p, f(2)=u, f(3)=p
    + $f:\;$ 3-tuple (f(1), f(2), f(3)) $\to$ (p, u, p)
    + { function from {1, 2, 3} to {p, u} } $\to \{p, u\} \times \{p, u\} \times \{p, u \}$
    + \# functions from {1, 2, 3} to {p, u} = $2 \times 2 \times 2 = |\{p,, u\}|^{|\{1, 2, 3\}|}$
  + generalization
    + { function from A to B } $\implies \underbrace{B \times B \times \cdots \times B}_{|A|} = B^{|A|}$
    + $\therefore\; \text{ # functions from A to B } = |B^{|A|}| = |B|^{|A|}$
  + Exponential growth

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/ycrand47" ismap target="_blank">
        <img src="img/t03-02.png" style="margin: 0.1em;" alt="Exponential growth" title="Exponential growth" width=250>
      </a>
    </div>

+ Cartesian powers and exponential in Python
  + Cartesian : using `product` function in `itertools` library

    ```python
    import itertools
    print(set(itertools.product({2, 5, 9}, repeat=2)))
    # {(5, 9), (5, 5), (2, 9), (9, 2), (9, 9), (2, 2), (9, 5), (2, 5), (5, 2)}
    ```
  
  + exponential: `**`

    ```python
    print(3**2)
    # 9
    ```

+ Example: Chess-Rice Legend
  + chess
    + invented by poor yet clever peasant, become very popular
    + King liked it, offered peasant any reward he wished
  + peasant
    + poor and humble farmer, just need a little rice
    + kindly place a single rice grand on first square, double on each subsequent square
  + king
    + such a modest request
    + granted!
  + king
    + paced one ($2^0$) grain on the first square
    + two ($2^1$) on the second
    + four ($2^2) on the third
    + ...
    + 64th square: $2^{63} \approx 10^{27}$
  + two endings
    + peasant became king
    + peasant beheaded
  + moral:
    + be peasant or be king: beware of exponential!

+ Example: Jeopardy
  + counting questions $\to$ answer

    \[ \# \begin{Bmatrix} \text{n-bit sequences} \\ \text{Subsets of } \{1, \dots, n\} \\ \text{Functions: } \{1, \dots, n\} \text { to } \{0, 1\} \end{Bmatrix} = 2^n \]

  + find a natural counting question whose answer is a double exponential!
    + $? \gets 2^{2^n}$
    + solution 1: power set
      + power set of S: set of subsets of S = $\mathbb{P}(S)$, e.g., $\mathbb({P}(\{a, b\}) = \{ \{\}, \{a\}, \{b\}, \{a, b\}\}$
      + $|\mathbb{P}(S)| = 2^{|S|}$, e.g., $|\mathbb{P}(S)| = 4 = 2^2 = 2^{|\{a, b\}|}$
      + $\mathbb{P}(S)$ is a set $\to$ power set of $\mathbb{P}(S)$
      + $\mathbb{P}(\mathbb{P}(S))$ - set of subsets of $\mathbb{P}(S)$
      + $|\mathbb{P}(S)| = 2^{|S|} \to |\mathbb{P}(\mathbb{P}(S)) | = 2^{|\mathbb{P}(S)s|} = 2^{2^n}$, e.g., $\mathbb{P}(\{a, b\}) = \{ \{\}, \{a\}, \{b\}, \{a, b\}\} \to $

        \[\begin{align*}
          \mathbb{P}(\mathbb{P}(\{a, b\})) &= \mathbb{P}(\{ \{\}, \{a\}, \{b\}, \{a. b\}\}) \\
          &= \left\{ \{\}, \{\{\}\}, \{\{a\}\}, \dots, \{\{\}, \{a\}\}, \dots, \{\{\}, \{a\}, \{b\}, \{a, b\}\}\right\}
        \end{align*}\]

      + $|\mathbb{P}(\mathbb{P}(\{a, b\}))| = 2^{|\mathbb{P}(\{a, b\})|} = 2^{2^{|\{a, b\}|}}$
      + $\therefore\; |\mathbb{P}(\mathbb{P}([n])) = 2^{2^n} \to$ Double exponential
    + solution 2: Boolean functions
      + Functions from A to B $\to B^A \to # = |B|^{|A|$
      + \# Boolean functions of $n$ Boolean (binary) variables
      + functions from $\{0, 1\}^n$ to $\{0, 1\} \to \{0, 1\}^{|\{0, 1\}|^n}$
      + $\# = |\{0, 1\}|^{|\{0, 1\}^n|} = 2^{2^n} \to$ Double exponential

+ circuit w/ $n$ binary inputs, one binary output
  + can implement $2^{2^n}$ functions $\to 2^{63} = 1/2 \cdot 2^{2^6}$

+ Analogies btw number and set operations

  <table style="font-family: arial,helvetica,sans-serif; width: 30vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Numbers</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Sets</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">Addition</td> <td style="text-align: center;">Disjoint union</td> </tr>
    <tr> <td style="text-align: center;">Subtraction</td> <td style="text-align: center;">Complement</td> </tr>
    <tr> <td style="text-align: center;">Multiplication</td> <td style="text-align: center;">Cartesian product</td> </tr>
    <tr> <td style="text-align: center;">Exponents</td> <td style="text-align: center;">Cartesian power</td> </tr>
    </tbody>
  </table>



+ [Original Slides](https://tinyurl.com/y84p7yro)


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





