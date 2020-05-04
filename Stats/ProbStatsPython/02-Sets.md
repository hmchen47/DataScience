# Topic 2: Sets

## 2.1 Notation

+ Elements
  + foundations, building blocks of sets
  + can be anything: football player, Google, aspirin
  + structure: letters, words, documents, or web pages
  + numbers: for probability and statistics

+ Elements to sets
  + beyond individual elements
  + "Bigger picture"
  + set: collection of elements
  + define { specify elements }

+ Specification
  + explicit:
    + coin: {heads, tails}
    + bits: {0, 1}
    + die: {1, 2, 3, 4 ,5 6}
  + implicit:
    + digits: {0, 1, 2, ..., 9}
    + letters: {a, b, ..., z}
    + days: {Monday, ..., Sunday}
  + descriptive: {four-letter words} = {love, like, dear, ...}
  + explicit $\to$ implicit $\to$ descriptive: compact & expressive $\to$ ambiguous

+ Common sets
  + integers: $\mathbb{Z}$ = {..., -2, -2, 0, 1, 2, ...}
  + natural: $\mathbb{N}$ = {0, 1, 2, ...}
  + positive: \mathbb{P}$ = (1, 2, 3, ...)
  + rationals: $\mathbb{Q}$ = {integer ratios $m/n, \; n \neq 0$}
  + Reals: $\mathbb{R}$ = { ... Google ...}
  + convention:
    + set: Upper case, e.g., A
    + elements: lower case;, e.g., a

+ Membership
  + if element $x$ is in a set $A$, it is a <span style="color: magenta; font-weigh: bold;">member</span> of, or <span style="color: magenta; font-weigh: bold;">belongs</span> to $A$, denoted $x \in A$
    + e.g., $0 \in \{0, 1\}, \;1 \in \{0, 1\}, \;\pi \in \mathbb{R}$
  + Equivalently, $A$ <span style="color: magenta; font-weigh: bold;">contains</span> $x$, written $A \ni x$
    + e.g., $\{0, 1\} \ni 0, \;\{0, 1\} \ni 1, \;\mathbb{R} \ni \pi$
  + If $x$ is <span style="color: magenta; font-weigh: bold;">not</span> in $A$, then $x$ is <span style="color: magenta; font-weigh: bold;">not a member</span>, or does <span style="color: magenta; font-weigh: bold;">not belong</span> to $A$, denoted $x \notin A$
    + e.g., $2 \notin \{0, 1\}, \;\pi \notin \mathbb{Q}$
  + Equivalently, $A$ does <span style="color: magenta; font-weigh: bold;">not contain</span> $x$, $A \not\ni x$
    + e.g., $\{0, 1\} \not\ni 2, \;\mathbb{Q} \not\ni \pi$

+ Don't matter
  + order: $\{0, 1 \} = \{1, 0 \}$
  + repetition: $\{0, 1 \} = \{0, 1, 1\}$
  + order matters: using <span style="color: cyan; font-weigh: bold;">ordemagenta tuples</span>: $(0, 1) \neq (1, 0)$
  + repetition matters: using <span style="color: cyan; font-weigh: bold;">multisets</span>, or <span style="color: cyan; font-weigh: bold;">bags</span>

+ Special sets
  + empty set: containing no elements, $\varnothing$ or $\{ \}$, e.g., $\forall\, x, \,x \in \varnothing$, $\forall$- All, every
  + universal set: all possible elements, $\Omega$, e.g., $\forall\,x, \;x \in \Omega$
    + $\Omega$: considering only relevant elements, e.g., integers - $\Omega = \mathbb{Z}$, "prime" = 2, 3, 5, ...
    + $\Omega$ depending on application, e.g., temperature - $\Omega = \mathbb{R}$, text - $\Omega = \{\text{words}\}$
  + only one $\varnothing$: set w/o elements

+ Set definition in Python
  + define a set: `{...}` or `set(...)`
    + e.g., `Set1 = {1, 2}; print(Set1) $ {1, 2}`, `Set2 = set({2, 3}); print(Set2) ${2, 3}`
  + empty set: using only `set()` or `set({})`
    + e.g., `Empty1 = set(); type(Empty1) # set; print(Empty1) # set{}`
    + e.g., `Empty2 = set({}); type(Empty2) # set; print(Empty2) # set{}`
    + e.g., `NotASet = {}; type(NotASet) # dict`, `{}` not an empty set

+ Membership
  + $\in$: `in`

    ```python
    Furniture = {'desk', 'chair'}
    'desk' in Furniture     # True
    'bed' in Furniture      # False
    ```

  + $\notin$: `not in`

    ```python
    Furniture = {'desk', 'chair'}
    'desk' not in  Furniture  # False
    'bed' not in Furniture    # True
    ```

+ Testing if empty set, size
  + test empty: `not`
    + e.g., `S = set(); not S # True`, `T = {1, 2}; not T # False`
  + size: `len()`
    + e.g., `print(len(S)) # 0`, `print(len(T)) # 2`
  + check if size is 0: `len() == 0`
    + e.g., `print(len(S) == 0) # True`, `print(len(T) == 0) # False`


### Problem Sets

0. The number zero is an element of the empty set. (True.False)

  Ans: False <br/>
  Explanation: Empty set has no elements. No zero, no zilch, no nada - nothing!


1. The empty set $\varnothing$ is unique.<br/>
   The universal set $\Omega$ is unique.

  Ans: True/False<br/>
  Explanation: $varnothing$ is the unique set having no elements.  $\Omega$ varies. It can be  \mathbb{R}, \mathbb{C}, etc.


2. Which of the following hold?<br/>
  a. $0 \in \{0, 1\}$<br/>
  b. $a \in \{A, B\}$<br/>
  c. $\{a, b\} \in \{\{a, b\}, c\}$<br/>
  d. $a \in \{{a, b}, c\}$<br/>
  e. $\{a\} \in \{a\}$<br/>

  Ans: ac<br/>
  Explanation:
  + True. $\{0, 1\}$  contains two elements, and 0 is one of them.
  + False. $\{A, B\}$ contains two elements,  A  and  B , but not  a .
  + True. $\{\{a, b\}, c\}$ has two elements, the set $\{a, b\}$, and  c .
  + False. $a$ is an element of $\{a, b\}$, not of $\{\{a, b\}, c\}$ .
  + False. $\{a\}$ has only one element $a$, not the set $\{a\}$. More about that in the next lecture.

3. Recall that $\varnothing$ is the empty set. How many elements do the following sets have?
  a. $\varnothing$
  b. $\{\varnothing\}$
  c. $\{\varnothing, \varnothing\}$
  d. $\{\{\varnothing\}, \varnothing\}$

  Ans: a - 0, b - 1, c - 1, d - 2<br/>
  Explanation:
  a. 0, the empty set has no elements.
  b. 1, just the empty set.
  c. 1,  $\{$\varnothing, \varnothing\} = \{\varnothing\}$ , hence contains one element.
  d. 2,  $\varnothing$ and $\{\varnothing\}$.


4. How many elements do the following sets have?
  a. $\{a\}$
  b. $\{a, a\}$
  c. $\{a, \varnothing\}$
  d. $\{\{a\}, a\}$
  e. $\{\{\{a\}\}\}$

  Ans: a - 1, b - 1, c - 2, d - 2, e - 1<br/>
  Explanation:
  + $\{a\}$ has one element, $a$.
  + As we mentioned $\{a, a\} = \{a\}$, hence both sets have one element.
  + $a$ is different from the set $\{a\}$ that contains $a$, hence two elements.
  + One element, the set $\{a\}$.


5. How many elements do the following sets have?<br/>
  a. $\{a, b\}$<br/>
  b. $\{\{a, b\}\}$<br/>
  c. $\{\{a, b\}, \{b, a\}, \{a, b, a\}\}$<br/>
  d. $\{a, b, \{a, b\}\}$<br/>

  Ans: 1 - 2, b - 1, c - 1, d - 3<br/>
  Explanation:
  + The elements are $a$ and $b$, hence 2.
  + There is a single element, the set $\{a, b\}$, hence 1.
  + Again just one element, the set $\{a, b\}$, written thrice, hence 1.
  + The elements are  $a$, $b$, and $\{a, b\}$, hence 3.


6. Animal anagrams<br/>
  Let $A$ be the set of anagrams of singular English animal names. For example, "nails" and "slain" are anagrams of "snail", so all three $\in A$, yet "bar" $\notin A$.

  Which of the following $\in A$?<br/>
  a. tan<br/>
  b. pea<br/>
  c. low<br/>
  d. bare<br/>
  e. loin<br/>
  f. bolster<br/>

  Ans: abcdef<br>
  Explanation: ant, ape, owl, bear, lion, lobster, so all $\in$. Maybe a bit much to learn the meaning of $\in$ and $\notin$, but hopefully you had fun.


7. Elements of a set<br/>
  List the elements of the following sets.

  Write each element once, using 'emptyset' for $\varnothing$ and 'none' if there are no elements. Separate elements by commas, without any spaces.

  a. $\{a\}$<br/>
  b. $\{\{a\}\}$<br/>
  c. $\{a, \{b\}\}$<br/>
  d. $\{\varnothing\}$<br/>
  e. $\varnothing$<br/>

  Ans: a - a; b - {a}; c - a,{b}; d - $\varnothing$; e - none


8. Sets from elements<br/>
  Write the sets containing the following elements.

  Enter your answer without any spaces, list every element once, and write emptyset for $\varnothing$.

  a. $a$<br/>
  b. $a, a$<br/>
  c. $\{a\}$<br/>
  d. $\{a, b\}, \{b, a\}$<br/>
  e. $a,{b}$<br/>
  f. $\varnothing$<br/>
  g. no elements<br/>

  Ans: <br/>
  a. {a}<br/>
  b. {{a}}<br/>
  c. {{a,b}}  or {{b,a}}<br/>
  d. {a,{b}}  or {{b},a}<br/>
  e. {$\varnothing$}  or {emptyset}<br/>
  f. $\varnothing$ or emptyset<br/>



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 2.2 Basic Sets

+ Sets within sets
  + specify a set within a universal, or any other set
    + $\{ x \in A \,{\color{Magenta}{|}}\, \dots\} = \{\text{element } x \text{ in } A {\color{Magenta}{\text{ such that }}} \dots \}$ or $\{ x \in A {\color{Magenta}{:}} \dots\}$
    + e.g., $\mathbb{N} = \{x \in \mathbb{Z} \,|\, x \geq 0\}, \mathbb{P} = \{x \in \mathbb{P} \,|\, x > 0\}$
  + solutions to equations
    + $\{x \in \mathbb{R} \,|\, x^2 \geq 0\} = \mathbb{R}$
    + $\{x \in \mathbb{R} \,:\, x^2 = 1\} = \{-1, 1\}$
    + $\{x \in \mathbb{R} \,|\, x^2 = 0\} = \{0\} \gets$ a single-element set is a <span style="color: Magenta;">singleton</span>
    + $\{ x \in \mathbb{R} \,|\, -1\} = \varnothing$
    + $\{x \in \mathbb{C} \,|\, x^2 = -1\} = \{1, -i \}$

+ Integer intervals
  + integers from $m$ to $n$, inclusive: $\{m, \dots, n\} = \{i \in \mathbb{Z} \,|\, m \leq i \leq n\}$
    + e.g., $\{3, \dots, 5\} = \{i \in \mathbb{Z} \,|\, 3 \leq i \leq 5\} = \{3, 4, 5\}$
    + e.g., $\{3, \dots, 4\} = \{i \in \mathbb{Z} \,|\, 3 \leq i \leq 4\} = \{3, 4\}$
    + e.g., $\{3, \dots, 3\} = \{i \in \mathbb{Z} \,|\, 3 \leq i \leq 3\} = \{3\}$
    + e.g., $\{3, \dots, 2\} = \{i \in \mathbb{Z} \,|\, 3 \leq i \leq 2\} = \varnothing$
  + convention: $[n] = \{1, \dots, n\}$

+ Real intervals
  + $[a, b] = \{x \in \mathbb{R} \,|\, a \leq x \leq b\}$, e.g. [3, 5]
  + $(a, b) = \{x \in \mathbb{R} \,|\, a < x < b\}$, e.g., (3. 5)
  + $[a, b) = \{x \in \mathbb{R} \,|\, a \leq x < b\}$, e.g., [3, 5)
  + $(a, b] = \{x \in \mathbb{R} \,|\, a < x \leq b\}$, e.g., (3, 5]
  + `( )`: not in set; `[ ]`: in set
  + singleton: $[3, 3] = \{3\}$
  + $[3, 2] = [3. 3) = (3, 3] = \varnothing$

+ Divisibility
  + $\exists\, m, n \in \mathbb{Z}$, if $n = c \cdot m$ for some $c \in \mathbb{Z}$, we say that <span style="color: magenta;">n is a multiple of $m$</span>, or <span style="color: magenta;">$m$ divides $n$</span> and written <span style="color: magenta;">$m \,|\, n$</span>
    + $\underbrace{6}_{n} = \underbrace{2}_{c} \cdot \underbrace{3}_{m} \to \underbrace{3}_{m}\,|\,\underbrace{6}_{n}$
    + $\underbrace{-8}_{n} = \underbrace{(-2)}_{c} \cdot \underbrace{4}_{m} \to \underbrace{4}_{m} \,|\, \underbrace{-8}_{n}$
    + $0 = 0 \cdot (-2) to -2 | 0$
  + if no such $c$ exists, <span style="color: magenta;">$m$ does not divide $n$</span>, or <span style="color: magenta;">$n$ is not a multiple of $m$</span>, denoted <span style="color: magenta;">$m \nmid n$</span>
    + $\not\exists\, c \in \mathbb{Z} \textrm{ s.t. } 4 = c \cdot 3 \to 3 \nmid 3$
    + $0 \nmid n \;\forall\, n \neq 0$

+ Quiz
  + Multiples:
    + $3 \,|\, ? \to \{\dots, -6, -3, 0, 3, 6, \dots\}$
    + $1 \,|\, ? \to \mathbb{Z}$
    + $0 \,|\, ? \to 0$
  + Divisors
    + $? \,|\, 4 \to \pm 1, \pm 2, \pm 4$
    + $? \,|\, 0 \to \mathbb{Z}$
    + $? \,|\, \forall\, n\neq 0 \to \pm 1, \pm n$

+ Set of multiples
  + integer multiples of $m$: $\exists\, m \in \mathbb{Z}, _m\mathbb{Z} \stackrel{\text{def}}{=} \{ i \in \mathbb{Z}: m \,|\, i\}$
    + even number: $_2\mathbb{Z} = \{\dots, -4, -2, 0, 2, 4, \dots\} \stackrel{\text{def}}{=} \mathbb{E}$
    + ${}_1\mathbb{Z} = \{\dots, -2, -1, 0, 1, 2, \dots\} = \mathbb{Z}$
    + ${}_0\mathbb{Z} = \{0\}$
  + multiplies of $m$ in {1..n}: $\exists\; m\in \mathbb{Z}, n \in \mathbb{P}, {}_m[n] \stackrel{\text{def}}{=} \{i \in [n]: m \,|\, i\}$
    + $_2[13] = \{i \in \{i, \dots, 13\}: 3\,|\,i\} = \{3, 6, 9, 12\}$
    + $_7[13] = \{7\},\; _1[13] = [13],\; _{14}[13] =\, _0[13] = \varnothing$

+ Intervals and Multiples in Python
  + $\{0, \dots, n-1\}$: `range(n)`
    + note: [n] = {1...n}
    + e.g., `print(set(range(3))) # {0, 1, 2}`
  + $\{m, \dots, n-1\}$: `range(m, n)`
    + e.g., `print(set(range(2, 5))) # {2, 3, 4}`
  + $\{m,\, m+d,\, m+2d,\, \dots\} \leq n-1$: `range(m, n, d)`
    + e.g., `print(set(range(2, 12, 3))) # {8, 2, 11, 5}`



### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 2.3 Venn Diagrams





### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 2.4 Relations





### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 2.5 Operations





### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 2.6 Cartesian Products





### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 2.7 Russell's Paradox





### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 2




## Programming Assignment 2

