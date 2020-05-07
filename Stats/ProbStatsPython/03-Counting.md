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
  + e.g., $f: \{1, 2, 3\} \to \{a, b, c, d\} \implies f(1) = b, f(2) = a, f(3) = d$
  + two elements able to share image

+ One-to-one mapping
  + $f: A \to B$ is <span style="color: magenta;">1-1</span>, or <span style="color: magenta;">injective</span>, if different elements have different images
  + Definition: (injective) $\forall\, a, a^\prime \in A, a \neq a^\prime \to f(a) \neq f(a^\prime) \text{ and } f(a) = f(a^\prime) \to a = a^\prime$
    + e.g: $A = \{a, b, c\}, f(a) \neq f(b), f(a) \neq f(c), f(b) \neq f(c)$
  + $f: A \to B$ is <span style="color: magenta;">not 1-1</span> if $\exists\, a \neq a^\prime \in A, f(a) = f(a^\prime)$
    + e.g. $f(b) = f(c)$

+ Set size
  + the number of elements in a set S is called its <span style="color: magenta;">size</span>, or <span style="color: magenta;">cardinality</span>, and denoted <span style="color: magenta;">$|S|$</span> or <span style="color: magenta;">$\#S$</span>
  + $n$-set: set of size $n$
  + examples
    + bits: $|\{0, 1\}| = 2$
    + coin: $\{\text{heads}, \text{tails}}| = 2$
    + die: $|\{1, 2, 3, ,4 ,5 ,6\}| = 6$
    + digits: $|\{0, 1, \dots, 9\}| = 10$
    + letters: $|\{a, \dots, z\}| = 26$
    + empty set: $|\varnothing| = 0$
    + integers: $\mathbb{Z} = \mathbb{N} = \mathbb{P} = \infty \to$ countable infinite $\aleph_0$
    + Rreals: $|\mathbb{R}| = \infty \to$ uncountably infinite $\aleph$

+ Integer intervals
  + $m \leq n$: $\{m, \dots, n\} = \{\text{integers from } m \text{ to } n, \text{inclusive}\}$, e.g., $\{3, \dots, 5\} = \{3, 4, 5\}$
  + size: $| \{m, \dots, n\} \,|\, n-m+1$
  + examples
    + $|\{5, \dots, 5\}| = |\{5\}| = 1 = 5 - 5 + 1$
    + $|\{1, \dots, 3\}| = |\{1, 2, 3\}| = 3 = 3 - 1 + 1$

+ Integer multiples
  + Definition: (integer multiples) $_d(n] = \{ 1 \leq i \leq n: d \,|\, i\}$
  + remark: 
    + $(n] = [n] = \{ 1, \dots, n\}$
    + size: $|\, _d(n] \,| = \lfloor n/d \rfloor$
  + examples
    + $_3(8] = \{3, 6\} = \{1\cdot 3, 2\cdots 3\}, \quad _3(9] = \{3, 6, 9\} = \{1 \cdot 3, 2 \cdot 3, 3 \cdot 3\}$
    + $|\, _3(8]\,| = \lfloor 8/3 \rfloor = 2, \quad|\, _3(9]\,| = \lfloor 9/3 \rfloor = 3$

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




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 3.2 Disjoint Unions





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





