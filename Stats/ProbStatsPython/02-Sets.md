# Topic 2: Sets


## 2.1 Notation

+ Elements
  + foundations, building blocks of sets
  + can be anything
    + structure
    + numbers

+ Elements to sets
  + beyond individual elements
  + "Bigger picture"
  + set: collection of elements
  + define specify elements

+ Specification
  + explicit: compact and expressive
    + coin: {heads, tails}
    + bits: {0, 1}
    + die: {1, 2, 3, 4 ,5 6}
  + implicit: ambiguous
    + digits: {0, 1, 2, ..., 9}
    + letters: {a, b, ..., z}
    + days: {Monday, ..., Sunday}
  + descriptive: {four-letter words} = {love, like, dear, ...}

+ Common sets
  + integers $\mathbb{Z}$: {..., -2, -2, 0, 1, 2, ...}
  + natural $\mathbb{N}$: {0, 1, 2, ...}
  + rationals $\mathbb{Q}$: {integer ratios $m/n, \; n \neq 0$}
  + Reals |mathbb{R}$
  + convention:
    + set: Upper case, e.g., A
    + elements: lower case;, e.g., a

+ Membership
  + if element $x$ is in a set $A$, it is a <span style="color: red; font-weigh: bold;">member</span> of, or <span style="color: red; font-weigh: bold;">belongs</span> to $A$, denoted $x \in A$
    + e.g., $0 \in \{0, 1\}, \;1 \in \{0, 1\}, \;\pi \in \mathbb{R}$
  + Equivalently, $A$ <span style="color: red; font-weigh: bold;">contains</span> $x$, written $A \ni x$
    + e.g., $\{0, 1\} \ni 0, \;\{0, 1\} \ni 1, \;\mathbb{R} \ni \pi$
  + If $x$ is <span style="color: red; font-weigh: bold;">not</span> in $A$, then $x$ is <span style="color: red; font-weigh: bold;">not a member</span>, or does <span style="color: red; font-weigh: bold;">not belong</span> to $A$, denoted $x \notin A$
    + e.g., $2 \notin \{0, 1\}, \;\pi \notin \mathbb{Q}$
  + Equivalently, $A$ does <span style="color: red; font-weigh: bold;">not contain</span> $x$, $A \not\ni x$
    + e.g., $\{0, 1\} \not\ni 2, \;\mathbb{Q} \not\ni \pi$

+ Don't matter
  + order: $\{0, 1 \} = \{1, 0 \}$
  + repetition: $\{0, 1 \} = \{0, 1, 1\}$
  + order matters: using <span style="color: blue; font-weigh: bold;">ordered tuples</span>: $(0, 1) \neq (1, 0)$
  + repetition matters: using <span style="color: blue; font-weigh: bold;">multisets</span>, or <span style="color: blue; font-weigh: bold;">bags</span>

+ Special sets
  + empty set: containing no elements, $\varnothing$ or $\{ \}$
    + e.g., $\forall\, x, \,x \in \varnothing$, $\forall$- All, every
  + universal set: all possible elements, $\Omega$
    + e.g., $\forall, \;x \in \Omega$
  + $\Omega$: considering only relevant elements
    + e.g., integers - $\Omega = \mathbb{Z}$, "prime" = 2, 3, 5, ...
  + $\Omega$ depending on application
    + e.g., temperature - $\Omega = \mathbb{R}$, text - $\Omega = \{words\}$
  + only one $\varnothing$: set w/o elements

+ Set definition in Python
  + define a set: $\{\dots\}$ or $\text{set(\{\dots\})}$
    + e.g., `Set1 = {1, 2}; print(Set1)`, `Set2 = set({2, 3}); print(Set2)`
  + empty set: using only `set()` or `set({})`
    + e.g., `Empty1 = set(); type(Empty1) # set; print(Empty1) #set{}`
    + e.g., `Empty2 = set({}); type(Empty2) # set; print(Empty2) $set{}`
    + e.g., `NotASet = {}; type(NotASet) #dict`

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

+ Testing if empty, size
  + test empty: `not`
    + e.g., `S = set(); not S # True`, `T = {1, 2}; not T # False`
  + size: `len()`
    + e.g., `print(len(S)) # 0`, `print(len(T)) # 2`
  + check if size is 0: `len() = 0`
    + e.g., `print(len(S) == 0) # true`, `print(len(T) == T) # False`


### Problem Sets



### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 2.2 Basic Sets





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

