# Probability: Basics


## Basic Concepts

+ [Why should you care about prob&stat?](../Stats/ProbStatsPython/01-Intro.md#11-introduction-to-probability-and-statistics)
  + a powerful tool to deal w/ uncertainty
  + example: Navigation software
    + Certainty: find the _shortest_ route from a to b
    + Uncertainty: find the _fastest_ rout from a to b

+ [What is Probability Theory?](../Stats/ProbStatsPython/01-Intro.md#12-what-is-probability-theory)
  + Probability theory: a __mathematical__ framework for computing the probability of complex events
  + Assumption: __we know the probability of the basic events.__
  + relying on common sense at first
  + executing some experiments summing $k$ random numbers: $S_k = x_1, x_2 + \cdots + x_k$
  + experiments show that the sum $S_k$ is (almost) always in the range $[-4\sqrt{k}, 4\sqrt{k}]$

    \[\begin{align*}
      k \to \infty &\text{ s.t. }\frac{4\sqrt{k}}{k} = \frac{4}{\sqrt{k}} \to 0 \\
      \therefore\; k \to \infty &\text{ s.t. } \frac{S_k}{k} \to 0
    \end{align*}\]

+ [Math interpretation](../Stats/ProbStatsPython/01-Intro.md#12-what-is-probability-theory)
  + math involved in __proving__ (a precise version of) the statements above
  + in most cases, __approximating__ probabilities using simulations (Monte-Carlo simulations)
  + calculating the probabilities is better because
    + providing a precise answer
    + much faster than Monte-Carlo simulations



## Basic Sets

+ [Set specification](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + classification
    + explicit
    + implicit
    + descriptive: {four-letter words} = {love, like, dear, ...}
  + explicit $\to$ implicit $\to$ descriptive: compact & expressive $\to$ ambiguous

+ [Common sets](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + integers: $\mathbb{Z}$ = {..., -2, -2, 0, 1, 2, ...}
  + natural: $\mathbb{N}$ = {0, 1, 2, ...}
  + positive: \mathbb{P}$ = (1, 2, 3, ...)
  + rationals: $\mathbb{Q}$ = {integer ratios $m/n, \; n \neq 0$}
  + Reals: $\mathbb{R}$ = { ... Google ...}
  + convention:
    + set: Upper case, e.g., A
    + elements: lower case;, e.g., a

+ [Membership](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + if element $x$ is in a set $A$, it is a <span style="color: magenta; font-weigh: bold;">member</span> of, or <span style="color: magenta; font-weigh: bold;">belongs</span> to $A$, denoted $x \in A$
    + e.g., $0 \in \{0, 1\}, \;1 \in \{0, 1\}, \;\pi \in \mathbb{R}$
  + Equivalently, $A$ <span style="color: magenta; font-weigh: bold;">contains</span> $x$, written $A \ni x$
    + e.g., $\{0, 1\} \ni 0, \;\{0, 1\} \ni 1, \;\mathbb{R} \ni \pi$
  + If $x$ is <span style="color: magenta; font-weigh: bold;">not</span> in $A$, then $x$ is <span style="color: magenta; font-weigh: bold;">not a member</span>, or does <span style="color: magenta; font-weigh: bold;">not belong</span> to $A$, denoted $x \notin A$
    + e.g., $2 \notin \{0, 1\}, \;\pi \notin \mathbb{Q}$
  + Equivalently, $A$ does <span style="color: magenta; font-weigh: bold;">not contain</span> $x$, $A \not\ni x$
    + e.g., $\{0, 1\} \not\ni 2, \;\mathbb{Q} \not\ni \pi$

+ [Special sets](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + empty set: containing no elements, $\varnothing$ or $\{ \}$, e.g., $\forall\, x, \,x \in \varnothing$, $\forall$- All, every
  + universal set: all possible elements, $\Omega$, e.g., $\forall\,x, \;x \in \Omega$

+ [Basic Sets](../Stats/ProbStatsPython/02-Sets.md#22-basic-sets)
  + $\{ x \in A \,{\color{Magenta}{|}}\, \dots\} = \{\text{element } x \text{ in } A {\color{Magenta}{\text{ such that }}} \dots \}$ or $\{ x \in A {\color{Magenta}{:}} \dots\}$
  + convention: $[n] = \{1, \dots, n\}$


## Set Relations

+ [Divisibility](../Stats/ProbStatsPython/02-Sets.md#22-basic-sets)
  + $\exists\, m, n \in \mathbb{Z}$, if $n = c \cdot m$ for some $c \in \mathbb{Z}$, we say that <span style="color: magenta;">n is a multiple of $m$</span>, or <span style="color: magenta;">$m$ divides $n$</span> and written <span style="color: magenta;">$m \,|\, n$</span>
  + if no such $c$ exists, <span style="color: magenta;">$m$ does not divide $n$</span>, or <span style="color: magenta;">$n$ is not a multiple of $m$</span>, denoted <span style="color: magenta;">$m \nmid n$</span>

+ [Set of multiples](../Stats/ProbStatsPython/02-Sets.md#22-basic-sets)
  + integer multiples of $m$: $\exists\, m \in \mathbb{Z},\; _m\mathbb{Z} \stackrel{\text{def}}{=} \{ i \in \mathbb{Z}: m \,|\, i\}$
  + multiplies of $m$ in {1..n}: $\exists\; m\in \mathbb{Z}, n \in \mathbb{P}, {}_m[n] \stackrel{\text{def}}{=} \{i \in [n]: m \,|\, i\}$

+ [Equality](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + $=$: <span style="color: cyan;">all</span> elements must be identical, e.g., $\{1, 2, 4\} = \{4, 1, 2\}$
  + $\neq$: <span style="color: cyan;">one different</span> element enough, e.g., $\{1, 2, 4\} \neq \{4, 1, 2, 8\}$

+ [Intersection](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + Remarks
    + $\varnothing$ disjoint any set
    + non-empty $\Omega$ intersects every set
    + a set intersects itself $\iff$ non-empty
  + generalization: several sets
    + <span style="color: magenta;">intersect</span> if <span style="color: cyan;">all share</span> a common element
    + <span style="color: magenta;">mutually disjoint</span> if <span style="color: cyan;">every two</span> are disjoint

+ [Subsets](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + every element in A is also in B $\implies$ A is a <span style="color: magenta;">subset of</span> B, denoted $A \,{\color{Magenta}{\subseteq}}\, B$
  + equivalently, B is a <span style="color: magenta;">superset</span> of, or contains, A, denoted $B \,{\color{Magenta}{\supseteq}}\, A$
  + A has an element that's not in B $\implies$ A is <span style="color: magenta;">not a subset</span> of B, denote $A {\color{Magenta}{\nsubseteq}} B$, or $B {\color{Magenta}{\nsupseteq}} A$
  + Remarks
    + $\mathbb{P} \subseteq \mathbb{N} \subseteq \mathbb{Z} \subseteq \mathbb{Q} \subseteq \mathbb{R}$
    + $\varnothing \subseteq A \subseteq A \subseteq \Omega$
    + $\subseteq$ is <span style="color: magenta;">transitive</span>: $A \subseteq B \wedge B \subseteq C \implies A \subseteq B$
    + $A \subseteq B \wedge B \subseteq A \implies A = B$

+ [Strict subset](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + if $A \subseteq B$ and $A \neq B$, A is a <span style="color: magenta;">strict subset</span> of B, denoted $A {\color{Magenta}{\subset}} B$, and B  is a <span style="color: magenta;">strict superset</span> of A, denoted $B {\color{Magenta}{\supset}} A$
    + e.g., $\{0\} \subset \{0, 1\}$, $\{0, 1\} \supset \{0\}$
  + if A is <span style="color: magenta;">not</span> a strict subset of B, we write $A {\color{Magenta}{\not\subset}} B$ or $B {\color{Magenta}{\not\supset}} A$, w/ two possible reasons

+ [Belongs to $\in$ vs, $\subseteq$ subset of](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + $\in$: relationship btw an <span style="color: cyan;">element</span> and a <span style="color: cyan;">set</span>
  + $\subseteq$: relationship btw <span style="color: cyan;">two sets</span>


## Set Operations

+ [Complement](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + the <span style="color: magenta;">complement</span> $A^c$ of $A$ is the set of $\Omega$ elements not in $A$
  + Definition: (complement set) $A^c = \overline{A} = A^\prime = \{x \in \Omega \,|\, x \not\in A\}$

+ [Set identities](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + relations that hold for all sets
  + Remark
    + $\varnothing^c = \Omega \implies \Omega^c = \varnothing$ 
    + $A$ adn $A^c$: disjoint
    + involution: $(A^c)^c = A$
    + $A \subseteq B \to A^c \supseteq B^c$

+ [Intersection](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + the <span style="color: magenta;">intersection</span>, $A \,{\color{Magenta}{\cap}}\, B$, is the set of elements in both $A$ and $B$
  + Definition: (intersection) $A \cap B = \{x: x\in A \wedge x \in B\}$

+ [Law of sets](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + identities - one set
    + identity: $A \cap \Omega = A \quad A \cup \Omega = \Omega$
    + universal bound: $A \cap \varnothing = \varnothing \quad A \cup \varnothing = A$
    + idempotent: $A \cap A = A \quad A \cup A = A$
    + complement: $A \cap A^c = \varnothing \quad A \cup A^c = \Omega$
  + two and three sets
    + commutative: $A \cap B = B \cap A \quad A \cup B = B \cup A$
    + associative: $(A \cap B) \cap C = A \cap (B \cap C) \quad (A \cup B) \cup C = A \cup (B \cup C)$
    + distributive: $A \cap (B \cup C) = (A \cap B) \cup (A \cap C) \quad A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$
    + De Morgan: $(A \cap B)^c = A^c \cup B^c \quad (A \cup B)^c = A^c \cap B^c$

+ [Set difference](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + the <span style="color: magenta;">difference</span>, $A {\color{Magenta}{-}} B$, is the set of elements in $A$ but not in $B$
  + Definition: (difference) $A - B = A \backslash B = \{x: x\in A \wedge x \not\in B\}$
  + Remark: $A - B = A \cap B^c$

+ [Symmetric difference](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + the <span style="color: magenta;">symmetric difference</span> of two sets is the set of elements in exactly one set
  + Definition: (symmetric difference) $A \Delta B = \{x: x \in A \wedge x \not\in B \vee x \not\in A \wedge x \in B\}$
  + remark: $A \Delta B = (A - B) \cup (B - A)$

+ [Analogies btw number and set operations](../Stats/ProbStatsPython/03-Counting.md#35-cartesian-powers)

  <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
    <thead>
    <tr>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Numbers</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Sets</th>
      <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Python Operator</th>
    </tr>
    </thead>
    <tbody>
    <tr> <td style="text-align: center;">Addition</td> <td style="text-align: center;">Disjoint union</td> <td style="text-align: center;">$+$</td> </tr>
    <tr> <td style="text-align: center;">Subtraction</td> <td style="text-align: center;">Complement</td> <td style="text-align: center;">$-$</td> </tr>
    <tr> <td style="text-align: center;">Multiplication</td> <td style="text-align: center;">Cartesian product</td> <td style="text-align: center;">$\times$</td> </tr>
    <tr> <td style="text-align: center;">Exponents</td> <td style="text-align: center;">Cartesian power</td> <td style="text-align: center;">$\ast\ast$</td> </tr>
    </tbody>
  </table>


## Set Counting

+ [Functions](../Stats/ProbStatsPython/03-Counting.md#31-counting)
  + a <span style="color: magenta;">function f from A to B</span>, denote <span style="color: magenta;">$f: A \to B$</span>, associates w/ every $a\in A$ and <span style="color: magenta;">image</span> $f(a) \in B$
  + $f: A \to B$ is <span style="color: magenta;">1-1</span>, or <span style="color: magenta;">injective</span>, if different elements have different images
  + Definition: (injective) $\forall\mid a, a^\prime \in A, a \neq a^\prime \to f(a) \neq f(a^\prime) \text{ and } f(a) = f(a^\prime) \to a = a^\prime$
  + $f: A \to B$ is <span style="color: magenta;">not 1-1</span> if $\exists\mid a \neq a^\prime \in A, f(a) = f(a^\prime)$

+ [Set size](../Stats/ProbStatsPython/03-Counting.md#31-counting)
  + the number of elements in a set S is called its <span style="color: magenta;">size</span>, or <span style="color: magenta;">cardinality</span>, and denoted <span style="color: magenta;">$|S|$</span> or <span style="color: magenta;">$\#S$</span>
  + $n$-set: set of size $n$

+ [Integer intervals](../Stats/ProbStatsPython/03-Counting.md#31-counting)
  + $m \leq n$: $\{m, \dots, n\} = \{\text{integers from } m \text{ to } n, \text{inclusive}\}$, e.g., $\{3, \dots, 5\} = \{3, 4, 5\}$
  + size: $| \{m, \dots, n\} \mid n-m+1$

+ [Integer multiples](../Stats/ProbStatsPython/03-Counting.md#31-counting)
  + Definition: (integer multiples) $_d(n] = \{ 1 \leq i \leq n: d \mid i\}$
  + remark: 
    + $(n] = [n] = \{ 1, \dots, n\}$
    + size: $|\mid _d(n] \mid| = \lfloor n/d \rfloor$

+ [n Digits: complement](../Stats/ProbStatsPython/03-Counting.md#36-variations)

    \[\begin{align*}
      Z_0^c &= \{x^n: \exists\,i\; x_i \in Z\}^c = \{x^n: \forall\,i\; x_i \notin Z\} = (Z^c)^n \triangleq Z_c\\
      |Z_c| &= |Z^c|^n = 9^n\\\\
      Z_0 &= D^n - Z_c \\
      |Z_0| &= |D^n| - |Z_c|
    \end{align*}\]




## Disjoint Unions

+ [Disjoint unions](../Stats/ProbStatsPython/03-Counting.md#32-disjoint-unions)
  + a union of disjoint sets is called a <span style="color: magenta;">disjoint union</span>
  + for disjoint union sets, the size of the union is the sum of the size for each set
  + addition rule: `+`
    + numerous applications & implications
    + reason: $\cup \approx +$

+ [Complements](../Stats/ProbStatsPython/03-Counting.md#32-disjoint-unions)
  + Quintessential disjoint sets: $A$ and $A^c$
    + $A \cup A^c = \Omega$
    + $|\Omega| = |A \cup A^c| = |A| + |A^c|$
  + subtraction (or complement) rule: `-`
    + $|A^c| = |\Omega| - |A|$
    + reason: set difference $\approx -$

+ [General subtraction rule](../Stats/ProbStatsPython/03-Counting.md#32-disjoint-unions): $\exists\, A, B \in \Omega \text{ s.t. } |B - A| = |B| - |A|$




## General Union

+ [Union](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + the <span style="color: magenta;">union</span>, $A \,{\color{Magenta}{\cup}}\, B$, is the collection of elements in $A$, $B$, or both
  + Definition: (union) $A \cup B = \{x: x \in A \vee x \in B\}$

+ [General unions](../Stats/ProbStatsPython/03-Counting.md#33-general-unions)
  + disjoint A and B: $|A \cup B| = |A| + |B| \to$ size of union = sum of sizes
  + in general: $|A \cup B| \neq |A| + |B|$
  + __Principle of Inclusion-Exclusion (PIE)__: $|A \cup B| = |A| + |B| - |A \cap B|$

+ [Multiple sets](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + $A \cup B \cup C = \{ x \in \Omega: x \in A \vee x \in B \vee x \in X\}$
  + generally, $\bigcup_{i=1}^t A_i = \{x: \exists\, 1\leq i \leq t, \, x \in A\}$
  + similarly, $\bigcap_{i=1}^t A_i = \{x: \exists\, 1\leq i \leq t, \, x \in A\}$

+ [Multiple sets union](../Stats/ProbStatsPython/03-Counting.md#33-general-unions)
  + two sets: $|A \cup B| = |A| + |B| - |A \cap B|$
  + 3 sets: $|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |B \cap C| - |C \cap A| + |A \cap B \cap C|$
  + n sets:

    \[ \left|\bigcup_{i=1}^n A_i \right| = \sum_{1 \leq i \leq n} |A_i| - \sum_{1 \leq i < j \leq n} |A_i \cap A_j| + \cdots + (-1)^{n-1} \left| \bigcap_{i=1}^n A_i \right| \]

+ [Sanity checks](../Stats/ProbStatsPython/03-Counting.md#33-general-unions)
  + compare PIE to some expected outcomes
  + $A, B$ disjoint: $|A \cup B| = |A| + |B| - |A \cap B| = |A| + |B|$
  + equal sets: $|A \cup A| = |A| + |A| - |A \cap A| = |A|$
  + general union

    \[ \max \{|A|, |B|\} \underbrace{\leq}_{= \iff \\ \text{nested}} |A \cup B| \underbrace{\leq}_{= \iff \\ \text{disjoint}} |A| + |B| \]

+ [n digit: Inclusion-Exclusion](../Stats/ProbStatsPython/03-Counting.md#36-variations)
  + $Z_0 = \{x^n: \exists\,i\; x_i = 0\} \quad x^n \triangleq x_1,\dots,x_n$
  + $Z_i = \{x^n: x_i = 0\}$, e.g., $n=4 \text{ s.t. } Z_2 = \{x0yz\} \quad Z_4 = \{xyz0\}$
  + $Z_0 = Z_1 \cup \dots \cup Z_n$
  
    \[\begin{align*}
      |Z_0| &=\quad |Z_1| + |Z_2| + \cdots + |Z_n| \\
        &\quad- |Z_1 \cap Z_2| - |Z_1 \cap Z_3| - \cdots - |Z_{n-1} \cap Z_n| \\
        &\quad+ |Z_1 \cap Z_2 \cap Z_3| + \cdots + |Z_{n-2} \cap Z_{n-1} \cap Z_n|\\
        &\quad \cdots\\
        &\quad+ (-1)^{n-1} |Z_1 \cap Z_2 \cap \cdots Z_n|
    \end{align*}\]




## Cartesian Products

+ [Tuples and ordered pairs](../Stats/ProbStatsPython/02-Sets.md#26-cartesian-products)
  + set: order and repetition not mattered
  + tuple: both order and repetition matter
  + n-tuple: tuple w/ $n$ elements
  + 2-tuple: order pair

+ [Cartesian products](../Stats/ProbStatsPython/02-Sets.md#26-cartesian-products)
  + the <span style="color: magenta;">Cartesian product</span> of $A$ and $B$ is the set $A \,{\color{Magenta}{\times}} B$ of ordered pairs $(a, b)$ where $a \in A$ and $b \in B$
  + Definition: (Cartesian product) $A \times B = \{(a, b) | a \in A,\, b \in B\}$
  + Cartesian plane: $\mathbb{R}^2 = \{(x, y) \,|\, x, y \in \mathbb{R}\}, \; \mathbb{R} \times \mathbb{R} = \mathbb{R}^2$

+ [Identity for Cartesian product](../Stats/ProbStatsPython/02-Sets.md#26-cartesian-products)
  + $A \times \varnothing = \varnothing \times A = \varnothing$
  + $A \times (B \cup C) = A \times B \cup A \times C$
  + $A \times (B \cap C) = A \times B \cap A \times C$
  + $A \times (B - C) = A \times B - A \times C$

+ [Counting of Cartesian products](../Stats/ProbStatsPython/03-Counting.md#34-cartesian-products)
  + the size of a Cartesian product = the product of the set sizes
  + product rule: $|A \times B| = |A| \times |B|$
  + for $n$  sets, $|A_1 \times A_2 \cdots \times A_n| = |A_1| \times \dots \times |A_n|$



## Cartesian Power

+ [Cartesian powers of a set](../Stats/ProbStatsPython/03-Counting.md#35-cartesian-powers)
  + Cartesian product of a set w/ itself is a <span style="color: magenta;">Cartesian power</span>
  + Cartesian square: $A^2 = A \times A$
  + $n$-th Cartesian power: $A^n \stackrel{\text{def}}{=} \underbrace{A \times A \times \cdots \times A}_{n}$
  
    \[ |A^n| = |A \times A \times A \times \cdots \times A| = |A| \times |A| \times \cdots \times |A| = |A|^n \]

+ [Binary strings](../Stats/ProbStatsPython/03-Counting.md#35-cartesian-powers)
  + n-bit string: $\{0, 1\}^n = \{\text{ length-n binary strings } \}$

+ [Subsets](../Stats/ProbStatsPython/03-Counting.md#35-cartesian-powers)
  + the <span style="color: magenta;">power set</span> of S, denoted <span style="color: magenta;">$\mathbb{P}(S)$</span>, is the collection of all subsets of S
  + 1-1 correspondence btw $\mathbb{P}(S)$ (subset of $S$) and $\{0, 1\}^{|S|}$ (binary strings of length $|S|$): mapping $\mathbb{P}(\{a, b\})$ to $\{0, 1\}^2$
  + $|\mathbb{P}(S)| = ?$

      \[ \left|\mathbb{P}(S)\right| = \left| \{0, 1\}^{|S|} \right| = 2^{|S|} \]

  + the size of the power set = the power of the set size

+ [Functions](../Stats/ProbStatsPython/03-Counting.md#35-cartesian-powers)
  + a <span style="color: magenta;">function from A to B</span> maps every elements $a \in A$ to an element $f(a) \in B$
  + define a function $f:\; $ by specifying $f(a), \;\forall\, a \in A$
  + generalization
    + { function from A to B } $\implies \underbrace{B \times B \times \cdots \times B}_{|A|} = B^{|A|}$
    + $\therefore\; \text{ # functions from A to B } = |B^{|A|}| = |B|^{|A|}$




## Tree Structure

+ [Trees and Cartesian products](../Stats/ProbStatsPython/03-Counting.md#37-trees)
  + tree advantages:
    + a tree representing any set of sequences, not just Cartesian products
    + enabling systematic counting technique
    + useful in modeling random phenomena
  + Cartesian products as trees: used only  when all nodes have same degree at any level


## Factorials

+ [0 factorial](../Stats/ProbStatsPython/04-Combinatorics.md#41-permutations)
  + for $n \geq 1$, n! = \# permutations of an n-set = $n \times (n-1) \times \cdots \times 2 \times 1$
  + $0! = 1 \to$ exact same exact same reason as $2^0 =1$

+ [Recursive definition](../Stats/ProbStatsPython/04-Combinatorics.md#41-permutations)
  + n! defined recursively

    \[\begin{align*}
      n! &= n \times (n-1) \times \cdots \times 2 \times 1 \\
      &= n \times [(n-1) \times \cdots \times 2 \times 1] \\
      &= n \times (n-1)! \quad \forall\, n \geq 1
    \end{align*}\]

  + 0 factorial: 1! = 1 x 0!
  + able to extend to negatives

+ [Stirling's approximations](../Stats/ProbStatsPython/04-Combinatorics.md#41-permutations)
  
  \[ n! \sim \sqrt{2\pi n} \left( \frac{n}{e} \right)^n \]


## Partial Permutations

+ [k-permutations](../Stats/ProbStatsPython/04-Combinatorics.md#42-partial-permutations)
  + n-permutation of an n-set: a permutation of the set
  + \# permutations of k out of n objects $\to$ <span style="color: cyan;">k-permutations</span> of n
  + \# k-permutations of an n-set
    + def: $n \times (n-1) \times \cdots \times (n-k+1) \stackrel{\text{def}}{=} n^{\underline{k}}$
    + $k$-th <span style="color: cyan;">falling power</span> of n
    + denoted as $P(n, k)$
  + falling powers simply related to factorial

    \[ n^{\underline{k}} = n \times (n-1) \times \cdots \times (n-k+1) = \frac{n!}{(n-k)!}  \]



## Combinations

+ [k-subsets](../Stats/ProbStatsPython/04-Combinatorics.md#43-combinations)
  + $k$-set: a k-element set
  + $k$-subset: a k-element subset
  + $\dbinom{[n]}{k}$: collection of k-subsets of $[n] = \{1, 2, \dots, n\}$

+ [Sequences w/ k 1's - an analogy to k-element subsets](../Stats/ProbStatsPython/04-Combinatorics.md#43-combinations)
  + $\dbinom{[n]}{k}$: collection of k-subsets of $[n] = \{1, 2, \dots, n\}$
  + 1-1 correspondence to n-bit sequences w/ k 1's
    + same number of elements
    + mostly count sequences
    + same applied to subsets

+ [Number of n-bit sequences w/ k 1's](../Stats/ProbStatsPython/04-Combinatorics.md#43-combinations)
  + binomial coefficient:
    + $\dbinom{n}{k} \triangleq \left|\dbinom{[n]}{k} \right|$ = \# n-bit sequences w/ k 1's
  + locations of 1's
    + ordered pairs from {1, 2, 3}: $\# = 3^\underline{2} = P(3, 2) = 6$
    + non-ordered: $\dbinom{3}{2} = \dfrac{3^{\underline{2}}}{2} = \dfrac{6}{2} = 3$

+ [Calculating the Binomial coefficients](../Stats/ProbStatsPython/04-Combinatorics.md#43-combinations)
  + \# ordered locations: $n^\underline{k} = P(n, k)$
  + every binary sequence w/ k 1's correspondence to k! ordered locations, e.g., $10101 \iff 1,3,5 \quad 1,5,3 \quad 3,1,5 \quad 3,5,1 \quad 5,1,3 \quad 5,3,1$

    \[ k! \dbinom{n}{k} = n^{\underline{k}} \to \dbinom{n}{k} = \frac{n^{\underline{k}}}{k!} = \frac{n!}{k!\,(n-k)!} \]

+ [Simple $\binom{n}{k}$](../Stats/ProbStatsPython/04-Combinatorics.md#43-combinations)
  + all-zero sequence: $\dbinom{n}{0} = \dfrac{n!}{0!n!} = 1$
  + all-one sequence: $\dbinom{n}{n} = \dfrac{n!}{n!0!} = 1$
  + choose location of single 1: $\dbinom{n}{1} = \dfrac{n!}{1!(n-1)!} = n$
    + alternative explanation
      + $\binom{[n]}{2} = \{ \text{n-bit strings w/ two 1's}\}$
      + $A_i = \{x^n: \text{first 1 at location } i\} \quad (1 \leq i \leq n-1)$ (see diagram)
      + $|A_i| = n-i \quad A_i$'s disjoint





