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




## Sets

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






