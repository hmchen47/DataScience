# Topic 6: Conditional Probability


## 6.1 Conditional Probability

+ Why condition
  + often having partial information about the world
  + modifying event probabilities
    + unemployment numbers $\to$ stock prices
    + LeBron James injured $\to$ Cavaliers game result
    + sunny weekend $\to$ beach traffic
  + can help:
    + improving estimates
    + determining original unconditional probabilities

+ Basic to basics
  + empirical frequency interpretation of probability
  + probability $\Pr(E)$ of event $E$: the fraction of experiments where $E$ occurs as \# experiments $\to$ \infty
  + to estimate $\Pr(E)$ repeating the experiment many times, findingfraction of experiments where $E$ occurs
  + example: fair die -- $\Pr(2) = \frac{2}{12} = \frac{1}{6}$

+ Conditional probability
  + Definition: (conditional probability) $\exists \text{ events } E, F.$ The conditional probability <span style="color: magenta;">$\Pr(F|E)$</span> of $F$ given $E$ is the fraction of times $F$ occurs in experiments where $E$ occurs
  + to estimate $\Pr(F|E)$ taking many samples, considering only experiments where $E$ occurs, and calculate the fraction therein where $F$ occurs too
  + e.g., Even = {2, 4, 6}, $\Pr(2 |\text{ Even }) = \frac{2}{6}{1}{3}$

+ Example: die
  + $\Pr(\{2\}) = \Pr(2) = 1/6 \to \Pr(2 | \text{Odd}) = \Pr(2 | \{1, 3, 5\}) = \frca{0}{6} = 0 $
  + $\Pr(\leq 2) = \Pr(\{1, 2\}) = 1/3 to \Pr(\leq 2 | \geq 2) = \Pr(\{1, 2\} | \{2, 3, 4, 5, 6\}) = \frac{2}{10} = \frac{1}{5} $

+ General events - uniform spaces
  
  \[\begin{align*}
    \Pr(F | E) & = \Pr(X \in F | X \in E) = \Pr(X \in E and X \in F | X \in E) \\
    &= \Pr(X \in E \cap F | X \in X \in E) = \Pr(E \cap F | E) \\
    &= \frac{|E \cap F|}{|E|}
  \end{align*}\]

  + example: fair die again
    + $\Pr(\text{ Prime } | \text{ Odd }) = \Pr(\{2, 3, 5\} | \{1, 3, 5\}) = \frac{|\{2, 3, 5} \cap \{1, 3, 5\}|}{|\{1, 3, 5\}|} = \frca{|\{3, 5\}|}{|\{1, 3, 5\}|} = \frac{2}{3}$
    + $\Pr(\{4\} | \text{ Prime}) = \Pr(\{4\} | \{2, 3, 5\}) = \frac{|\{4\} \cap \{2, 3, 5\}|}{|{2, 3, 5\}|} = \frac{\varnothing|}{|\{2, 3, 5\}|} = 0$

+ General spaces
  
  \[\begin{align*}
    \Pr(F | E) &= \Pr(X \in F | X \in E) \\
      &= \Pr[X\in E \cap X \in F | X \in E] = \Pr[X \in E \cap F | X \in E] \\
      &= \frac{\Pr(E \cap F)}{\Pr(E)}
  \end{align*}\]

  + example: Tetrahedral die = 4-sided die

    \[\begin{align*}
      \Pr(\geq 2 | \leq 3) &= \frac{\Pr(\geq 2 \cap \leq 3)}{\Pr(\leq 3)} = \frac{\Pr(\{2, 3, 4\}) \cap \{1, 2, 3\}}{\Pr(\{1, 2, 3\})} \\
      &= \frca{\Pr(\{2, 3\})}{\Pr(\{1, 2, 3\})} = \frca{.5}{.6} = \frac{5}{6}
    \end{align*}\]

+ Conditionals are probabilities too
  + Non-negativity: $\Pr(B | A) \geq 0$
  + Unitarity: $\Pr(\Omega | A) = 1$
  + Addition: B, C disjoint $\to \Pr(B \cup C | A) = \Pr(B | A) + \Pr(C | A)$


+ [Original Slides](https://tinyurl.com/y98gt8qw)


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.2 Independence






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.3 Sequential Probability






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.4 Total Probability






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.5 Bayes' Rule






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 6










## Programming Assignment 6









