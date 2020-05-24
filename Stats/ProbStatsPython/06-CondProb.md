# Topic 6: Conditional Probability


## 6.1 Conditional Probability

+ Motivation
  + often having partial information about the world
  + modifying event probabilities
    + unemployment numbers $\to$ stock prices
    + LeBron James injured $\to$ Cavaliers game result
    + sunny weekend $\to$ beach traffic
  + importance
    + improving estimates
    + helping determine original unconditional probabilities

+ Intuitive definition
  + $E, F$: events
  + $\Pr(F \,|\, E)$ = probability that $F$ happens given that $E$ happened<br/>
    $\hspace{4em}$ = fraction pf $E$ occurrences that F also occurs
  + e.g., Even = {2, 4, 6}, $\Pr(2 |\text{ Even }) = \frac{2}{6} = \frac{1}{3}$

+ Example: fair die
  + $\Pr(\{4\}) = \Pr(4) = 1/6$
    + $\Pr( 4 \,|\, \geq 3) = \Pr(4 \,|\, \{3, 4, 5, 4\}) = \tfrac{1}{4}$
    + $\Pr(4 \,|\, \leq 3) = \Pr(4 \,|\, \{1, 2, 3\}) = \tfrac{0}{3} = 0$
  + $\Pr(\leq 2) = \Pr(\{1, 2\}) = 1/3$
    + $\Pr(\leq 2 \,|\, \leq 4) = \Pr(\{1, 2\} \,|\, \{1, 2, 3, 4\}) = \frac{2}{4} = \frac{1}{4}$
    + $\Pr(\leq 2 \,|\, \geq 2) = \Pr(\{1, 2\} \,|\, \{2, 3, 4, 5, 6\}) = \frac{1}{5}$

+ General events - uniform spaces
  
  \[\begin{align*}
    \Pr(F \,|\, E) & = \Pr(X \in F \,|\, X \in E) = \Pr(X \in E \wedge X \in F \,|\, X \in E) \\
    &= \Pr(X \in E \cap F \,|\, X \in X \in E) = \Pr(E \cap F \,|\, E) \\
    &= \frac{|E \cap F|}{|E|}
  \end{align*}\]

  + example: fair die again
    + $\Pr(\text{ Prime } \,|\, \text{ Odd }) = \Pr(\{2, 3, 5\} \,|\, \{1, 3, 5\}) = \frac{|\{2, 3, 5\} \cap \{1, 3, 5\}|}{|\{1, 3, 5\}|} = \frac{|\{3, 5\}|}{|\{1, 3, 5\}|} = \frac{2}{3}$
    + $\Pr(\{4\} \,|\, \text{ Prime}) = \Pr(\{4\} \,|\, \{2, 3, 5\}) = \frac{|\{4\} \cap \{2, 3, 5\}|}{|\{2, 3, 5\}|} = \frac{|\varnothing|}{|\{2, 3, 5\}|} = 0$

+ General spaces
  
  \[\begin{align*}
    \Pr(F \,|\, E) &= \Pr(X \in F \,|\, X \in E) \\
      &= \Pr(X\in E \cap X \in F \,|\, X \in E) = \Pr(X \in E \cap F \,|\, X \in E) \\
      &= \frac{n \cdot \Pr(E \cap F)}{n \cdot \Pr(E)} = \frac{\Pr(E \cap F)}{\Pr(E)}
  \end{align*}\]

  + example: Tetrahedral die = 4-sided die

    \[\begin{align*}
      \Pr(\geq 2 \,|\, \leq 3) &= \frac{\Pr(\geq 2 \cap \leq 3)}{\Pr(\leq 3)} = \frac{\Pr(\{2, 3, 4\}) \cap \{1, 2, 3\})}{\Pr(\{1, 2, 3\})} \\
      &= \frac{\Pr(\{2, 3\})}{\Pr(\{1, 2, 3\})} = \frac{.5}{.6} = \frac{5}{6}
    \end{align*}\]

+ Product rule

  \[ \Pr(F \,|\, E) = \frac{\Pr(E \cap F)}{\Pr(E)}  \to \Pr(E \cap F) = \Pr(E) \cdot \Pr(F \,|\, E) \]

  + example: probability of both red?
    + urn: 1 blue, 2 reds
    + $R_1$: first ball red; $R_2$: second ball red

      \[\Pr(\text{ both red }) = \Pr(R_1) \cdot \Pr(R_2 \,|\, R_1) = \frac{2}{3} \cdot \frac{1}{2} \]

+ General product rule

  \[\begin{align*}
    \Pr(E \cap  F \cap G) &= P((E \cap G) \cap G) = \Pr(E \cap F) \cdot \Pr(G \,|\, E \cap F) \\
    &= \Pr(E) \cdot \Pr(F \,|\, E) \cdot \Pr(G \,|\, E \cap F)
  \end{align*}\]

+ Conditionals are probabilities too
  + Non-negativity: $\Pr(B \,|\, A) \geq 0$
  + Unitarity: $\Pr(\Omega \,|\, A) = 1$
  + Addition: B, C disjoint $\to \Pr(B \cup C \,|\, A) = \Pr(B \,|\, A) + \Pr(C \,|\, A)$


+ [Original Slides](https://tinyurl.com/y98gt8qw)


### Problem Sets

0. Let A and B be two positive-probability events. Does P(A|B)>P(A) imply P(B|A)>P(B)?<br/>
  a. Yes<br/>
  b. Not necessarily<br/>

  Ans: <span style="color: magenta;">a</span><br/>
  Explanation: Yes. $P(A|B)=P(A,B) / P(B)$ and $P(B|A)=P(A,B) / P(A)$. Hence, $P(A|B)>P(A) \iff P(A,B)>P(A) * P(B) \iff P(B|A)>P(B)$.


1. Suppose $P(A)>0$. Find $P(B|A)$ when:<br/>
  a. $B=A$,<br/>
  b. $B \supseteq A$,<br/>
  c. $B=Ω$,<br/>
  d. $B=A \subset $,<br/>
  e. $A∩B= \varnothing$,<br/>
  f. $B= \varnothing$.<br/>

  Ans: a. (1); b. (1); c. (1); d. (0); d. (0); e. (0)<br/>
  Explanation:
    + Given that $A$ happens, $B$ must happens. Hence $P(B|A)=1$.
    + Same as above.
    + Same as above.
    + Given that $A$ happens, $B$ can never happens. Hence $P(B|A)=0$.
    + Same as above.
    + Same as above.


2. If \(A\) and \(B\) are disjoint positive-probability events, then \(P(A|B)\)=<br/>
  a. \(P(A)\),<br/>
  b. \(P(B|A)\),<br/>
  c. \(P(A\cup B)\),<br/>
  d. \(P(A\cap B)\).<br/>

  Ans: bc<br>
  Explanation: Since $A$ and $B$ are disjoint, $P(A|B)=0$. $P(A\ \cap B)=P(B|A)=0$, while $P(A)$ and $P(A∪B)$ are positive as $A$ and $B$ are positive-probability events.


3. Given events $A$, $B$ with $P(A)=0.5$, $P(B)=0.7$, and $P(A \cap B)=0.3$ , find:<br/>
  a. $P(A|B)$ ,<br/>
  b. $P(B|A)$ ,<br/>
  c. $P(A^c|B^c)$ ,<br/>
  d. $P(B^c|A^c)$ .<br/>

  Ans: a. (3/7); b. (3/5); c. (1/3); d. (1/5)<br/>
  Explanation: 
    + $P(A|B)=P(A \cap B)/P(B)=0.3/0.7=3/7$.
    + $P(B|A)=P(B \cap A)/P(A)=0.3/0.5=3/5$.
    + $P(A^c|B^c)=P(A^c \cap B^c)/P(B^c)=0.1/0.3=1/3$.
    + $P(B^c|A^c)=P(B^c \cap A^c)/P(A^c)=0.1/0.5=1/5$.  


4. Find the probability that the outcome of a fair-die roll is at least 5, given that it is at least 4.<br/>
  a. \(\frac{2}{3}\)<br/>
  b. \(\frac{2}{4}\)<br/>
  c. \(\frac{1}{3}\)<br/>
  d. \(\frac{1}{2}\)<br/>

  Ans: a<br/>
  Explanation: $P(\text{at least 5 }|\text { at least 4})=P(\text{at least 5 } \cap \text{ at least 4})P(\text{ at least 4 })=P(\text{ at least 5 })P(\text{ at least 4 })=2/3$.


5. Two balls are painted red or blue uniformly and independently. Find the probability that both balls are red if:<br/>
  a. at least one is red,<br/>
  b. a ball is picked at random and it is pained red.<br/>

  Ans: a. (1/3); b(0.5)<br/>
  Explanation:
    + $\Pr(2R| \text{ at least 1R })=\frac{\Pr( \text{ 2R } \cap \text{ at least 1R })}{\Pr( \text{ at least 1R })} = \frac{\Pr(2R)}{\Pr(\text{ at least 1R })}= \frac{1/4}{3/4}=\frac{1}{3}$.
    + $\Pr(2R| \text{ random ball is R} )= \Pr(2R \wedge \text{ random ball is R }) \Pr(\text{ random ball is R })=\Pr(2R)\Pr(\text{ random ball is R })=\frac{1/4}{1/2}=\frac{1}{2}$.


6. Three fair coins are sequentially tossed. Find the probability that all are heads if:<br/>
  a. the first is tails,<br/>
  b. the first is heads,<br/>
  c. at least one is heads.<br/>

  Ans: a. (0); b. (1/4); c. (1/7)<br/>
  Explanation:
    + If the first coin is tails, it's impossible for all coins to be heads, hence the probability is 0. More formally, $\Pr(X_1 \cap X_2 \cap X_3|\overline{X_3}) = \frac{\Pr(X_1 \cap X_2 \cap X_3 \cap \overline{X_3})}{P(\overline{X_3})= \frac{\Pr(\varnothing)}{P(\overline{X_3})= \frac{0}{1/2}=0$.
    + First intuitively, if the first coin is heads, then all are heads iff the second and third coins are heads, which by independence of coin flips happens with probability $\frac{1}{2} \cdot \frac{1}{2}=14$.  A bit more formally, let $X_1,X_2,X_3$ be the events that the first, second, and third coin is heads. Then $\Pr(X_1 \cap X_2 \cap X_3|X_1)=\frac{\Pr(X_1 \cap X_2 \cap X_3 \cap X_1)}{\Pr(X_1)}=\frac{\Pr(X_1 \cap X_2 \cap X_3)}{P(X_1)}=\frac{1/8}{1/2}=\frac{1}{4}$.
    + First intuitively, there are seven possible outcome triples where at least one of the coins is heads, and only one of them has all heads. Hence the probability of all heads given that one is heads is $1/7$.  More formally, $\Pr(X_1 \cap X_2 \cap X_3|X_1 \cup X_2 \cup X_3)=\frac{\Pr((X_1 \cap X_2 \cap X_3) \cap (X_1 \cup X_2 \cup X_3))}{\Pr(X_1 \cup X_2 \cup X_3)}=\frac{P(X_1 \cap X_2 \cap 3)}{\Pr(X_1 \cup X_2 \cup X_3)}=\frac{1/8}{7/8}=\frac{1}{7}$.


7. A 5-card poker hand is drawn randomly from a standard 52-card deck. Find the probability that:<br/>
  a. all cards in the hand are  ≥7  (7, 8,..., K, Ace), given that the hand contains at least one face card (J, Q, or K),<br/>
  b. there are exactly two suits given that the hand contains exactly one queen.<br/>

  Ans: a. (<span style="color: magenta;">0.0957</span>); b. (0.16153846)<br/>
  Explanation:
    + There are where $4 \cdot (13−3)=40$ non-face cards, hence $\tbinom{40}{5}$ hands without face cards. Therefore, of the $\tbinom{52}{5}$ hands, $\binom{52}{5}-\binom{40}{5}$ hands contain a face card.  Similarly, there are $\tbinom{32}{5}$ hands consisting of cards $\geq 7$, of which $\tbinom{20}{5}$ contain no face cards, and $\tbinom{32}{5} − \tbinom{20}{5}$ hands contain a face card. Hence, the requested probability is  $\frac{\tbinom{32}{5} − \tbinom{20}{5}}{\tbinom{52}{5}−\tbinom{40}{5}}=0.0957$
    + There are $4 \cdot \tbinom{48}{4}$ hands with exactly one queen.  To count the number of hands with exactly one queen and two suites, observe that there are 4 ways to choose the queen, then 3 ways to select the other suit, and from the $26−2=24$ non-queens of these two suits, $\tbinom{24}{4}$ ways to select the remaining 4 cards, but of those, $\tbinom{12}{4}$ hands will have all cards of the same suit as the queen. Hence there are $4 \cdot 3 \cdot \left(\tbinom{24}{4}−\tbinom{12}{4}\right)$  ways to select cards with exactly one queen and two suits.  The desired probability is therefore, $\frac{4 \cdot 3 \cdot \left(\tbinom{24}{4}−\tbinom{12}{4}\right)}{4 \cdot \tbinom{48}{4}}=0.156$.


### Lecture Video

<a href="https://tinyurl.com/y8wa5gpx" target="_BLANK">
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









