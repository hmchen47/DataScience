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
  + $P(F \mid  E)$ = probability that $F$ happens given that $E$ happened<br/>
    $\hspace{4em}$ = fraction pf $E$ occurrences that F also occurs
  + e.g., Even = {2, 4, 6}, $P(2 \mid \text{ Even }) = \frac{2}{6} = \frac{1}{3}$

+ Example: fair die
  + $P(\{4\}) = P(4) = 1/6$
    + $P( 4 \mid  \geq 3) = P(4 \mid  \{3, 4, 5, 4\}) = \tfrac{1}{4}$
    + $P(4 \mid  \leq 3) = P(4 \mid  \{1, 2, 3\}) = \tfrac{0}{3} = 0$
  + $P(\leq 2) = P(\{1, 2\}) = 1/3$
    + $P(\leq 2 \mid  \leq 4) = P(\{1, 2\} \mid  \{1, 2, 3, 4\}) = \frac{2}{4} = \frac{1}{4}$
    + $P(\leq 2 \mid  \geq 2) = P(\{1, 2\} \mid  \{2, 3, 4, 5, 6\}) = \frac{1}{5}$

+ General events - uniform spaces
  
  \[\begin{align*}
    P(F \mid  E) & = P(X \in F \mid  X \in E) = P(X \in E \wedge X \in F \mid  X \in E) \\
    &= P(X \in E \cap F \mid  X \in X \in E) = P(E \cap F \mid  E) \\
    &= \frac{|E \cap F|}{|E|}
  \end{align*}\]

  + example: fair die again
    + $P(\text{ Prime } \mid  \text{ Odd }) = P(\{2, 3, 5\} \mid  \{1, 3, 5\}) = \frac{|\{2, 3, 5\} \cap \{1, 3, 5\}|}{|\{1, 3, 5\}|} = \frac{|\{3, 5\}|}{|\{1, 3, 5\}|} = \frac{2}{3}$
    + $P(\{4\} \mid  \text{ Prime}) = P(\{4\} \mid  \{2, 3, 5\}) = \frac{|\{4\} \cap \{2, 3, 5\}|}{|\{2, 3, 5\}|} = \frac{|\varnothing|}{|\{2, 3, 5\}|} = 0$

+ General spaces
  
  \[\begin{align*}
    P(F \mid  E) &= P(X \in F \mid  X \in E) \\
      &= P(X\in E \cap X \in F \mid  X \in E) = P(X \in E \cap F \mid  X \in E) \\
      &= \frac{n \cdot P(E \cap F)}{n \cdot P(E)} = \frac{P(E \cap F)}{P(E)}
  \end{align*}\]

  + example: Tetrahedral die = 4-sided die

    \[\begin{align*}
      P(\geq 2 \mid  \leq 3) &= \frac{P(\geq 2 \cap \leq 3)}{P(\leq 3)} = \frac{P(\{2, 3, 4\}) \cap \{1, 2, 3\})}{P(\{1, 2, 3\})} \\
      &= \frac{P(\{2, 3\})}{P(\{1, 2, 3\})} = \frac{.5}{.6} = \frac{5}{6}
    \end{align*}\]

+ Product rule

  \[ P(F \mid  E) = \frac{P(E \cap F)}{P(E)}  \to P(E \cap F) = P(E) \cdot P(F \mid  E) \]

  + example: probability of both red?
    + urn: 1 blue, 2 reds
    + $R_1$: first ball red; $R_2$: second ball red

      \[P(\text{ both red }) = P(R_1) \cdot P(R_2 \mid  R_1) = \frac{2}{3} \cdot \frac{1}{2} \]

+ General product rule

  \[\begin{align*}
    P(E \cap  F \cap G) &= P((E \cap G) \cap G) = P(E \cap F) \cdot P(G \mid  E \cap F) \\
    &= P(E) \cdot P(F \mid  E) \cdot P(G \mid  E \cap F)
  \end{align*}\]

+ Conditionals are probabilities too
  + Non-negativity: $P(B \mid  A) \geq 0$
  + Unitarity: $P(\Omega \mid  A) = 1$
  + Addition: B, C disjoint $\to P(B \cup C \mid  A) = P(B \mid  A) + P(C \mid  A)$


+ [Original Slides](https://tinyurl.com/y98gt8qw)


### Problem Sets

0. Let A and B be two positive-probability events. Does $P(A\mid B)>P(A) \implies P(B\mid A)>P(B)$?<br/>
  a. Yes<br/>
  b. Not necessarily<br/>

  Ans: <span style="color: magenta;">a</span><br/>
  Explanation: Yes. $P(A\mid B)=P(A,B) / P(B)$ and $P(B\mid A)=P(A,B) / P(A)$. Hence, $P(A\mid B)>P(A) \iff P(A,B)>P(A) * P(B) \iff P(B\mid A)>P(B)$.


1. Suppose $P(A)>0$. Find $P(B\mid A)$ when:<br/>
  a. $B=A$,<br/>
  b. $B \supseteq A$,<br/>
  c. $B=\Omega$,<br/>
  d. $B=A^c$,<br/>
  e. $A∩B= \varnothing$,<br/>
  f. $B= \varnothing$.<br/>

  Ans: a. (1); b. (1); c. (1); d. (0); d. (0); e. (0)<br/>
  Explanation:
    + Given that $A$ happens, $B$ must happens. Hence $P(B\mid A)=1$.
    + Same as above.
    + Same as above.
    + Given that $A$ happens, $B$ can never happens. Hence $P(B\mid A)=0$.
    + Same as above.
    + Same as above.


2. If \(A\) and \(B\) are disjoint positive-probability events, then \(P(A\mid B)\)=<br/>
  a. \(P(A)\),<br/>
  b. \(P(B\mid A)\),<br/>
  c. \(P(A\cup B)\),<br/>
  d. \(P(A\cap B)\).<br/>

  Ans: bd<br>
  Explanation: Since $A$ and $B$ are disjoint, $P(A\mid B)=0$. $P(A\ \cap B)=P(B\mid A)=0$, while $P(A)$ and $P(A∪B)$ are positive as $A$ and $B$ are positive-probability events.


3. Given events $A$, $B$ with $P(A)=0.5$, $P(B)=0.7$, and $P(A \cap B)=0.3$ , find:<br/>
  a. $P(A\mid B)$ ,<br/>
  b. $P(B\mid A)$ ,<br/>
  c. $P(A^c\mid B^c)$ ,<br/>
  d. $P(B^c\mid A^c)$ .<br/>

  Ans: a. (3/7); b. (3/5); c. (1/3); d. (1/5)<br/>
  Explanation: 
    + $P(A\mid B)=P(A \cap B)/P(B)=0.3/0.7=3/7$.
    + $P(B\mid A)=P(B \cap A)/P(A)=0.3/0.5=3/5$.
    + $P(A^c\mid B^c)=P(A^c \cap B^c)/P(B^c)=0.1/0.3=1/3$.
    + $P(B^c\mid A^c)=P(B^c \cap A^c)/P(A^c)=0.1/0.5=1/5$.  


4. Find the probability that the outcome of a fair-die roll is at least 5, given that it is at least 4.<br/>
  a. \(\frac{2}{3}\)<br/>
  b. \(\frac{2}{4}\)<br/>
  c. \(\frac{1}{3}\)<br/>
  d. \(\frac{1}{2}\)<br/>

  Ans: a<br/>
  Explanation: $P(\text{at least 5 }\mid \text { at least 4})=P(\text{at least 5 } \cap \text{ at least 4})P(\text{ at least 4 })=P(\text{ at least 5 })P(\text{ at least 4 })=2/3$.


5. Two balls are painted red or blue uniformly and independently. Find the probability that both balls are red if:<br/>
  a. at least one is red,<br/>
  b. a ball is picked at random and it is pained red.<br/>

  Ans: a. (1/3); b(0.5)<br/>
  Explanation:
    + $P(2R\mid  \text{ at least 1R })=\frac{P( \text{ 2R } \cap \text{ at least 1R })}{P( \text{ at least 1R })} = \frac{P(2R)}{P(\text{ at least 1R })}= \frac{1/4}{3/4}=\frac{1}{3}$.
    + $P(2R\mid  \text{ random ball is R} )$ $= P(2R \wedge \text{ random ball is R }) P(\text{ random ball is R })$ $=P(2R)P(\text{ random ball is R })$ $=\frac{1/4}{1/2}$ $=\frac{1}{2}$.


6. Three fair coins are sequentially tossed. Find the probability that all are heads if:<br/>
  a. the first is tails,<br/>
  b. the first is heads,<br/>
  c. at least one is heads.<br/>

  Ans: a. (0); b. (1/4); c. (1/7)<br/>
  Explanation:
    + If the first coin is tails, it's impossible for all coins to be heads, hence the probability is 0. More formally, $P(X_1 \cap X_2 \cap X_3\mid \overline{X_3}) $ $= \frac{P(X_1 \cap X_2 \cap X_3 \cap \overline{X_3})}{P(\overline{X_3})}$ $= \frac{P(\varnothing)}{P(\overline{X_3}})$ $= \frac{0}{1/2}=0$.
    + First intuitively, if the first coin is heads, then all are heads iff the second and third coins are heads, which by independence of coin flips happens with probability $\frac{1}{2} \cdot \frac{1}{2}=14$.  A bit more formally, let $X_1,X_2,X_3$ be the events that the first, second, and third coin is heads. Then $P(X_1 \cap X_2 \cap X_3\mid X_1)=\frac{P(X_1 \cap X_2 \cap X_3 \cap X_1)}{P(X_1)}=\frac{P(X_1 \cap X_2 \cap X_3)}{P(X_1)}=\frac{1/8}{1/2}=\frac{1}{4}$.
    + First intuitively, there are seven possible outcome triples where at least one of the coins is heads, and only one of them has all heads. Hence the probability of all heads given that one is heads is $1/7$.  More formally, $P(X_1 \cap X_2 \cap X_3\mid X_1 \cup X_2 \cup X_3)=\frac{P((X_1 \cap X_2 \cap X_3) \cap (X_1 \cup X_2 \cup X_3))}{P(X_1 \cup X_2 \cup X_3)}=\frac{P(X_1 \cap X_2 \cap 3)}{P(X_1 \cup X_2 \cup X_3)}=\frac{1/8}{7/8}=\frac{1}{7}$.


7. A 5-card poker hand is drawn randomly from a standard 52-card deck. Find the probability that:<br/>
  a. all cards in the hand are  ≥7  (7, 8,..., K, Ace), given that the hand contains at least one face card (J, Q, or K),<br/>
  b. there are exactly two suits given that the hand contains exactly one queen.<br/>

  Ans: a. (<span style="color: magenta;">0.0957</span>); b. (0.1562)<br/>
  Explanation:
    + There are where $4 \cdot (13−3)=40$ non-face cards, hence $\tbinom{40}{5}$ hands without face cards. Therefore, of the $\tbinom{52}{5}$ hands, $\binom{52}{5}-\binom{40}{5}$ hands contain a face card.  Similarly, there are $\tbinom{32}{5}$ hands consisting of cards $\geq 7$, of which $\tbinom{20}{5}$ contain no face cards, and $\tbinom{32}{5} − \tbinom{20}{5}$ hands contain a face card. Hence, the requested probability is  $\frac{\tbinom{32}{5} − \tbinom{20}{5}}{\tbinom{52}{5}−\tbinom{40}{5}}=0.0957$
    + There are $4 \cdot \tbinom{48}{4}$ hands with exactly one queen.  To count the number of hands with exactly one queen and two suites, observe that there are 4 ways to choose the queen, then 3 ways to select the other suit, and from the $26−2=24$ non-queens of these two suits, $\tbinom{24}{4}$ ways to select the remaining 4 cards, but of those, $\tbinom{12}{4}$ hands will have all cards of the same suit as the queen. Hence there are $4 \cdot 3 \cdot \left(\tbinom{24}{4}−\tbinom{12}{4}\right)$  ways to select cards with exactly one queen and two suits.  The desired probability is therefore, $\frac{4 \cdot 3 \cdot \left(\tbinom{24}{4}−\tbinom{12}{4}\right)}{4 \cdot \tbinom{48}{4}}=0.156$.


### Lecture Video

<a href="https://tinyurl.com/y8wa5gpx" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.2 Independence
 
+ Motivation
  + $P(F \mid  E) > P(F)$
    + $E \nearrow$ probability of $F$
    + e.g., $P(2 \mid  \text{Even}) = 1/3 > 1/6 = P(2)$
  + $P(F  \mid   E) < P(F)$
    + $E \searrow$ probability of $F$
    + $P(2 \mid  \text{Odd}) = 0 < 1/6 = P(2)$
  + $P(F \mid  E) = P(F)$
    + $E$ neither $\nearrow$ nor $\searrow$ probability of $F$
    + e.g., $P(\text{Even} \mid  \leq 4) = 1/2 = P(\text{Even})$
    + whether or not $E$ occurs, does not change $\mid Pr(F)$
  + motivation $\to$ intuitive definition $\to$ formal

+ Independence - Intuitive
  + informal definition: (independence) Events $E$ and $F$ are <span style="color: magenta;">independent</span> (<span style="color: magenta;">$ E {\perp \!\!\!\! \perp} F$</span>) if occurrence of one does not change the probability  that the other occurs.
  + more formally, $P(F \mid  E) = P(F)$
  + visual interpretation
    + $P(F) = \frac{P(F)}{P(\Omega)}$: $F$ as a fraction of $\Omega$
    + $P(F \mid  E) \triangleq \frac{P(E \cap F)}{P(E)}$: $E \cap F$ as a fraction of $E$

      <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
        <a href="url" ismap target="_blank">
          <img src="img/t06-01.png" style="margin: 0.1em;" alt="Visual interpretation of $P(F\mid E)$" title="Visual interpretation of $P(F\mid E)$" width=200>
        </a>
      </div>
  
+ Independence - formal
  + informal

    \[ P(F) = P(F \mid  E) \triangleq \dfrac{P(E \cap F)}{P(E)} \]

  + two issues:
    + asymmetric: $P(E \mid  F)$
    + undefined if $P(E) = 0$
  + formal definition: (independent) $E$ and $F$ are <span style="color: magenta;">independent</span> if $P(E \cap F) = P(E) \cdot P(F)$, otherwise, <span style="color: magenta;">dependent</span>
  + symmetric and applied when $P(\varnothing) = 0$
  + $\implies$ to intuitive definition
    + symmetric: $P(F \mid  E) = P(F) \quad P(E \mid  F) = P(E)$
    + $P(F \mid  \overline{E}) = P(F) \quad P(E \mid  \overline{F}) = P(E)$

+ Non-surprising independence
  + two coins
    + $H_1$: first coin heads, $P(H_1) = 1/2$
    + $H_2$: second coin heads. $P(H_2) = 1/2$
    + $H_1 \cap H_2$: both coin heads, $P(H_1 \cap H_2) = 1/4$
    + $P(H_1 \cap H_2) = 1/4 = P(H_1) \cdot P(H_2) \to H_1 {\perp \!\!\! \perp} H_2$
  + not surprising
    + two separate coins
    + "independent" experiments always
  + surprising (?): can have ${\perp \!\!\!\! \perp}$ even for one experiment

+ Example: single die
  + three events

    <table style="font-family: arial,helvetica,sans-serif; width: 30vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Event</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Probability</th>
      </tr>
      </thead>
      <tbody>
      <tr> <td style="text-align: center;">Prime</td> <td style="text-align: center;">$\{2, 3, 5\}$</td> <td style="text-align: center;">1/2</td> </tr>
      <tr> <td style="text-align: center;">Odd</td> <td style="text-align: center;">$\{1, 3, 5\}$</td> <td style="text-align: center;">1/2</td> </tr>
      <tr> <td style="text-align: center;">Square</td> <td style="text-align: center;">$\{1, 4\}$</td> <td style="text-align: center;">1/3</td> </tr>
      </tbody>
    </table>

  + which pairs are ${\perp \!\!\!\! \perp}$ and ${\not\!\perp \!\!\!\! \perp}$

    <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:25%;">Intersection</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Prob</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">product</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">=?</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Independent</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td style="text-align: center;">Prime $\cap$ Odd</td> <td style="text-align: center;">$\{3, 5\}$</td>
        <td style="text-align: center;">1/3</td> <td style="text-align: center;">$1/2 \cdot 1/2 = 1/4$</td>
        <td style="text-align: center;">$\neq$</td> <td style="text-align: center;">dependent</td>
      </tr>
      <tr>
        <td style="text-align: center;">Prime $\cap$ Square</td> <td style="text-align: center;">$\varnothing$</td>
        <td style="text-align: center;">0</td> <td style="text-align: center;">$1/2 \cdot 1/3 = 1/6$</td>
        <td style="text-align: center;">$\neq$</td> <td style="text-align: center;">dependent</td>
      </tr>
      <tr>
        <td style="text-align: center;">Odd $\cap$ Square</td> <td style="text-align: center;">$\{1\}$</td>
        <td style="text-align: center;">1/6</td> <td style="text-align: center;">$1/2 \cdot 1/3 = 1/6$</td>
        <td style="text-align: center;">$=$</td> <td style="text-align: center;">independent</td>
      </tr>
      </tbody>
    </table>

+ Example: three coins
  + three events

    <table style="font-family: arial,helvetica,sans-serif; width: 40vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Event</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Description</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Probability</th>
      </tr>
      </thead>
      <tbody>
      <tr> <td style="text-align: center;">$H_1$</td> <td style="text-align: center;">1st coin heads</td> <td style="text-align: center;">{h**}</td> <td style="text-align: center;">1/2</td> </tr>
      <tr> <td style="text-align: center;">$H_2$</td> <td style="text-align: center;">2nd coin heads</td> <td style="text-align: center;">{*h*}</td> <td style="text-align: center;">1/2</td> </tr>
      <tr> <td style="text-align: center;">$HH$</td> <td style="text-align: center;">exactly 2 heads in a row</td> <td style="text-align: center;">{hht, thh}</td> <td style="text-align: center;">1/4</td> </tr>
      </tbody>
    </table>

  + which pairs are ${\perp \!\!\!\! \perp}$ and ${\not\!\perp \!\!\!\! \perp}$

    <table style="font-family: arial,helvetica,sans-serif; width: 50vw;" table-layout="auto" cellspacing="0" cellpadding="5" border="1" align="center">
      <thead>
      <tr style="font-size: 1.2em;">
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Intersection</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:15%;">Set</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">Prob</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:10%;">=?</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Product</th>
        <th style="text-align: center; background-color: #3d64ff; color: #ffffff; width:20%;">Independence</th>
      </tr>
      </thead>
      <tbody>
      <tr>
        <td style="text-align: center;">$H_1 \cap H_2$</td> <td style="text-align: center;">{hh*}</td>
        <td style="text-align: center;">1/4</td> <td style="text-align: center;">$=$</td>
        <td style="text-align: center;">$1/2 \cdot 1/2 = 1/4$</td> <td style="text-align: center;">independent</td>
      </tr>
      <tr>
        <td style="text-align: center;">$H_2 \cap HH$</td> <td style="text-align: center;">{hht, thh}</td>
        <td style="text-align: center;">1/4</td> <td style="text-align: center;">$\neq$</td>
        <td style="text-align: center;">$1/2 \cdot 1/4 = 1/8$</td> <td style="text-align: center;">dependent</td>
      </tr>
      <tr>
        <td style="text-align: center;">$H_1 \cap HH$</td> <td style="text-align: center;">{hht}</td>
        <td style="text-align: center;">1/8</td> <td style="text-align: center;">$=$</td>
        <td style="text-align: center;">$1/2 \cdot 1/4 = 1/8$</td> <td style="text-align: center;">independent</td>
      </tr>
      </tbody>
    </table>

+ Independence of $\Omega$ and $\varnothing$
  + $\Omega {\perp \!\!\!\! \perp}$ of any event
    + $\forall\,A \;P(\Omega \cap A) = P(A) = P(\Omega) \cdot P(A)$
    + $A$ occurring doesn't modify likelihood of $\Omega$
  + $\varnothing {\perp \!\!\!\! \perp}$ of any event
    + $\forall\,A \;P(\varnothing \cap A) = P(\varnothing) = P(\varnothing) \cdot P(A)$
    + $A$ occurring doesn't modify likelihood of $\varnothing$


+ [Original Slides](https://tinyurl.com/ybeanef9)


### Problem Sets

0. Two disjoint events cannot be independent.<br/>
  a. Yes<br/>
  b. Not exactly<br/>

  Ans: b<br/>
  Explanation: Not exactly. If the two disjoint events have positive probability, they are dependent. But if one of the two events has zero probability, they are independent.


1. Two dice are rolled. The event that the first die is 1 and the event that two dice sum up to be 7 are<br/>
  a. Independent<br/>
  b. Dependent<br/>

  Ans: a<br/>
  Explanation: Let $X$ be the outcome of the first die and $Y$ be the outcome of the second die. $P(X=1\mid X+Y=7)=1/6=P(X=1)$. Hence, they are independent.


2. Of 10 students, 4 take only history, 3 take only math, and 3 take both history and math. If you select a student at random, the event that the student takes history and the event that the student takes math are:<br/>
  a. Independent<br/>
  b. Dependent<br/>

  Ans: <span style="color: magenta;">b</span><br/>
  Explanation: Let $H$ be the event that the student takes history, and $M$ the event that the student takes math. Then $P(H)=7/10$, $P(M)=6/10$, and $P(H,M)=3/10$. Since $P(H)P(M) \neq P(H,M)$, the two events are dependent.


3. 4 freshman boys, 6 freshman girls, and 6 sophomore boys go on a trip. How many sophomore girls must join them if a student's gender and class are to be independent when a student is selected at random?

  Ans: 9 <br/>
  Explanation: 
    + First, let's do it the formal but hard way. Let $SG$ denote the number of sophomore girls. Then the total number of students is $4+6+6+SG=16+SG$.  If a student is selected at random, the probability that the student is a freshman is $4+616+SG$, the probability that a random student is a boy is $\frac{4+6}{16+SG}$, and the probability that the student is both a freshman and boy is $\frac{4}{16+SG}$. If the student's gender and class are independent, then by the product rule, the probability of the intersection is the product of the probabilities, hence $\frac{4}{16+SG}=\frac{4+6}{16+SG} \cdot \frac{4+6}{16+SG}$, hence $100=4 \cdot (16+SG)$, or $SG=9$.
    + Another way to see this is to observe that if the gender and class are independent, then the fraction of girls that are freshmen, namely $\frac{6}{6+SG}$ should be the same as the fraction of boys that are freshmen, namely $\frac{4}{4+6}=2/5$. Therefore $\frac{6}{6+SG}=2/5$, or $SG=9$. 


4. Every event $A$ is independent of:<br/>
  a. $\varnothing$,<br/>
  b. $\Omega$,<br/>
  c. $A$ itself,<br/>
  d. $A^c$.<br/>

  Ans: ab<br/>
  Explanation: Intuitively: $A$ is independent of the null event because occurrence of $A$ doesn't change the 0 probability of the null event. Similarly $A$ is independent of $\Omega$ because occurrence of $A$ does not change the probability 1 of $\Omega$.  If $A$ has probability strictly between 0 and 1, then its occurrence changes the probability of both itself and $A^c$, implying dependence. Mathematically:
    + True. $P(\varnothing\mid A)=0=P(\varnothing)$.
    + True. $P(A\mid \Omega)=P(A \cap \Omega)P(\Omega)=P(A)P(\Omega)=P(A)$.
    + False.
    + False.

5. Which of the following ensure that events $A$ and $B$ are independent:<br/>
  a. $A$ and $B^c$ are independent,<br/>
  b. $A \cap B= \varnothing$,<br/>
  c. $A \subseteq B$,<br/>
  d. at least one of $A$ or $B$ is $\varnothing$ or $\Omega$?<br/>

  Ans: ad<br/>
  Explanation
    + True. If $A$ and $B^c$ are independent, $1−P(B\mid A)=P(B^c\mid A)=P(B^c)=1−P(B)$, which implies $P(B\mid A)=P(B)$.
    + False.
    + False.
    + True. For $\varnothing$, $P(\varnothing\mid A)=0=P(\varnothing)$. For $\Omega$, $P(A\mid \Omega)=\frac{P(A \cap\Omega)}{P(\Omega)}=\frac{P(A)}{P(\Omega)}=P(A)$. $\varnothing$ and $\Omega$ are independent with any sets.

6. When rolling two dice, which of the following events are independent of the event that the first die is 4:<br/>
  a. the second is 2,<br/>
  b. the sum is 6,<br/>
  c. the sum is 7,<br/>
  d. the sum is even.<br/>

  Ans: acd<br/>
  Explanation: Let  X  be the outcome of the first dice, and  Y  be the second one.
    + True.  $P(X=4\mid Y=2)=P(X=4)=1/6$.
    + False. $P(X+Y=6\mid X=4)=1/6 .  P(X+Y=6)=5/36 .  P(X+Y=6) \neq P(X+Y=6\mid Y=4)$.
    + True.  $P(X+Y=6\mid X=4)=1/6=P(X+Y=7)$.
    + True.  $P(X+Y \text{ is even } \mid X=4)=P(Y \text{ is even })=1/2=P(X+Y \text{ is even })$.


7. Roll two dice, and let $F_e$ be the event that the first die is even, $S_4$ the event that the second die is 4, and $\Sigma_o$ the event that the sum of the two dice is odd. Which of the following events are independent:<br/>
  a. $F_e$ and $S_4$,<br/>
  b. $F_e$ and $\Sigma_o$,<br/>
  c. $S_4$ and $\Sigma_o$,<br/>
  d. $F_e$, $S_4$, and $\Sigma_o$ (mutually independent)?<br/>
  
  Ans: abc<br/>
  Explanation
    + True.  $P(F_e,S_4)=1/12, P(F_e)=1/2, P(S4)=1/6$. As $P(F_e,S_4)= P(F_e)P(S_4)$, $F_e$ and $S_4$ are independent.
    + True.  $P(F_e,\Sigma_o)=1/4, P(F_e)=1/2, P(\Sigma_o)=1/2$. As $P(F_e,\Sigma_o) = P(F_e) P(\Sigma_o)$, $F_e$ and $\Sigma_o$ are independent.
    + True.  $P(S_4,\Sigma_o)=1/12, P(S4)=1/6, P(\Sigma_o)=1/2$. As $P(S_4,\Sigma_o)= P(S_4) P(\Sigma_o)$, $S_4$ and $\Sigma_o$ are independent.
    + False. $P(F_e,S_4,\Sigma_o)=0 \neq P(F_e) P(S_4) P(\Sigma_o)$.


8. Two dice are rolled. Let $F_3$ be the event that the first die is 3, $S_4$ the event that the second die is 4, and $\Sigma_7$ the event that the sum is 7. Which of the following are independent:<br/>
  a. $F_3$ and $S_4$,<br/>
  b. $F_3$ and $\Sigma_7$,<br/>
  c. $S_4$ and $\Sigma_7$,<br/>
  d. $F_3$, $S_4$, and $\Sigma_7$ (mutually independent)?<br/>

  Ans: abc<br/>
  Explanation
    + True.  $P(F_3,S_4)=1/36, P(F_3)=1/6, P(S_4)=1/6$. As $P(F_3,S_4)=P(F_3)P(S_4)$, $F_3$ and $S_4$ are independent.
    + True.  $P(F_3,\Sigma_7)=1/36, P(F_3)=1/6, P(\Sigma_7)=1/6$. As $P(F_3,\Sigma_7)=P(F_3)P(\Sigma_7)$, $F_3$ and $\Sigma_7$ are independent.
    + True.  $P(S_4,\Sigma_7)=1/36, P(S_4)=1/6, P(\Sigma_7)=1/6$. As $P(S_4,\Sigma_7)=P(S_4)P(\Sigma_7)$, $S_4$ and $\Sigma_7$ are independent.
    + False. $P(F_3,S_4,\Sigma_7)=1/36 \neq P(F_3) P(S_4) P(\Sigma_7)=\frac{1}{6}\frac{1}{6}\frac{1}{6}=1/216$.


### Lecture Video

<a href="https://tinyurl.com/y82dpv98" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.3 Sequential Probability

+ Product / chain rule
  + conditional probability

    \[ P(F \mid  E) = \frac{P(E \cap F)}{P(E)} \to P(E \cap F) = P(E) \cdot P(F \mid  E) \]

  + helping to calculate regular (not conditional) probabilities

+ Sequential selections
  + urn: 1 blue, 2 red balls
  + draw 2 balls w/o replacement
  + task: $P(\text{ both red }) = ?$
  + Solution:
    + $R_i$: $i$-th ball is red
    + $P(\text{both red}) = P(R_1, R_2) = P(R_1) \cdot P(R_2 \mid  R_1) = \frac{2}{3} \cdot \frac{1}{2} = \frac{1}{3}$

+ General product rule

  \[\begin{align*}
    P(E \cap F \cap G) &=  P((E \cap F) \cap G) = P(E \cap F) \cdot P(G \mid  E \cap F) \\
    &= P(E) \cdot P(D \mid  E) \cdot P(G \mid  E \cap F)
  \end{align*}\]

  + example: odd ball
    + $n-1$ red balls and one blue ball
    + pick $n$ balls w/o replacement
    + $P(\text{last ball is blue}) = ?$
      + $R_i$: $i$-th ball is red
      + $R^i = R_1, R_2, \cdots, R_i$

        \[\begin{align*}
          P(\text{last ball blue}) &= P(R_1)P(R_2 \mid  R_1)P(R_3 \mid  R^2) \cdots P(R_{n-1} \mid  R^{n-2}) \\
          &= \frac{n-1}{n}\frac{n-2}{n-1}\frac{n-3}{n-2} \cdots \frac{2}{3}\frac{1}{2} = \frac{1}{n}
        \end{align*}\]

    + alternatively, arrange in row, probability last ball is blue = 1/n

+ The birthday paradox
  + how many people does it take so that two will likely share a birthday?
    + assume that every year has 365 days
    + everyone is equally likely to be born on any given day
  + probabilistically
    + choose $n$ random integers, each $\in \{1, 2, \dots, 365\}$, w/ replacement
    + $B(n)$: probability that two (or more) are the same
    + for which $n$ does $B(n)$ exceed, say , 1/2?
  + some first think it $n \approx 365$, but in fact much smaller
  + first attempt
    + consider the $n$ people in order, day alphabetically
    + e.g., list their birthdays: 2, 10, 265, 180, 10, ...
    + selection w/ replacement
    + set of all possible birthdays sequences: $\Omega = \{1, 2, \dots, 365\}^n \to |\Omega| = 365^n$
    + individual birthday uniform $\to \Omega$ uniform
    + $B_n = \{\text{sequences w/ repetition}\}$
    + $P(\text{ repetition }) = |B_n| / |\Omega| \to$ evaluating $|B_n|$ involved
  + complement
    + $B_n$: $n$ people w/ birthday repetition
    + $B_n^c$: $n$ people, no two share a birthday
    + evaluating sequentially
    + person $i$ different birthday from all previous
  + calculation
    + among $n$ people

      \[\begin{align*}
        P\left(\substack{\text{ no two people}\\ \text{share a birthday}}\right) &= \frac{364}{365} \cdot \frac{363}{365} \cdots \frac{365-n+1}{365} = \prod_{i=1}^{n-1} \left(1 - \frac{i}{365}\right) \quad (1-x \leq e^{-x})\\
        &\leq \prod_{i=1}^{n-1} e^{-\frac{i}{365}} = \exp\left( -\frac{1}{365} \cdot \sum_{i=1}^{n-1} i \right) = \exp\left( -\frac{n(n-1)}{2 \cdot 365} \right) \\
        &\approx \exp\left( -\frac{n^2}{2 \cdot 365} \right) = 0.5
      \end{align*}\]

    + when the probability is 0.5

      \[ -\frac{n^2}{2 \cdot 365} = \ln(0.5) = -\ln(2) \to n \approx \sqrt{-2 \cdot 365 \cdot \ln(0.5)} = 22.494 \]

      <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
        <a href="https://tinyurl.com/ycsx4c8w" ismap target="_blank">
          <img src="img/t06-02.png" style="margin: 0.1em;" alt="Number of people and probability of a pair" title="Number of people and probability of a pair" width=300>
        </a>
      </div>


+ [Original Slides](https://tinyurl.com/ycsx4c8w)


### Problem Sets

0. The equality $P(A \cap B)=P(A)P(B)$ holds whenever the events A and B are<br/>
  a. independent<br/>
  b. disjoint<br/>
  c. intersecting<br/>

  Ans: a<br/>
  Explanation: Independent. In fact, that's the definition of independence.


1. An urn contains $b$ black balls and $w$ white balls. Sequentially remove a random ball from the urn, till none is left.

  Which of the following observed color sequences would you think is more likely: first all white balls then all black ones (e.g. wwbbb), or alternating white (first) and black, till one color is exhausted, then the other color till it is exhausted (e.g. wbwbb)?

  For $b=4$ and $w=2$, calculate the probability of:

  a. white, white, black black, black black,<br/>
  b. white, black, white, black, black, black,<br/>
  
  Try to understand the observed outcome.<br/>

  Ans: a. (1/15); b. (1/15)<br/>
  Explanation:
    + By sequential probability, it is easy to see that the for any order of the colors, the denominator will be $(b+w)!$ while the numerator will be $b! \cdot w!$.
    + This can also be seen by symmetry. Imagine that the balls are colored from 1 to  b+w . Then each of the ( b+w)!  permutations of the balls is equally likely to be observed, hence will happen with probability $1/(b+w)!$, and $b! \cdot w!$ of them will correspond to each specified order of the colors.


2. An urn contains one red and one black ball. Each time, a ball is drawn independently at random from the urn, and then returned to the urn along with another ball of the same color. For example, if the first ball drawn is red, the urn will subsequently contain two red balls and one black ball.<br/>
  a. What is the probability of observing the sequence r,b,b,r,r?<br/>
  b. What is the probability of observing 3 red and 2 black balls?<br/>
  c. What is the probability of observing 7 red and 9 black balls?<br/>

  Ans: a. (1/60); b. (1/6); c. (1/17)<br/>
  Hint: (Part b) Note that any sequence with 3 red and 2 black balls, e.g. r,r,r,b,b is observed with the same probability.
  <span style="color: magenta;">Explanation</br>
    + $P(r,b,b,r,r)$ $=P(r) \cdot P(b\mid r) \cdot P(b\mid r,b) \cdot P(r\mid r,b,b) \cdot P(r\mid r,b,b,r)$ $=1/2 \cdot 1/3 \cdot2/4 \cdot 2/5 \cdot 3/6$ $=1/60=0.01667$
    + any sequence with 3 red and 2 black balls, e.g. r,r,r,b,b is observed with the same probability.
    + It can be verified that for any sequence with $n_r$ red balls and $n_b$ black balls, the probability $p=n_r! \cdot n_b!/(n_r+n_b+1)!$.  Hence the probability of observing $n_r$ red balls and $n_b$ black balls is <span style="color: cyan;">$\tfrac{n_r! \cdot n_b!}{(n_r+n_b+1)!} \tbinom{n_r+n_b}{n_b}=\frac{1}{n_r+n_b+1}$</span>.
    + [StackExchange](https://tinyurl.com/yc8x5e6j)


3. A box has seven tennis balls. Five are brand new, and the remaining two had been previously used. Two of the balls are randomly chosen, played with, and then returned to the box. Later, two balls are again randomly chosen from the seven and played with. What is the probability that all four balls picked were brand new.

  Ans: 10/147<br/>
  Explanation: The first two must be brand new, which happens with probability $\tbinom{5}{2}/\tbinom{7}{2}$. After that, $2+2=4$ balls had been played with, and  $5−2=3$ are still brand new. The probability that the next two chosen are brand new is $\tbinom{3}{2}/\tbinom{7}{2}$. By the chain rule, the probability that all four were brand new is $\tbinom{5}{2} \cdot \tbinom{3}{2}/\tbinom{7}{2}^2=10/147$.


4. A box contains six tennis balls. Peter picks two of the balls at random, plays with them, and returns them to the box. Next, Paul picks two balls at random from the box (they can be the same or different from Peter's balls), plays with them, and returns them to the box. Finally, Mary picks two balls at random and plays with them. What is the probability that each of the six balls in the box was played with exactly once?

  Ans: <span style="color: cyan;">2/75</span><br/>
  Explanation: The probability that every ball picked was played with exactly once is the probability that the 2 balls Paul picks differ from the 2 Peter picked, and that the 2 balls Mary picks differ from the 4 Peter or Paul picked. This probability is $\frac{\tbinom{6−2}{2}}{\tbinom{6}{2}} \cdot \frac{\tbinom{6−2−2}{2}}{\tbinom{6}{2}}=\frac{\tbinom{4}{2}}{\tbinom{6}{2}} \cdot \frac{\tbinom{2}{2}}{\tbinom{6}{2}}=6/15 \cdot 1/15=2/75$.


5. A bag contains 4 white and 3 blue balls. Remove a random ball and put it aside. Then remove another random ball from the bag. What is the probability that the second ball is white?<br/>
  a. 3/6<br/>
  b. 4/6<br/>
  c. 3/7<br/>
  d. 4/7<br/>

  Ans: d<br/>
  Hint: This problem can be solved using basic symmetry agruments, or using total probability discussed in the next section.<br/>
  Explanation: This can be done in two simple ways.
    + First, by symmetry. There are 4 white balls and 3 blue balls. The second ball picked is equally likely to be any of the 7 balls, hence the probability that it is white is 4/7.
    + Second, by total probability. The probability that the second ball is white is the probability that the first is white and the second is white namely $\frac47 \cdot \frac36$, plus the probability that the first is blue and the second is white, namely $\frac37 \cdot \frac46$, and $\frac47 \cdot \frac36+\frac37 \cdot \frac46=\frac47$. Note that the first, symmetry, argument is easier to extend to the third ball picked etc. But both derivation are of interest, and you may want to use the total-variation for a general case with W white balls and R red balls.
    + [Find probability of specific ball getting selected on second turn](https://tinyurl.com/yya2refb)


6. An urn contains  15  white and  20  black balls. The balls are withdrawn randomly, one at a time, until all remaining balls have the same color. Find the probability that:<br/>
  a. all remaining balls are white (if needed, see hints below),<br/>
  b. there are 5 remaining balls.<br/>

  Ans: a. (<span style="color: magenta;">3/7</span>); b. (<span style="color: magenta;">0.0300463</span>)<br/>
  Hint: If you drew all balls, with what probability would the last be white?<br/>
  Explanation:
    + [StackExchange](https://tinyurl.com/y7v3tury)

      <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
        <a href="https://tinyurl.com/y7v3tury" ismap target="_blank">
          <img src="https://i.stack.imgur.com/2JNnP.jpg" style="margin: 0.1em;" alt="Derivative of probability w/ remaining while balls" title="Derivative of probability w/ remaining while balls" width=450>
        </a>
        <a href="https://tinyurl.com/y7v3tury" ismap target="_blank">
          <img src="https://i.stack.imgur.com/KAla7.jpg" style="margin: 0.1em;" alt="Derivative of probability w/ remaining while balls" title="Derivative of probability w/ remaining while balls" width=450>
        </a>
        <a href="https://tinyurl.com/y7v3tury" ismap target="_blank">
          <img src="https://i.stack.imgur.com/c3YtL.jpg" style="margin: 0.1em;" alt="Derivative of probability w/ remaining while balls" title="Derivative of probability w/ remaining while balls" width=450>
        </a>
      </div>

    + Let $S$ be the sequence of balls you draw, for example $S$ could be $BWWBW \cdots$, with $B$ being a black ball, $W$ being a white ball. $S$ is of length $35$.  The remaining colors will be all white iff the last ball of $S$ is white, which happens with probability $15/35$.
    + Since there are five balls left, the last five balls need to be of the same color, and the ball just before them, of a different color. That is, the last 6 positions of $S$ should be either $BWWWWW$ or $WBBBBB$. Hence, the answer is $(\tbinom{29}{10}+\tbinom{29}{15})/\tbinom{35}{15}=0.03$.


7. Eight equal-strength players, including Alice and Bob, are randomly split into  4  pairs, and each pair plays a game, resulting in four winners. Fnd the probability that:<br/>
  a. both Alice and Bob will be among the four winners,<br/>
  b. neither Alice and Bob will be among the four winners.<br/>

  Ans: a. (<span style="color: magenta;">3/14</span>); b. (3/14)<br/>
  Explanation:
    + Here are two ways of solving the problem. One using sequential probability, the other by symmetry.
      + Sequential Probability: <br/>For both Alice and Bob to win, they first need to be paired with other players (not with each other), and then win their games. The probability that they are paired with other players is the probability that Alice was not paired with Bob, namely $6/7$. The probability that both win their respective games is $(1/2)^2=1/4$.  Hence the probability that both win is $(6/7)/4=3/14$.
      + Symmetry.<br/> In the end, 4 of the 8 players will be declared winners. There are $\tbinom{8}{4}$ such 4-winner "quartets", all equally likely. The number of "quartets" that contain both Alice and Bob is $\tbinom{6}{2}$, corresponding to picking two out of the remaining 6 players. Hence the probability that this occurs is $\frac{\tbinom{6}{2}}{\tbinom{8}{4}}=3/14$.
    + This is the same as the first part, except they need to lose rather than win both games, and using either sequential probability $(6/7)/4=3/14$. or symmetry  $\tbinom{6}{4}/\tbinom{8}{4}=3/14$, the probability is as before.

  

### Lecture Video

<a href="https://tinyurl.com/y9r7c6nc" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.4 Total Probability

+ Divide and conquer
  + when evaluating probability of an event
  + sometimes easier to split event into different parts
  + calculating probability of each part
  + add probabilities

+ Law of total probability
  + $\exists\, E, F$ events, $P(F) = ?$
  + $F = (E \cap F) \cup (E^c \cap F)$ s.t.

    \[\begin{align*}
      P(F) &= P()E \cap F) + P(E^c \cap F)  \quad (\text{Product rule}) \\
      &= P(E) \cdot P(F \mid  E) + P(E^c) \cdot P(F \mid  E^c)
    \end{align*}\]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yd7pzjew" ismap target="_blank">
      <img src="img/t06-03.png" style="margin: 0.1em;" alt="Example of total probability" title="Example of total probability" width=250>
    </a>
  </div>

  + example: 2 fair coins
    + $H_i$: coin $i$ w/ h
    + $\exists H$: at least one h, e.g., {hh, ht, th, tt} $P(\exists H) = \frac{|\exists H|}{|\Omega|} = \frac{3}{4}$
    + $P(\exists H) = ?$

      \[\begin{align*}
        P(\exists H) &= P(H_1 \cap \exists H) + P(H_1^c \cap \exists H) \\
        &= P(H_1) \cdot P(\exists H \mid  H_1) + P(H_1^c) \cdot P(\exists H \mid  H_1^c)\\
        &= 1/2 \cdot 1 + 1/2 \cdot 1/2 = 3/4
      \end{align*}\]

+ Total probability - n conditions
  + Let $E_1, E_2, \dots, E_n$ partition $\Omega$
  + $F = \displaystyle \biguplus_{i=1}^n (E_i \cap F)$ s.t.

    \[ P(F) = \sum_{i=1}^n P(E_i \cap F) = \sum_{i=1}^n P(E_i) \cdot P(F \mid  E_i) \]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yd7pzjew" ismap target="_blank">
        <img src="img/t06-04.png" style="margin: 0.1em;" alt="Example of total probability w/ n conditions" title="Example of total probability w/ n conditions" width=200>
      </a>
    </div>
  
  + example: 3 dice
    + $D_i$: outcome of die $i$
    + $S = D_1 + D_2$: sum of 2 dice
    + $P(S = 5) = ?$

      \[\begin{align*}
        P(S=5) &= \sum_{i=1}^4 P(D_1 = i) \cdot P(D_2 = 5 - i \mid  D_1 = i) \\
        &= \sum_{i=1}^4 P(D_1 = i) \cdot P(D_2 = 5-i) \\
        &= 4 \cdot \tfrac{1}{36} = 1/9
      \end{align*}\]

+ Example: iPhone X
  + three factories produce 50%, 30%, and 20% of iPhones
  + their defective rates are 4%, 10%, and 5% respectively
  + what is the overall fraction of defective iPhones?

    \[\begin{align*}
      P(D) &= P(F_1 \cap D) + P(F_2 \cap D) + P(F_2 \cap D) \\
      &= P(F_1) P(D \mid  F_1) + P(F_2) P(D \mid  F_2) + P(F_3) P(D \mid  F_3) \\
      & = .5 \times .04 + .3 \times .1 + .2 \times .05 = .02 + .03 + .01  = 0.06
    \end{align*}\]



+ [Original Slides](https://tinyurl.com/yd7pzjew)


### Problem Sets

0. 60% of our students are American (born), and 40% are foreign (born). 20% of the Americans and 40% of the foreigners speak two languages. What is the probability that a random student speaks two languages?<br/>
  a. 0.18<br/>
  b. 0.28<br/>
  c. 0.34<br/>
  d. 0.45<br/>

  Ans: b <br/>
  Explanation: The probability is 0.6 * 0.2 + 0.4 * 0.4 = 0.28.


1. Three 100-marble bags are placed on a table. One bag has 60 red and 40 blue marbles, one as 75 red and 25 blue marbles, and one has 45 red and 55 blue marbles.

  You select one bag at random and then choose a marble at random. What is the probability that the marble is red?<br/>
  a. 0.2025 <br/>
  b. 0.33 <br/>
  c. 0.50 <br/>
  d. 0.60<br/>

  Ans: d<br/>
  Explanation: The probability is $\tfrac{1}{3}(0.6+0.75+0.45)=0.60$.


2. Each of Alice and Bob has an identical bag containing 6 balls numbered 1, 2, 3, 4, 5, and 6. Alice randomly selects one ball from her bag and places it in Bob’s bag, then Bob randomly select one ball from his bag and places it in Alice’s bag. What is the probability that after this process the content in two bags remain unchanged?

  Ans: 2/7<br/>
  Explanation: The two bags will remain unchanged if the ball Bob picks has the same number as the one Alice placed there. Once Alice puts a ball numbered $n$ in Bob's bag, the probability the Bob picks a ball numbered $n$ is $2/7$. The total probability is $\sum_{n=1}^n 1/6 \cdot 2/7=2/7$.


3. Let $A$ and $B$ be two random subsets of $\{1,2,3,4\}$. What is the probability that $A \subseteq B$?

  Ans: $81/16^2$<br/>
  Explanation: 
    + This can be calculated via total probability, conditioning on the possible sizes of $A$ from 0 to 4, namely $(\tbinom{4}{0}2^0+\tbinom{4}{1}2^1+ \cdots+\tbinom{4}{4}2^4 )/(2^4)^2=0.3164$.  A cleaner derivation is to observe that $A \subseteq  B \iff$ for every element $x$, $x \in A \implies x \in B$, namely $x \notin A$ and $x \notin B$, or $x \notin A$ and $x \in B$, or $x \in A$ and $x \in B$. (Just $x \in A$ and $x \notin B$ is excluded.) Since any given $x$ is in $A$ and $B$ independently at random, the probability that this holds for $x$ is $3/4$, and the probability that it holds for all four elements is $(3/4)^4=0.3164$.
    + [StackExchange](https://tinyurl.com/yxdy3ym8)


4. Eight equal-strength players, including Alice and Bob, are randomly split into $4$ pairs, and each pair plays a game (i.e. 4 games in total), resulting in four winners. What is the probability that exactly one of Alice and Bob will be among the four winners?

  Ans: 4/7<br/>
  Explanation: Here are two ways of solving the problem. One using total probability, the other by symmetry.
    + Total Probability.<br/>Let $E$ be the desired event that exactly one of Alice or Bob is a winner. We divide the sample space into two disjoint events, $E_1$, $E_2$. $E_1$ is the event that Alice and Bob play against each other and $E_2$ is the complimentary event that Alice and Bob play against other players. Since Alice is equally likely to play with any of the seven other players, $P(E_1)=1/7$, hence $P(E_2)=6/7$. Now $P(E\mid E_1)=1$, while $P(E\mid E_2)=1/2$ since Alice and Bob each play an independent game where the probability of winning is $1/2$. Therefore $P(E)=P(E_1) \cdot P(E\mid E_1)+P(E_2) \cdot P(E\mid E_2)=1/7 \cdot 1+6/7 \cdot 1/2=4/7$.
    + Symmetry.<br/>In the end, 4 of the 8 players will be declared winners. There are $\tbinom{8}{4}$ such 4-winner "quartets", all equally likely.  The number of "quartets" that contain exactly one of Alice and Bob is $\tbinom{2}{1} \cdot \tbinom{6}{3}$.  Hence the probability that this occurs is $\frac{\tbinom{2}{1} \cdot \tbinom{6}{3}}{\tbinom{8}{4}}=\frac 4 7$.
    + [Quora - Jacob Miller](https://tinyurl.com/y5t329ze)
    + [Brainly](https://brainly.com/question/15019135)



### Lecture Video

<a href="https://tinyurl.com/y8aeo2qc" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.5 Bayes' Rule

+ Asymmetry
  + bough Google at IPO $\to$ wealthy $\gets ?$
  + alive today $\to$ born after 1800 $\gets ?$

+ Forward - backward
  + at times: $P(F \mid  E)$ - easy $\quad P(E \mid  F)$ - hard
  + example:
    + 2 coins
      + $H_i$: coin $i$ is h; $\exists H$: at least one h
      + $P(\exists H \mid  H_1) = 1$
      + $P(H_1 \mid  \exists H) ?$
    + 2 dice
      + $D_i$: face of die $i$; $S = D_1 + D_2$: sum of 2 faces
      + $P(S=5 \mid  D_1 = 2) = P(D_2 = 3) = 1/6$
      + $P(D_1 = 2 \mid  S = 5) ?$
  + Bayes' rule: method for converting $P(F \mid  E)$ to $P(E \mid  F)$

+ Bayes' rule
  + given $P(F \mid  E)$ (and a bit more, e.g., $P(E), P(F)$) determining $P(E \mid  F)$

    \[ P(E \mid  F) = \frac{P(E) \cdot P(F \mid  E)}{P(F)} \]

  + $\mu$-proof

    \[ P(E \mid  F) = \frac{P(E \cap F)}{P(F)} = \frac{P(E) \cdot P(F \mid  E)}{P(F)} \]

  + another view

    \[ P(F) \cdot P(E \mid  F) = P(E \cap F) = P(E) \cdot P(F \mid  E) \]

  + proof

    \[\begin{align*}
      P(F \mid  E) = \frac{\beta}{\alpha + \beta} \quad & \quad P(E \mid  F) = \frac{\beta}{\beta+\gamma} \\\\
      P(E \mid  F) = \frac{\beta}{\beta + \gamma} = \frac{\beta}{\alpha + \beta} \cdot& \frac{\alpha + \beta}{\beta + \gamma} = \frac{P(F \mid  E) \cdot P(E)}{P(F)}
    \end{align*}\]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="url" ismap target="_blank">
        <img src="img/t06-05.png" style="margin: 0.1em;" alt="Geometric view of conditonal probability and sets" title="Geometric view of conditonal probability and sets" width=150>
      </a>
    </div>


+ Example: two fair coins
  + $H_i$: coin $i$ h; $\exists H$: at least one h
  + set interpretation: $H_1 = \{hh, ht\}, \quad \exists H = \{hh, ht, th\}$

    \[ P(H_1 \mid  \exists H) = \frac{|H_1 \cap \exists H|}{|\exists H|} \]

  + probability

    \[ P(H_1 \mid  \exists H) = P(\exists H \mid  H_1) \cdot \frac{P(H_1)}{P(\exists H)} = 1 \cdot \frac{1/2}{3/4} = 2/3 \]

    where

    \[ P(\exists H \mid  H_1) = 1 \qquad P(H_1) = 1/2 \qquad P(\exists H) = 3/4 \]

+ Example: two fair dice
  + $D_i$: outcome of die $i$
  + $S = D_1 + D_2$: sum of 2 dice
  + $P(D_1 = 2 \mid  S = 5)?$
  + probability

    \[ P(D_1 = 2 \mid  S = 5) = \frac{P(S=5 \mid  D_1=2) \cdot P(D_1 = 2)}{P(S=5)} = \frac{\tfrac{1}{6} \cdot \tfrac{1}{6}}{1/9} = \frac{1}{4} \]

    where

    \[ P(S=5 \mid  D_1 = 2) = P(D_2=3 \mid  D_1=2) = P(D_2=3) = \frac{1}{6} \\ P(D_1 = 2) = \tfrac{1}{6} \quad P(S=5) = \tfrac{1}{9} \]

  + set interpretation
    + $S = 5 = \{(1, 4), (2, 3), (3, 2), (4, 1)\}$
    + $D_1 = 2 =\{(2, 3)\}$

    \[ P(D_1 = 2 \mid  S = 5) = \frac{|D_1 = 2 \cap S = 5|}{|S = 5|} \]

+ Example: Foxconn
  + Foxconn has 3 factories producing 50%, 30%, and 20% of its iPhones
  + Factory defective fractions 4%, 10%, and 5% respectively
  + Overall fraction of defective phones?

    \[\begin{align*}
      P(D) &= P(F_1 \cap D) + P(F_2 \cap D) + P(F_2 \cap D) \\
      &= P(F_1) P(D \mid  F_1) + P(F_2) P(D \mid  F_2) + P(F_3) P(D \mid  F_3) \\
      & = .5 \times .04 + .3 \times .1 + .2 \times .05 = .02 + .03 + .01  = 0.06
    \end{align*}\]

  + probability of factories w/ a given defective

    \[
      P(F_1 \mid  D) = \frac{P(D \mid  F_1) \cdot P(F_1)}{P(D)} = \frac{.04 \cdot .5}{.06} = \frac{.02}{.06} = \frac{1}{3} \\
      P(D \mid  F_1) = .04 \qquad P(F_1) = .5 \qquad P(D) = .06 \\
      P(F_2 \mid  D) = \frac{.1 \cdot .3}{.06} = \frac{.03}{.06} = \frac{1}{2} \qquad P(F_3 \mid  D) = \frac{.05 \cdot .2}{.06} = \frac{.01}{.06} = \frac{1}{6}
    \]

  + conditional probabilities add to 1
  + conditional order determined by both $P(F_i)$ and $P(D \mid  F_i)$


+ [Original Slides](https://tinyurl.com/y77gn7y3)


### Problem Sets

0. Suppose you're on a game show, and you're given the choice of three doors. Behind one door is a car and behind the others are goats. You pick a door, say door 1. The host knows what is behind each door. He opens another door, say door 3, which has a goat. He then says to you, "Do you want to change your selection to door 2?" Is it to your advantage to switch your choice?<br/>
  a. It is better to keep my choice of door 1.<br/>
  b. It is better to switch to door 2.<br/>
  c. There is no difference.<br/>

  Ans: b<br/>
  Explanation: It is better to switch.


1. A rare disease occurs randomly in one out of 10,000 people, and a test for the disease is accurate 99% of the time, both for those who have and don't have the disease. You take the test and the result is postive. The chances you actually have the disease are approximately:<br/>
  a. 10%<br/>
  b. 1%<br/>
  c. 0.1%<br/>
  d. 0.01%<br/>

  Ans: <span style="color: magenta;">b</span><br/>
  Explanation: Let $H$ and $D$  be the events that you Have and Don't have the disease, respectively, and let $S$ be the event that the result is positive.  By the streamlined version of Bayes' Rule, $P(H\mid S) = \frac{P(H,S)}{P(S)} = \frac{P(H,S)}{P(H,S) + P(D,S)}$.  Now, $P(H,S) =$ $P(H) \cdot P(S\mid H)=$ $0.0001 \cdot 0.99 \approx 0.0001$, and $P(D,S) = P(D) \cdot P(S\mid D) = 0.9999 \cdot 0.01≈0.01$.  Hence $P(H\mid S) = \frac{0.0001}{0.0001+0.01}≈0.01$.


2. A car manufacturer has three factories producing 21%, 35%, and 44% of its cars, respectively. Of these cars, 7%, 6%, and 2%, respectively, are defective. A car is chosen at random from the manufacturer’s supply.<br/>
  a. What is the probability that the car is defective?<br/>
  b. Given that the car is defective, what is the probability that was produced by the first factory?<br/>

  Ans: a. (0.0455); b. (0.33)<br/>
  Explanation: 
    + Let $F_1,F_2,F_3$ be the events that the care is made by the first, second, and third factory, respectively, and let  D  be the event that the car is defective. By the law of total probability, $P(D)=P(F_1) \cdot P(D\mid F_1)+P(F_2) \cdot P(D\mid F_2)+P(F_3) \cdot P(D\mid F_3)= $ $0.21 \cdot 0.07+0.35 \cdot 0.06+0.44 \cdot 0.02=0.0445$.
    + By Bayes' Rule and using $P(D)$ from above, $P(F_1\mid D)=P(F_1) \cdot P(D\mid F_1)P(D)=0.21 \cdot 0.070.0445=0.3303$.


3. A college graduate is applying for a job and has 3 interviews. She passes the first, second, and third interviews with probabilities 0.9, 0.8, and 0.7, respectively. If she fails any interview, she cannot proceed with subsequent interview(s) and will not get the job. If she didn’t get the job, what is the probability that she failed the second interview?

  Ans: <span style="color: magenta;">0.3629</span><br/>
  Explanation: Let $F$, $S$, and $T$ denote the events that the applicant passed the first, second, and third interviews, respectively. The probability that she failed the second interview given that she didn't get the job is  $P(\overline S\mid \overline{FST})$ $=P(F\overline S\mid \overline{FST})$ $=\frac{P(F\overline S\land\overline{FST})}{P(\overline{FST})}$ $=\frac{P(F\overline S)}{P(\overline{FST})}$ $=\frac{0.9\cdot 0.2}{1-0.9\cdot0.8\cdot0.7}$, where the first equality follows as the applicant fails the second interview iff she passes the first interview and fails the second. [EstackExchange](https://tinyurl.com/ya73scpb)


4. An ectopic pregnancy is twice as likely to develop when a pregnant woman is a smoker than when she is a nonsmoker. If 32% of women of childbearing age are smokers, what fraction of women having ectopic pregnancies are smokers?

  Ans: <span style="color: magenta;">0.4848</span><br/>
  Explanation: Let $S$ and $E$ denote the events that a pregnant woman is a smoker, and has an ectopic pregnancy, respectively. We are told that $P(S)=.32$ and $P(E\mid S^c)=.5 \cdot P(E\mid S)$. By Bayes' Rule, $P(S\mid E)$ $=\frac{P(E\mid S)\cdot P(S)}{P(E\mid S)\cdot P(S)+P(E\mid \overline S)\cdot P(\overline S)}$ $=\frac{P(E\mid S)\cdot P(S)}{P(E\mid S)\cdot P(S)$ $+.5\cdot P(E\mid S)\cdot P(\overline S)}$ $=\frac{P(S)}{P(S)$ $+.5\cdot P(\overline S)}$ $=\frac{.32}{.32+.5\cdot(1-.32)}$ $=\frac{16}{33}.$. [Waterloo](https://ece.uwaterloo.ca/~oamin/Tut3.pdf)


5. Each of Alice, Bob, and Chuck shoots at a target once, and hits it independently with probabilities 1/6, 1/4, and 1/3, respectively. If only one shot hit the target, what is the probability that Alice's shot hit the target?<br/>
  a. 31/72<br/>
  b. 6/31<br/>
  c. 10/31<br/>
  d. 15/31<br/>

  Ans: b<br/>
  Explanation: Let $A$, $B$, and $C$, be the events that Alice, Bob, and Chuck hit the target, respectively, and let $E=AB^cC^c \cap A^cBC^c \cup A^cB^cC$ be the event that only one shot hit the target. Then $P(E)=16 \cdot 34 \cdot 23+56 \cdot 14 \cdot 23+56 \cdot 34 \cdot 13=3172$. By Bayes' Rule, $P(A\mid E)=\frac{P(AE)}{P(E)}=\frac{P(AB^cC^c)}{P(E)}=\frac{6/72}{31/72}=6/31$.


6. Jack has two coins in his pocket, one fair, and one "rigged" with heads on both sides. Jack randomly picks one of the two coins, flips it, and observes heads. What is the probability that he picked the fair coin?<br/>
  a. 3/4<br/>
  b. 2/3<br/>
  c. 1/3<br/>
  d. 1/4<br/>

  Ans: c<br/>
  Explanation: Let $F$ and $R$ be the events that Jack picked the fair and rigged coin, respectively, and let  H  be the event that he observed heads. By the "streamlined" Bayes' rule, $P(F\mid H)=\frac{P(F,H)}{P(H)}=\frac{P(F,H)}{P(F,H)+P(R,H)}$. Now, $P(F,H)=P(F) \cdot P(H\mid F)=\frac12 \cdot \frac12=\frac14$, while $P(R,H)=P(R) \cdot P(H\mid R)=\frac12 \cdot 1=\frac12$. Hence $P(F\mid H)=1/41/4+1/2=13$.


7. It rains in Seattle one out of three days, and the weather forecast is correct two thirds of the time (for both sunny and rainy days). You take an umbrella if and only if rain is forecasted.<br/>
  a. What is the probability that you are caught in the rain without an umbrella?<br/>
  b. What is the probability that you carry an umbrella and it does not rain?<br/>

  Ans: a. (1/9); b. (2/9)<br/>
  Explanation: 
    + Let $R$ be the event that it rains, and $C$ the event that the forecast is correct. We are told that $P(R)=1/3$ and $P(C\mid R)=P(C\mid R^c)=2/3$. The probability you are caught in the rain without an umbrella is $P(R∧C^c)=P(R) \cdot P(C^c\mid R)=1/3 \cdot 1/3=1/9$.
    + The probability you are carry and umbrella and it does not rain is $P(R^c∧C^c)=P(R^c) \cdot P(C^c\mid R^c)=2/3 \cdot 1/3=2/9$.
    + [StackExchange 1](https://tinyurl.com/y8muf6ye), [StackExchange 2](https://tinyurl.com/y8bxt2ew)


8. On any night, there is a 92% chance that an burglary attempt will trigger the alarm, and a 1% chance of a false alarm, namely that the alarm will go off when there is no burglary. The chance that a house will be burglarized on a given night is 1/1000. What is the chance of a burglary attempt if you wake up at night to the sound of your alarm?

  Ans: <font style="color: cyan">0.084</font><br/>
  Explanation: Let $A$ be the event of triggering alarm, $B$ be the event of an burglary attempt, $NB$ be the event of no attempts. As $P(A\mid B)=0.92,P(A\mid NB)=0.01,P(B)=0.001$. Following Bayes rule $P(B\mid A) = \frac{P(A\mid B) P(B)}{P(A)} = \frac{P(A\mid B) P(B)}{P(A\mid B) P(B) + P(A\mid NB) P(NB)} = 0.084$.


9. An urn labeled "heads" has  5  white and  7  black balls, and an urn labeled "tails" has  3  white and  12  black balls. Flip a fair coin, and randomly select on ball from the "heads" or "tails" urn according to the coin outcome. Suppose a white ball is selected, what is the probability that the coin landed tails?

  Ans: 12/37<br/>
  Explanation: Let $H$ and $T$ be the events that the coin turnd up heads and tails, and let $W$ be the event of selecting a white ball. $P(H)=P(T)=0.5,P(W\mid H)=5/12,P(W\mid T)=3/15$. Following Bayesian rule $P(T\mid W)=12/37$. [Math UConn](https://tinyurl.com/yy69563z)


10. A car manufacturer receives its air conditioning units from 3 suppliers. 20% of the units come from supplier A, 30% from supplier B, and 50% from supplier C. 10% of the units from supplier A are defective, 8% of units from supplier B are defective, and 5% of units from supplier C are defective. If a unit is selected at random and is found to be defective.

  What is the probability that a unit came from supplier A if it is:<br/>
  a. defective,<br/>
  b. non-defective,<br/>

  Ans: a. (20/69); b. (0.1933)


11. Suppose that 15% of the population have cancer, 50% of the population smokes, and 75% of those with cancer smoke. What fraction of smokers have cancer?<br/>
  a. 0.05625 <br/>
  b. 0.225 <br/>
  c. 0.25 <br/>
  d. 0.75<br/>

  Ans: b<br/>
  Explanation: Let $S$ and $C$ be the events that a person smokes, and has cancer, respectively. Then $P(C\mid S)=P(S\mid C)P(C))P(S)=0.225$.


12. A fair coin with $P(heads)=0.5$ and a biased coin with $P(heads)=0.75$ are placed in an urn. One of the two coins is picked at random and tossed twice. Find the probability:<br/>
  a. of observing two heads,<br/>
  b. that the biased coin was picked if two heads are observed.<br/>

  Ans: a. (13/32); b. (9/13)<br/>
  Explanation:
    + Let $F,B$, and $T$, be the events that the coin is Fair, Biased, and we observe Two heads, respectively. By the law of total probability, $P(T)=P(F) \cdot P(T\mid F)+P(B) \cdot P(T\mid B)=1/2 \cdot 1/4+1/2 \cdot \tbinom{3}{4}^2=13/32$.
    + $P(B\mid T)=P(B,T)P(T)=P(B) \cdot P(T\mid B)P(T)=\frac{1/2 \cdot (3/4)^2}{13/32}=\frac{1/2 \cdot (3/4)^2}{13/32}=9/13$.



### Lecture Video

<a href="https://tinyurl.com/yday6dyk" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 6

+ [Original Lecture NB](https://tinyurl.com/yah8s5yr)

+ [Local Lecture NB](src/Topic06_Lecture.ipynb)

+ [Local Lecture Python Code](src/Topic06_Lecture.py)

+ [Original lecture NB - Pandas](https://tinyurl.com/y8mz9dl5)

+ [Local Lecture NB - Panadas](src/Topic06_Lecture_pandas.ipynb)

+ [Local Lecture Python Code](src/Topic06_Lecture_pandas.py)



## Programming Assignment 6

+ [Original HW NB](https://tinyurl.com/y7halvbv)

+ [Local HW NB](src/HW_Topic06.ipynb)

+ [Local Python code for HW](src/HW_Topic06.py)


1. Which of the following is the correct output for `conditional__probability( 2., 1., 6., 4.)`<br/>
  a. 0.3872<br/>
  b. 0.4545<br/>
  c. 0.5015<br/>
  d. 0.6666<br/>

  Ans: b<br/>
  Explanation: One of the urns is picked at random, hence $P(A)=P(B)=0.5$.  $P(A\mid white)=P(white\mid A)P(A)P(white)=P(white\mid A)P(A)P(white\mid A)P(A)+P(white\mid B)P(B)=\frac{\frac{w_A}{w_A+r_A}}{\frac{w_A}{w_A+r_A}+\frac{w_B}{w_B+r_B}}$

  ```python
  def conditional__probability(rA, wA, rB, wB):
    # inputs: all of them are of type 'float'
    # output: a variable of type 'float'

    return 1 / (1 + wB * (rA + wA) / wA / (rB + wB))
  ```







