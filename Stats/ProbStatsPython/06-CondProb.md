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
  c. $B=\Omega$,<br/>
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

+ Motivation
  + $\Pr(F \,|\, E) > \Pr(F)$
    + $E \nearrow$ probability of $F$
    + e.g., $\Pr(2 \,|\, \text{Even}) = 1/3 > 1/6 = \Pr(2)$
  + $\Pr(F  \,|\,  E) < \Pr(F)$
    + $E \searrow$ probability of $F$
    + $\Pr(2 \,|\, \text{Odd}) = 0 < 1/6 = \Pr(2)$
  + $\Pr(F \,|\, E) = \Pr(F)$
    + $E$ neither $\nearrow$ nor $\searrow$ probability of $F$
    + e.g., $\Pr(\text{Even} \,|\, \leq 4) = 1/2 = \Pr(\text{Even})$
    + whether or not $E$ occurs, does not change $|Pr(F)$
  + motivation $\to$ intuitive definition $|to$ formal

+ Independence - Intuitive
  + informal definition: (independence) Events $E$ and $F$ are <span style="color: magenta;">independent</span> (<span style="color: magenta;">$ E {\perp \!\!\!\! \perp} F$</span>) if occurrence of one does not change the probability  that the other occurs.
  + more formally, $\Pr(F \,|\, E) = \Pr(F)$
  + visual interpretation
    + $\Pr(F) = \frac{\Pr(F)}{\Pr(\Omega)}$: $F$ as a fraction of $\Omega$
    + $\Pr(F \,|\, E) \triangleq \frac{\Pr(E \cap F)}{\Pr(E)}$: $E \cap F$ as a fraction of $E$

      <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
        <a href="url" ismap target="_blank">
          <img src="img/t06-01.png" style="margin: 0.1em;" alt="Visual interpretation of P(F|E)" title="Visual interpretation of P(F|E)" width=200>
        </a>
      </div>
  
+ Independence - formal
  + informal

    \[ \Pr(F) = \Pr(F \,|\, E) \triangleq \dfrac{\Pr(E \cap F)}{\Pr(E)} \]

  + two issues:
    + asymmetric: $\Pr(E \,|\, F)$
    + undefined if $\Pr(E) = 0$
  + formal definition: (independent) $E$ and $F$ are <span style="color: magenta;">independent</span> if $\Pr(E \cap F) = \Pr(E) \cdot \Pr(F)$, otherwise, <span style="color: magenta;">dependent</span>
  + symmetric and applied when $\Pr(\varnothing) = 0$
  + $\implies$ to intuitive definition
    + symmetric: $\Pr(F \,|\, E) = \Pr(F) \quad \Pr(E \,|\, F) = \Pr(E)$
    + $\Pr(F \,|\, \overline{E}) = \Pr(F) \quad \Pr(E \,|\, \overline{F}) = \Pr(E)$

+ Non-surprising independence
  + two coins
    + $H_1$: first coin heads, $\Pr(H_1) = 1/2$
    + $H_2$: second coin heads. $\Pr(H_2) = 1/2$
    + $H_1 \cap H_2$: both coin heads, $\Pr(H_1 \cap H_2) = 1/4$
    + $\Pr(H_1 \cap H_2) = 1/4 = \Pr(H_1) \cdot \Pr(H_2) \to H_1 {\perp \!\!\! \perp} H_2$
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
    + $\forall\,A \;\Pr(\Omega \cap A) = \Pr(A) = \Pr(\Omega) \cdot \Pr(A)$
    + $A$ occurring doesn't modify likelihood of $\Omega$
  + $\varnothing {\perp \!\!\!\! \perp}$ of any event
    + $\forall\,A \;\Pr(\varnothing \cap A) = \Pr(\varnothing) = \Pr(\varnothing) \cdot \Pr(A)$
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
  Explanation: Let $X$ be the outcome of the first die and $Y$ be the outcome of the second die. $P(X=1|X+Y=7)=1/6=P(X=1)$. Hence, they are independent.


2. Of 10 students, 4 take only history, 3 take only math, and 3 take both history and math. If you select a student at random, the event that the student takes history and the event that the student takes math are:<br/>
  a. Independent<br/>
  b. Dependent<br/>

  Ans: <span style="color: magenta;">b</span><br/>
  Explanation: Let $H$ be the event that the student takes history, and $M$ the event that the student takes math. Then $P(H)=7/10$, $P(M)=6/10$, and $P(H,M)=3/10$. Since $P(H)P(M) \neq P(H,M)$, the two events are dependent.


3. 4 freshman boys, 6 freshman girls, and 6 sophomore boys go on a trip. How many sophomore girls must join them if a student's gender and class are to be independent when a student is selected at random?

  Ans:9 <br/>
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
    + True. $P(\varnothing|A)=0=P(\varnothing)$.
    + True. $P(A|\Omega)=P(A \cap \Omega)P(\Omega)=P(A)P(\Omega)=P(A)$.
    + False.
    + False.

5. Which of the following ensure that events $A$ and $B$ are independent:<br/>
  a. $A$ and $B^c$ are independent,<br/>
  b. $A \cap B= \varnothing$,<br/>
  c. $A \subseteq B$,<br/>
  d. at least one of $A$ or $B$ is $\varnothing$ or $\Omega$?<br/>

  Ans: ad<br/>
  Explanation
    + True. If $A$ and $B^c$ are independent, $1−\Pr(B|A)=\Pr(B^c|A)=\Pr(B^c)=1−\Pr(B)$, which implies $\Pr(B|A)=\Pr(B)$.
    + False.
    + False.
    + True. For $\varnothing$, $\Pr(\varnothing|A)=0=\Pr(\varnothing)$. For $\Omega$, $\Pr(A|\Omega)=\frac{\Pr(A \cap\Omega)}{\Pr(\Omega)}=\frac{\Pr(A)}{\Pr(\Omega)}=\Pr(A)$. $\varnothing$ and $\Omega$ are independent with any sets.

6. When rolling two dice, which of the following events are independent of the event that the first die is 4:<br/>
  a. the second is 2,<br/>
  b. the sum is 6,<br/>
  c. the sum is 7,<br/>
  d. the sum is even.<br/>

  Ans: acd<br/>
  Explanation: Let  X  be the outcome of the first dice, and  Y  be the second one.
    + True.  $P(X=4|Y=2)=P(X=4)=1/6$.
    + False. $P(X+Y=6|X=4)=1/6 .  P(X+Y=6)=5/36 .  P(X+Y=6) \neq P(X+Y=6|Y=4)$.
    + True.  $P(X+Y=6|X=4)=1/6=P(X+Y=7)$.
    + True.  $P(X+Y \text{ is even } |X=4)=P(Y \text{ is even })=1/2=P(X+Y \text{ is even })$.


7. Roll two dice, and let $F_e$ be the event that the first die is even, $S_4$ the event that the second die is 4, and $\Sigma_o$ the event that the sum of the two dice is odd. Which of the following events are independent:<br/>
  a. $F_e$ and $S_4$,<br/>
  b. $F_e$ and $\Sigma_o$,<br/>
  c. $S_4$ and $\Sigma_o$,<br/>
  d. $F_e$, $S_4$, and $\Sigma_o$ (mutually independent)$?<br/>
  
  Ans: abc<br/>
  Explanation
    + True.  $\Pr(F_e,S_4)=1/12, \Pr(F_e)=1/2, \Pr(S4)=1/6$. As $\Pr(F_e,S_4)= \Pr(F_e)\Pr(S_4)$, $F_e$ and $S_4$ are independent.
    + True.  $\Pr(F_e,\Sigma_o)=1/4, \Pr(F_e)=1/2, \Pr(\Sigma_o)=1/2$. As $\Pr(F_e,\Sigma_o) = \Pr(F_e) \Pr(\Sigma_o)$, $F_e$ and $\Sigma_o$ are independent.
    + True.  $\Pr(S_4,\Sigma_o)=1/12, \Pr(S4)=1/6, \Pr(\Sigma_o)=1/2$. As $\Pr(S_4,\Sigma_o)= \Pr(S_4) \Pr(\Sigma_o)$, $S_4$ and $\Sigma_o$ are independent.
    + False. $\Pr(F_e,S_4,\Sigma_o)=0 \neq P(F_e) \Pr(S_4) \Pr(\Sigma_o)$.


8. Two dice are rolled. Let $F_3$ be the event that the first die is 3, $S_4$ the event that the second die is 4, and $\Sigma_7$ the event that the sum is 7. Which of the following are independent:<br/>
  a. $F_3$ and $S_4$,<br/>
  b. $F_3$ and $\Sigma_7$,<br/>
  c. $S_4$ and $\Sigma_7$,<br/>
  d. $F_3$, $S_4$, and $\Sigma_7$ (mutually independent)?<br/>

  Ans: abc<br/>
  Explanation
    + True.  $\Pr(F_3,S_4)=1/36, \Pr(F_3)=1/6, \Pr(S_4)=1/6$. As $\Pr(F_3,S_4)=\Pr(F_3)\Pr(S_4)$, $F_3$ and $S_4$ are independent.
    + True.  $\Pr(F_3,\Sigma_7)=1/36, \Pr(F_3)=1/6, \Pr(\Sigma_7)=1/6$. As $\Pr(F_3,\Sigma_7)=\Pr(F_3)\Pr(\Sigma_7)$, $F_3$ and $\Sigma_7$ are independent.
    + True.  $\Pr(S_4,\Sigma_7)=1/36, \Pr(S_4)=1/6, \Pr(\Sigma_7)=1/6$. As $\Pr(S_4,\Sigma_7)=\Pr(S_4)\Pr(\Sigma_7)$, $S_4$ and $\Sigma_7$ are independent.
    + False. $\Pr(F_3,S_4,\Sigma_7)=1/36 \neq \Pr(F_3) \Pr(S_4) \Pr(\Sigma_7)=\frac{1}{6}\frac{1}{6}\frac{1}{6}=1/216$.


### Lecture Video

<a href="https://tinyurl.com/y82dpv98" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 6.3 Sequential Probability

+ Product / chain rule
  + conditional probability

    \[ \Pr(F \,|\, E) = \frac{\Pr(E \cap F)}{\Pr(E)} \to \Pr(E \cap F) = \Pr(E) \cdot \Pr(F | E) \]

  + helping to calculate regular (not conditional) probabilities

+ Sequential selections
  + urn: 1 blue, 2 red balls
  + draw 2 balls w/o replacement
  + task: $\Pr(\text{ both red }) = ?$
  + Solution:
    + $R_i$: $i$-th ball is red
    + $\Pr(\text{both red}) = \Pr(R_1, R_2) = \Pr(R_1) \cdot \Pr(R_2 \,|\, R_1) = \frac{2}{3} \cdot \frac{1}{2} = \frac{1}{3}$

+ General product rule
  
  \[\begin{align*}
    \Pr(E \cap F \cap G) &=  \Pr((E \cap F) \cap G) = \Pr(E \cap F) \cdot \Pr(G \,|\, E \cap F) \\
    &= \Pr(E) \cdot \Pr(D \,|\, E) \cdot \Pr(G \,|\, E \cap F)
  \end{align*}\]

  + example: odd ball
    + $n-1$ red balls and one blue ball
    + pick $n$ balls w/o replacement
    + $\Pr(\text{last ball is blue}) = ?$
      + $R_i$: $i$-th ball is red
      + $R^i = R_1, R_2, \cdots, R_i$

        \[\begin{align*}
          \Pr(\text{last ball blue}) &= \Pr(R_1)\Pr(R_2 \,|\, R_1)\Pr(R_3 \,|\, R^2) \cdots \Pr(R_{n-1} \,|\, R^{n-2}) \\
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
    + $\Pr(\text{ repetition }) = |B_n| / |\Omega| \to$ evaluating $|B_n|$ involved
  + complement
    + $B_n$: $n$ people w/ birthday repetition
    + $B_n^c$: $n$ people, no two share a birthday
    + evaluating sequentially
    + person $i$ different birthday from all previous
  + calculation
    + among $n$ people

      \[\begin{align*}
        \Pr\left(\substack{\text{ no two people}\\ \text{share a birthday}}\right) &= \frac{364}{365} \cdot \frac{363}{365} \cdots \frac{365-n+1}{365} = \prod_{i=1}^{n-1} \left(1 - \frac{i}{365}\right) \quad (1-x \leq e^{-x})\\
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
  <span style="color: magenta;">Explanation</br>
    + $\Pr(r,b,b,r,r)=\Pr(r) \cdot \Pr(b|r) \cdot \Pr(b|r,b) \cdot \Pr(r|r,b,b) \cdot \Pr(r|r,b,b,r)=1/2 \cdot 1/3 \cdot2/4 \cdot 2/5 \cdot 3/6=160=0.01666 \cdots$
    + any sequence with 3 red and 2 black balls, e.g. r,r,r,b,b is observed with the same probability.
    + t can be verified that for any sequence with $n_r$ red balls and $n_b$ black balls, the probability $p=n_r! \cdot n_b!/(n_r+n_b+1)!$.  Hence the probability of observing $n_r$ red balls and $n_b$ black balls is <span style="color: cyan;">$\tfrac{n_r! \cdot n_b!}{(n_r+n_b+1)!} \tbinom{n_r+n_b}{n_b}=\frac{1}{n_r+n_b+1}$</span>.
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
  Explanation: This can be done in two simple ways. First, by symmetry. There are 4 white balls and 3 blue balls. The second ball picked is equally likely to be any of the 7 balls, hence the probability that it is white is 4/7. Second, by total probability. The probability that the second ball is white is the probability that the first is white and the second is white namely $47 \cdot 36$, plus the probability that the first is blue and the second is white, namely $37 \cdot 46$, and $47 \cdot 36+37 \cdot 46=47$. Note that the first, symmetry, argument is easier to extend to the third ball picked etc. But both derivation are of interest, and you may want to use the total-variation for a general case with W white balls and R red balls.


6. An urn contains  15  white and  20  black balls. The balls are withdrawn randomly, one at a time, until all remaining balls have the same color. Find the probability that:<br/>
  a. all remaining balls are white (if needed, see hints below),<br/>
  b. there are 5 remaining balls.<br/>

  Ans: a. (<span style="color: magenta;">3/7</span>); b. (<span style="color: magenta;">0.0300463</span>)<br/>
  Explanation:
    + [StackExchange](https://tinyurl.com/y7v3tury)

      <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
        <a href="https://tinyurl.com/y7v3tury" ismap target="_blank">
          <img src="https://i.stack.imgur.com/2JNnP.jpg" style="margin: 0.1em;" alt="Derivative of probability w/ remaining while balls" title="Derivative of probability w/ remaining while balls" width=450>
        </a>
        <a href="https://tinyurl.com/y7v3tury" ismap target="_blank">
          <img src="https://i.stack.imgur.com/KAla7.jpg" style="margin: 0.1em;" alt="Derivative of probability w/ remaining while balls" title="Derivative of probability w/ remaining while balls" width=450>
        </a>
      </div>

    + Let $S$ be the sequence of balls you draw, for example $S$ could be $BWWBW \cdot$), with $B$ being a black ball, $W$ being a white ball. $S$ is of length $35$.  The remaining colors will be all white iff the last ball of $S$ is white, which happens with probability $15/35$.
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
  + $\exists\, E, F$ events, $\Pr(F) = ?$
  + $F = (E \cap F) \cup (E^c \cap F)$ s.t.

    \[\begin{align*}
      \Pr(F) &= \Pr()E \cap F) + \Pr(E^c \cap F)  \quad (\text{Product rule}) \\
      &= \Pr(E) \cdot \Pr(F \,|\, E) + \Pr(E^c) \cdot \Pr(F \,|\, E^c)
    \end{align*}\]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/yd7pzjew" ismap target="_blank">
      <img src="img/t06-03.png" style="margin: 0.1em;" alt="Example of total probability" title="Example of total probability" width=250>
    </a>
  </div>

  + example: 2 fair coins
    + $H_i$: coin $i$ w/ h
    + $\exists H$: at least one h, e.g., {hh, ht, th, tt} $\Pr(\exists H) = \frac{|\exists H|}{|\Omega|} = \frac{3}{4}$
    + $\Pr(\exists H) = ?$

      \[\begin{align*}
        \Pr(\exists H) &= \Pr(H_1 \cap \exists H) + \Pr(H_1^c \cap \exists H) \\
        &= \Pr(H_1) \cdot \Pr(\exists H \,|\, H_1) + \Pr(H_1^c) \cdot \Pr(\exists H \,|\, H_1^c)\\
        &= 1/2 \cdot 1 + 1/2 \cdot 1/2 = 3/4
      \end{align*}\]

+ Total probability - n conditions
  + Let $E_1, E_2, \dots, E_n$ partition $\Omega$
  + $F = \displaystyle \biguplus_{i=1}^n (E_i \cap F)$ s.t.

    \[ \Pr(F) = \sum_{i=1}^n \Pr(E_i \cap F) = \sum_{i=1}^n \Pr(E_i) \cdot \Pr(F \,|\, E_i) \]

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="https://tinyurl.com/yd7pzjew" ismap target="_blank">
        <img src="img/t06-04.png" style="margin: 0.1em;" alt="Example of total probability w/ n conditions" title="Example of total probability w/ n conditions" width=200>
      </a>
    </div>
  
  + example: 3 dice
    + $D_i$: outcome of die $i$
    + $S = D_1 + D_2$: sum of 2 dice
    + $\Pr(S = 5) = ?$

      \[\begin{align*}
        \Pr(S=5) &= \sum_{i=1}^4 \Pr(D_1 = i) \cdot \Pr(D_2 = 5 - i \,|\, D_1 = i) \\
        &= \sum_{i=1}^4 \Pr(D_1 = i) \cdot \Pr(D_2 = 5-i) \\
        &= 4 \cdot \tfrac{1}{36} = 1/9
      \end{align*}\]

+ Example: iPhone X
  + three factories produce 50%, 30%, and 20% of iPhones
  + their defective rates are 4%, 10%, and 5% respectively
  + what is the overall fraction of defective iPhones?

    \[\begin{align*}
      \Pr(D) &= \Pr(F_1 \cap D) + \Pr(F_2 \cap D) + \Pr(F_2 \cap D) \\
      &= \Pr(F_1) \Pr(D \,|\, F_1) + \Pr(F_2) \Pr(D \,|\, F_2) + \Pr(F_3) \Pr(D \,|\, F_3) \\
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
  Explanation: This can be calculated via total probability, conditioning on the possible sizes of $A$ from 0 to 4, namely $(\tbinom{4}{0}2^0+\tbinom{4}{1}2^1+ \cdots+\tbinom{4}{4}2^44)/(2^4)^2=0.3164$.  A cleaner derivation is to observe that $A \subseteq  B \iff$ for every element $x$, $x \in A \implies x \in B$, namely $x \notin A$ and $x \notin B$, or $x \notin A$ and $x \in B$, or $x \in A$ and $x \in B$. (Just $x \in A$ and $x \notin B$ is excluded.) Since any given $x$ is in $A$ and $B$ independently at random, the probability that this holds for $x$ is $3/4$, and the probability that it holds for all four elements is $(3/4)^4=0.3164$.


4. Eight equal-strength players, including Alice and Bob, are randomly split into $4$ pairs, and each pair plays a game (i.e. 4 games in total), resulting in four winners. What is the probability that exactly one of Alice and Bob will be among the four winners?

  Ans: 4/7<br/>
  Explanation: Here are two ways of solving the problem. One using total probability, the other by symmetry.
    + Total Probability.<br/>Let $E$ be the desired event that exactly one of Alice or Bob is a winner. We divide the sample space into two disjoint events, $E_1$, $E_2$. $E_1$ is the event that Alice and Bob play against each other and $E_2$ is the complimentary event that Alice and Bob play against other players. Since Alice is equally likely to play with any of the seven other players, $P(E_1)=1/7$, hence $P(E_2)=6/7$. Now $P(E|E_1)=1$, while $P(E|E_2)=1/2$ since Alice and Bob each play an independent game where the probability of winning is $1/2$. Therefore $P(E)=P(E_1)⋅P(E|E_1)+P(E_2)⋅P(E|E_2)=1/7⋅1+6/7⋅1/2=4/7$.
    + Symmetry.<br/>In the end, 4 of the 8 players will be declared winners. There are $\tbinom{8}{4}$ such 4-winner "quartets", all equally likely.  The number of "quartets" that contain exactly one of Alice and Bob is $\tbinom{2}{1} \cdot \tbinom{6}{3}$.  Hence the probability that this occurs is $\frac{\tbinom{2}{1} \cdot \tbinom{6}{3}}{\tbinom{8}{4}}=47$.



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









