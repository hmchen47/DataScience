# Topic 4: Combinatorics


## 4.1 Permutations

+ Permutations
  + a <span style="color: magenta;">permutation</span> of a set is an ordering of objects
  + \# permutations of n objects
  + objects can be anything
  + using letters to represent the objects

+ Counting permutations
  + 2 objects:
    + by letters - 1st choice: 2; 2nd choice: 1
    + by tree structure - 1st level node: 2 branches; 2nd level node: 1 branch
    + \$ permutation = 2 x 1 = 2
  + 3 objects
    + by letters - 1st choice: 3; 2nd choice: 2; 3rd choice: 1
    + by tree structure - 1st level node: 3 branches; 2nd level node: 2 branches; 3rd level node: 1 branch
    + \# permutations = 3 x 2 x 1 = 6
  + \# permutations of n objects = $n \times (n-1) \times \cdots \times 2 \times 1 \triangleq n! \to n$ factorial

+ 0 factorial
  + for $n \geq 1$, n! = \# permutations of an n-set = $n \times (n-1) \times \cdots \times 2 \times 1$
  + what about 0! ?
  + how many ways can you permute 0 objects?
    + 2 objects: a, b $\to$ (ab), (ba)
    + 1 object: a $\to$ (a)
    + 0 object: $\varnothing \to ()$
  + $0! = 1 \to$ exact same exact same reason as $2^0 =1$
  
+ Alternative factorial view
  + counting by writing elements left to right $\to$ smallest to largest
    + one position for the 1st element: only one possibility
    + two positions for the 2nd element: one on each side of the 1st element
    + 3 positions for the 3rd element: left-most, middle, right-most
    + 4 positions for the 4th element: 2 outer and 2 middle
    + and so on
  + $n \times (n-1) \times \cdots \times 2 \times 1 = n!$

+ Recursively definition
  + n! defined recursively

    \[\begin{align*}
      n! &= ncdot (n-1) \cdot \dots \cdot 2 \cdot 1 \\
      &= n \cdot [(n-1) \cdot \dots \cdot 2 \cdot 1] \\
      &= n \cdot (n-1)! \quad \forall\, n \geq 1
    \end{align*}\]

  + 0 factorial: 1! = 1 x 0!
  + able to extend to negatives

+ Example: Basic permutations
  + \# orders to visit 3 cities: LA, SD, SF
    + 3! = 3 x 2 x 1 = 6
  + \# anagrams of 5 distinct letter: PEARS
    + 5! = 5 x 4 x 3 x 2 x 1 = 120

+ Constrained anagrams of PEARS
  + A, R staying adjacent in order
    + permutations of P, E, AR, S
    + \# of permutation: 4! = 4 x 3 x 2 x 1 = 24
  + A, R adjacent in any order
    + permutations of P, E, (AR, RA), S
    + 2 orders, 24 anagrams each
    + 2 x 24 = 48
  + A, R not adjacent: 5! - 48 = 120 - 48 = 72

+ More constrained permutations
  + \# ways 3 distinct boys and 2 distinct girls can stand in a row
  + unconstrained: (3+2)! = 5! = 120
  + alternating boys and girls: must be 'b, g, b, g, b' $\to$ 3! x 2! = 6 x 2 = 12
  + boys together and girls together: '3b, 2g' or '2g, 3b' $\to$ 2 x 3! x 2! = 24
  + unconstrained, but orientation (left to right) doesn't matter: 5! / 2 = 60

+ Circular arrangements
  + \# ways 5 people can sit at a round table
  + rotations matter: 5! = 120
  + rotations don't matter: 5!/5 = 4! = 24 $\gets$ alternatively, start w. A and arrange 4 other clockwise

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://tinyurl.com/y9cevx3r" ismap target="_blank">
      <img src="img/t04-01a.png" style="margin: 0.1em;" alt="Circularr arrangements w/o rotation" title="Circular arrangements w/o rotation" height=100>
      <img src="img/t04-01b.png" style="margin: 0.1em;" alt="Circularr arrangements w/ rotation" title="Circular arrangements w/ rotation" height=100>
    </a>
  </div>

+ Stirling's approximations
  
  \[ n! \sim \sqrt{2\pi n} \left( \frac{n}{e} \right)^n \]

  <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
    <a href="https://en.wikipedia.org/wiki/Stirling%27s_approximation" ismap target="_blank">
      <img src="https://tinyurl.com/yav2a9r7" style="margin: 0.1em;" alt="text" title="caption" width=350>
    </a>
  </div>

+ [Original Slides](https://tinyurl.com/y9cevx3r)


### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 4.2 Partial Permutations






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 4.3 Combinations






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 4.4 Applications of Binomial Coefficients






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 4.5 Properties of Binomial Coefficient






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 4.6 Binomial Theorem






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 4.7 Multinomials






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## 4.8 Stars and Bars






### Problem Sets




### Lecture Video

<a href="url" target="_BLANK">
  <img style="margin-left: 2em;" src="https://bit.ly/2JtB40Q" width=100/>
</a><br/>


## Lecture Notebook 4









## Programming Assignment 4






