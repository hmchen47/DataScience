# Topic 4: Combinatorics


## 4.1 Permutations

+ Permutations
  + a <span style="color: magenta;">permutation</span> of a set is an ordering of its elements
  + \# permutations of a set: determined by just set size
  + \# permutations of an $n$-set: $n \times (n-1) \times \cdots \times 2 \times 1 \triangleq n! \to n$ factorial
  + for $n \geq 1$, $n! = #$ permutations of an n-set = $n \times (n-1) \times \cdots \times 2 \times 1$
  + 0 factorial
    + how many ways can you permute 0 objects?
    + $0! = 1 \to$ exact same exact same reason as $2^0 =1$
  
+ Alternative factorial view
  + $n! = n \times (n-1) \times \cdots \times 2 \times 1 = 1 \times 2 \times \cdots \times n$
  + write elements left to right $\to$ smallest to largest

+ n factorial and (n-1) factorial
  
  \[ n! = \underbrace{1 \cdot 2 \cdot \dots \cdot (n-1)}_{(n-1)!} \cdot n \]

  \[ \therefore\; n! = (n-1)! \cdot n \qquad \forall\, n \geq 1 \]

  + determine n! recursively

+ Basic permutations
  + \# orders to visit 3 cities: LA, SD, SF
    + $3! = 3 \times 2 \times  =6$
  + \# ways to rank 4 students
    + $4! = 4 \times 3 \times 2 \times 1 = 24$

+ Anagrams
  + definition: a word or phrase made by transposing the letters of another word or phrase
  + ignoring whether the word has a meaning
  + for now, all letters distinct, but letters repeat later
  + e.g., \# Anagrams of PEARS: all permutations of {P, E, A, R, S} $\to 5! = 120$

+ Constrained anagrams of PEARS
  + A, R staying adjacent in order
    + permutations of P, E, AR, S
    + \# of permutation: 4! = 4 x 3 x 2 x 1 =24
  + A, R are adjacent in either order
    + 2 orders, 24 anagrams each
    + 2 x 24 = 48
  + A, R not adjacent: 5! - 48 = 120 - 48 = 72

+ More constrained permutations
  + \# ways 3 distinct boys and 2 distinct girls can stand in a row
  + unconstrained: (3+2)! = 5! = 120
  + alternating boys and girls: must be 'b, g, b, g, b' $\to$ 3! x 2! = 6 x 2 = 12
  + boys together and girls together: '3b, 2g' or ''2g, 3b' $\to$ 2 x 3! x 2! = 24
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






