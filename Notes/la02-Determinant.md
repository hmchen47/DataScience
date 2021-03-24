# Determinant

[Origin](https://en.wikipedia.org/wiki/Determinant)


## Introduction

+ Overview of determinant
  + denoted as $\det(A)$, $\det A$ and $|A|$
  + viewed as the scaling factor of the transformation described by the matrix
  + $2 \times 2$ matrix

    \[ |A| = \begin{vmatrix} a&b\\c&d \end{vmatrix} = ad - bc \]

  + $3 \times 3$ matrix

    \[\begin{align*}
      |A| &= \begin{vmatrix} a&b&c\\d&e&f\\g&h&i \end{vmatrix} \\
      &= a \begin{vmatrix} e&f\\h&i \end{vmatrix} - b \begin{vmatrix} d&f\\g&i \end{vmatrix} + c \begin{vmatrix} d&e\\g&h \end{vmatrix}
    \end{align*}\]

  + usages
    + solving linear equations $\to$ not efficient
    + Jacobian determinant in calculus: change of variables rule for integrals of functions of several variables
    + define the characteristics polynomial of a matrix $\to$ essential for eigenvalue problems in linear algebra
    + analytic geometry: the signed n-dim volume of n-dim parallel
  + unique inverse w/ a matrix $A \iff \det(A) \neq 0$


## Definitions

+ General definition of determinant
  + consider top row and the respective minors: from left, multiply the element by the mirror, the subtract the product of the next element and its minor, and alternate adding and subtract such product until all elements in the top row exhausted
  + example: $4 \times 4$ matrix

    \[\begin{align*}
      \det(\mathbf{C}) &= \det\left(\begin{bmatrix} a&b&c&d\\e&f&g&h\\i&j&k&l\\m&n&o&p \end{bmatrix}\right) \\
        &= a \begin{vmatrix}f&g&h\\j&k&l\\n&o&p \end{vmatrix} - b \begin{vmatrix} e&g&h\\i&k&l\\m&o&p \end{vmatrix} + c \begin{vmatrix} e&f&h\\i&j&l\\m&n&p \end{vmatrix} -d \begin{vmatrix} e&f&g\\i&j&k\\m&n&o \end{vmatrix}
    \end{align*}\]

  + alternative expression: columns of matrix

    \[\begin{align*}
      &A = [a_1, a_2, \dots, a_n],  \hspace{1em}  a_j: \text{the } j^{\text{th}} \text{ column of the mathbftor w/ size} \\
      &\text{ s.t.} \\
      &\det[a_1, \dots, ba_j+cv, \dots, a_n] = b \cdot \det(A) + c \cdot \det([a_1, \dots, v, \dots, a_n])\\
      &\det([a_1, \dots, a_j, a_{j+1}, \dots, a_n]) = -\det([a_1, \dots, a_{j=1}, a_j, \dots, a_n]) \\
      &\det(I) = 1
    \end{align*}\]

    + $a, c$: scalar
    + $v$: mathbftor (n)
    + $I$: identity matrix ($n \times n$)

  + the determinant is an algorithm multilinear function of the columns that maps the identity matrix to the underlying unit scalar
  + determinant:
    + sun of products of entities of the matrix
    + each product has $n$ terms
    + the coefficient of each product is -1 or 1 or 0 according to a given rule, a polynomial expression of the matrix entities
    + assume $A$ is a square matrix w/ $n$ rows and $n$ columns
      + the entries can be numbers and expressions
      + the determinant of $A$ denoted by $\det(A)$

      \[ A = \begin{bmatrix} a_{1,1} & a_{1, 2} & \cdots & a_{1,n} \\ a_{2, 1} & a_{2, 2} & \cdots & a_{2, n} \\
        \vdots & \vdots & \ddots & \vdots \\ a_{n, 1} & a_{n, 2} & \cdots & a_{n, n} \end{bmatrix} \hspace{1em}\to\hspace{1em}
        \det(A) = \begin{vmatrix} a_{1,1} & a_{1, 2} & \cdots & a_{1,n} \\ a_{2, 1} & a_{2, 2} & \cdots & a_{2, n} \\
        \vdots & \vdots & \ddots & \vdots \\ a_{n, 1} & a_{n, 2} & \cdots & a_{n, n} \end{vmatrix}
      \]

+ $2 \times 2$ matrix
  + the [Leibniz formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_determinants) for the determinant

    \[ \begin{vmatrix} a&b\\c&d \end{vmatrix} =  ad - bc \]

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 10vw;"
        onclick= "window.open('https://en.wikipedia.org/wiki/Determinant')"
        src    = "https://upload.wikimedia.org/wikipedia/commons/a/ad/Area_parallellogram_as_determinant.svg"
        alt    = "The area of the parallelogram is the absolute value of the determinant of the matrix formed by the mathbftors representing the parallelogram's sides."
        title  = "The area of the parallelogram is the absolute value of the determinant of the matrix formed by the mathbftors representing the parallelogram's sides."
      />
    </figure>

    + the absolute value of $ad-bc$ = the area of parallelogram 
    + the scale factor by which areas are transformed by $A$
    + the sign becomes the oriented are of the parallelogram $\to |a||b| sin\theta$ 

      \[ \text{signed area } = |\mathbf{u}||\mathbf{v}| sin(\theta) = |\mathbf{u}^{\perp}||\mathbf{v}| cos(\theta^\prime\}) = \binom{-b}{a} \binom{c}{d} = ad-bc \]

  + the determinant gives the scaling factor and orientation introduced by the mapping represented by $A$
  + $\det(A) = 1$: the linear mapping defined by the matrix is equi-areal and orientation-preservation

+ $3 \times 3$ matrix
  + the [Laplace formula](https://en.wikipedia.org/wiki/Laplace_expansion) of the determinant

    \[ \det(A) = \begin{vmatrix} a&b&c\\d&e&f\\g&h&i \end{vmatrix} = a \begin{vmatrix} e&f\\h&i \end{vmatrix} - b \begin{vmatrix} d&f\\g&i \end{vmatrix} + c \begin{vmatrix} d&e\\g&h \end{vmatrix} \]

  + the [Leibniz formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_determinants) of the determinant

    \[ \begin{align*}
      \begin{vmatrix} a&b&c\\d&e&f\\g&h&i \end{vmatrix} &= a(ei-fh) -b(di-fg) +c(dh-eg) \\
      &= aei + bfg+cdh+ceg-bdi-afh
    \end{align*}\]

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 10vw;"
        onclick= "window.open('https://en.wikipedia.org/wiki/Determinant')"
        src    = "https://upload.wikimedia.org/wikipedia/commons/b/b9/Determinant_parallelepiped.svg"
        alt    = "The volume of this parallelepiped is the absolute value of the determinant of the matrix formed by the columns constructed from the vectors r1, r2, and r3."
        title  = "The volume of this parallelepiped is the absolute value of the determinant of the matrix formed by the columns constructed from the vectors r1, r2, and r3."
      />
    </figure>

  + the [rule of Sarrus](https://en.wikipedia.org/wiki/Rule_of_Sarrus)
    + a monotonic for the $3 \times 3$ matrix determinant
      + the sum of the products of 3 diagonal north-west to south-east lines of the matrix elements
      + minus the sum of products of 3 diagonal south-west to north-east lines of elements

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 10vw;"
        onclick= "window.open('https://en.wikipedia.org/wiki/Determinant')"
        src    = "https://upload.wikimedia.org/wikipedia/commons/6/60/Sarrus_ABC_red_blue_solid_dashed.svg"
        alt    = "Rule of Sarrus: The determinant of the three columns on the left is the sum of the products along the down-right diagonals minus the sum of the products along the up-right diagonals."
        title  = "Rule of Sarrus: The determinant of the three columns on the left is the sum of the products along the down-right diagonals minus the sum of the products along the up-right diagonals."
      />
    </figure>
  
+ $n \times n$ matrix
  + the Leibniz formula of the determinant of a matrix of arbitrary size

    \[ \det(A) = \sum_{\sigma \in S_n} \left( \text{sgn}(\sigma) \prod_{i=1}^n a_{i, \sigma_)i} \right) \]

    + sum: computed over all permutations $\sigma$ of the set $\{1, 2, \dots, n\}$
    + permutation: a function reordering this set of integers
    + the value in the $i^{th}$ position after the reordering $\sigma$ denoted by $\sigma_i$
    + the set of all such permutations (known as the symmetric group on $n$ elements) denoted by $S_n$

      \[ \text{sgn}(\sigma) = \begin{cases} 1 & \text{even number interchanges} \\ -1 &\text{odd number interchanges} \end{cases} \]

      e.g., $[1, 2, 3] \to \sigma = [2, 3, 1] w/ \sigma_1 = 2$, $\sigma_2 = 3$, and $\sigma_3 = 1 \implies$ even
    + the product of the entries at position $(i, \sigma_i)$, $\prod_{i=1}^n a_{i, \sigma_i}$, where $i$ ranges from 1 to $n$, $a_{i, \sigma_i} \cdot a_{2, \sigma_2} \cdots a_{n, \sigma_n}$
    + example: the determinant of a $3 \times 3$ matrix $A (n=3)$

      \[\begin{align*}
        &\sum _{\sigma \in S_{n}}\text{sgn}(\sigma )\prod _{i=1}^{n}a_{i,\sigma _{i}}\\
        ={}&\text{sgn}([1,2,3])\prod _{i=1}^{n}a_{i,[1,2,3]_{i}}+\text{sgn}([1,3,2])\prod _{i=1}^{n}a_{i,[1,3,2]_{i}}+\text{sgn}([2,1,3])\prod _{i=1}^{n}a_{i,[2,1,3]_{i}}+{}\\
        &\text{sgn}([2,3,1])\prod _{i=1}^{n}a_{i,[2,3,1]_{i}}+\text{sgn}([3,1,2])\prod _{i=1}^{n}a_{i,[3,1,2]_{i}}+\text{sgn}([3,2,1])\prod _{i=1}^{n}a_{i,[3,2,1]_{i}}\\
        ={}&\prod _{i=1}^{n}a_{i,[1,2,3]_{i}}-\prod _{i=1}^{n}a_{i,[1,3,2]_{i}}-\prod _{i=1}^{n}a_{i,[2,1,3]_{i}}+\prod _{i=1}^{n}a_{i,[2,3,1]_{i}}+\prod _{i=1}^{n}a_{i,[3,1,2]_{i}}-\prod _{i=1}^{n}a_{i,[3,2,1]_{i}}\\[2pt]={}&a_{1,1}a_{2,2}a_{3,3}-a_{1,1}a_{2,3}a_{3,2}-a_{1,2}a_{2,1}a_{3,3}+a_{1,2}a_{2,3}a_{3,1}+a_{1,3}a_{2,1}a_{3,2}-a_{1,3}a_{2,2}a_{3,1}.
      \end{align*}\]

+ [Leibniz formula for determinants](https://en.wikipedia.org/wiki/Leibniz_formula_for_determinants)
  + expressing the determinant of a square matrix in terms of permutations of the matrix elements
  + suppose that $A$ as an $n \times n$ matrix, where $a_{i, j}$ is the entry in the $i^{th}$ row and $i^{th}$ column of $A$

    \[ \det(A) = \sum_{\tau \in S_n} \text{sgn}(\tau) \prod_{i=1}^n a_{i, \tau(i)} = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{\sigma(i)} \prod_{i=1}^n a_{\sigma(i), i} \]

  + $\text{sgn}$: the sign function pf permutations in the permutation group $S_n$ which returns +1 nd -1 for even and odd permutations, respectively

+ Levi-Civita symbol
  + extending the Leibniz formula to a summation in which not only permutations
  + all sequences of $n$ indices in the range $1, \dots, n$ occur, ensuring that the contribution of a sequence will be zero unless it denotes a permutation
  + representing a collection of numbers
  + defined from the sign of a permutation of the natural numbers
  + a.k.a., permutation symbol, antisymmetric symbol and alternating symbol
  + index notation: display permutations in a way compatible w/ tensor analysis, $\varepsilon_{i_1 i_2 \dots i_n}$
    + each index $i_1, i_2, \dots, i_n$ taking values $1, 2, \dots, n$
    + there are $n^n$ indexed values of $\varepsilon_{i_1 i_2 \dots i_n}$, arranged into an n-dim array
  + key property: total antisymmetry in the indices
    + any two indices interchanged, the symbol negated
    + $\varepsilon_{\dots i_p \dots i_q \dots} = -\varepsilon_{\dots i_q \dots i_p \dots}$
  + any two equal indices: $\varepsilon = 0$
  + all indices unequal: $\varepsilon_{i_1 i_2 \dots i_n} = (-1)^p \varepsilon_{1 2 \dots n}$
    + $p$: parity of the permutation, the number of pairwise interchanges of indices necessary to unscramble $i_1, i_2, \dots, i_n$ into the order $1, 2, \dots, n$
    + $(-1)^p$: the sign or signature of the permutation
  + mostly choose $\varepsilon_{1 2 \dots n} = +1$: the Levi-Citvita symbol equals the sign of a permutation when the indices are unequal
  + the determinant for an $n \times n$ matrix expressed using an $n$-fold summation

    \[\begin{align*} \det(A) &= \sum_{i_1, i_2, \dots, i_n = 1} \varepsilon_{i_1 i_2 \dots i_n} a_{a_{1, i_1} \cdots a_{a_{n, i_n}}} \\
      &= \frac{1}{n!} \sum \varepsilon_{i_1 \dots i_n} \varepsilon_{j_1 \dots j_n} a_{i_1 j_1} \cdots a_{i_nj_n}
    \end{align*} \]

    + each $i_r$ and each $j_r$ summed over $1, \dots, n$

## Properties




## Applications





