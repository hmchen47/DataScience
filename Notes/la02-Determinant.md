# Determinant

Organization: Wikipedia

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

+ [Levi-Civita symbol](https://en.wikipedia.org/wiki/Levi-Civita_symbol)
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

+ Basic properties
  1. $\det(I_n) = 1, I_n: n \times n$ identity matrix
  2. $\det(A^T) = \det(A)$
  3. $\det(A^{-1}) = 1/\det(A) = \det(A)^{-1}$
  4. $\det(AB) = \det(A) \cdot \det(B)$
  5. $\det(cA) = c^n \cdot \det(A), A_{n \times n}$
  6. $A, B, C$ are positive semidefinite matrices w/ same size, $\det(A+B+C) + \det(C) \geq \det(A+C) + \det(B+C), \;\forall A, B, C > 0$ with the corollary $\det(A+B) \geq \det(A) + \det(B)$
  7. $A$ is triangle matrix, i.e., $a_{ij} = 0, \;\forall i > j \text{ or } i < j \implies$its determinant equals the product of the diagonal entities

    \[ \det(A) = a_{11} a_{22} \cdots a_{nn} = \prod_{i=1}^n a_{ii} \]

  8. the determinant as an $n$-linear equation by viewing an $n \times n$ matrix w/ $n$ columns
      + the $j^{th}$ column of a matrix $A$ as a sum $\mathbf{a}_j = \mathbf{v} + \mathbf{w}$ of two column vectors and others remaining unchanged
      + the determinant of $A$ = the sum of the determinant of the matrices obtained from $A$ by replacing the $j^{th}$ by $\mathbf{v}$ (denoted $A_v$) and then by $\mathbf{w}$ (denoted $A_w$)

      \[\begin{align*}
        \det(A) &= \det([a_1| \cdots |a_j| \cdots a_n]) = \det([\cdots | \mathbf{v + w} \cdots ]) \\
          &= \det([\cdots | \mathbf{v} \cdots ]) + \det([\cdots | \mathbf{w} \cdots ]) = \det(A_v) + \det(A_w)
      \end{align*}\]

  9. $\exists \text{ a } n \times n \text{ matrix } A$ w/ $a_{ij} = 0, \forall\; i, j \in [1, n] \implies \det(A) = 0$
  10. alternating form of $n$-linear function: two or more columns identical of an matrix $A \implies \det(A) = 0$
  11. $n \times n$ matrix composed of $n$ rows, the determinant as an $n$-linear function
  12. alternating form of $n$-linear function: two or more rows of an matrix $A$ identical $\implies \det(A) = 0$
  13. interchanging any pair of columns or rows of a matrix multiplies its determinant by -1
      + any permutation of the rows and columns multiplies the determinant by the sign of the permutation
      + permutation: viewing each row as a vector $\mathbf{R}_i$ (equivalently each column as $\mathbf{C}_j$) and reordering the rows (or columns) by interchange of $\mathbf{R}_j$ and $\mathbf{R}_k$ (or $\mathbf{C}_j$ and $\mathbf{C}_k$), where $j, k \in [1, n]$
  14. adding a scalar multiple of one column to another column not change the value of the determinant

+ Example: Gaussian elimination

  \[ A = \begin{bmatrix} -2&2&-3\\-1&1&3\\2&0&-1 \end{bmatrix}, B = \begin{bmatrix} -2&2&-3\\0&0&4.5\\2&0&-1 \end{bmatrix}, C = \begin{bmatrix} -2&2&-3\\0&0&4.5\\0&2&-4 \end{bmatrix}, D = \begin{bmatrix} -2&2&-3\\0&2&-4\\0&0&4.5 \end{bmatrix} \]

  + $\det(A) = \det(B) = - \det(C)$
  + $D$: an (upper) triangular matrix, the determinant as the product of the entries ion the main diagonal, $(-2) \cdot 2 \cdot 4.5 = -18$

+ [Schur complement](https://en.wikipedia.org/wiki/Schur_complement)
  + suppose $p, q$ are non-negative integers
  + suppose $A, B, C, D$ are respectively $p \times q$, $p\times q$, $p \times p$, and $q \times q$ matrices of complex numbers
  + let $M$ as a $(p + q) \times (p + q)$ matrix

    \[ M = \begin{bmatrix} A&B\\C&D \end{bmatrix} \]

  + Def: w/ invertible $D$, the __Schur complement__ of the block $D$ of the matrix $M$ is the $p \times p$ matrix defined by

    \[ M / D :=  A - BD^{-1}C \]

  + Def: w/ invertible $A$, the __Schur complement__ of the block $A$ of the matrix $M$ is the $q \times q$ matrix defined by

    \[ M / A := A - CA^{-1}B \]

  + generalized Schur complement: substituting a generalized inverse for the inverses on $M/A$ or $M/D$ if $A$ or $D$ singular
  + property on determinant

    \[\begin{align*}
      \det(M) &= \det(A)\det(D - CA^{-1}B), \text{ if } A \text{ invertible}\\
      \det(M) &= \det(D)\det(A - BD^{-1}C), \text{ if } D \text{ invertible}
    \end{align*}\]

+ Multiplicativity and matrix groups
  + multiplicative map
  
    \[ \det(AB) = \det(A) \det(B) \]

  + a consequence of the characterization given above of the determinant as the unique n-linear alternating function of the columns w/ value 1 on the identity matrix
  + $M_n(K) \to K$ mapping $M \mapsto \det(AM)$
  + generalized to (square) products of rectangular matrices, given the Cauchy-Binet formula, providing an independent proof of the multiplicative property
  + $\det(A) \neq 0 \iff A$ invertible
  + the determinant of the inverse matrix

    \[ \det(A^{-1}) = \frac{1}{\det(A)} = \left[\det(A) \right]^{-1} \]

+ Laplace's expansion and the adjugate matrix
  + expression the determinant of a matrix in terms of its minors
  + Minor $M_{ij}$: the determinate of the $(n-1) \times (n-1)$-matrix
  + $(-1)^{i+j} M_{ij}$: cofactor of $M_{i, j}$
  + Laplace expansion along the $i$th row of the determinant of $A$

    \[ \det(A) = \sum_{j=1)^n (-1)^{i+j} a_{ij}}M_{ij}, \;\;\forall i \]

  + Laplace expansion along the $j$th column of the determinant of $A$

    \[ \det(A) = \sum_{i=1}^n (-1)^{i+j} a_{ij}M_{ij} \]

  + efficient for small matrices and spare matrices only
  + exponential number of determinants for general matrices
  + $adj(A)$
    + adjugate matrix
    + transpose of the matrix of cofactor

      \[ (\operatorname{adj}(A))_{ij} = (-1)^{i+j} M_{ji} \]

    + $\forall\;$ matrix $A$

      \[ \det(A)I = A \operatorname{A} = (\operatorname{A}} A \]

    + used for expressing the inverse of the nonsingular matrix

      \[ A^{-1} = \frac{1}{\det A} \operatorname{A} A \]

  + example: Laplace expansion of $3 \times 3$ matrix

    \[A = \begin{bmatrix} -2&2&-3\\-1&1&3\\2&0&-1 \end{bmatrix} \]

    along the second column ($j = 2$ and the sum over $i$)

    \[\begin{align*}
      \det(A) &= (-1)^{1+2} \cdot 2 \cdot \begin{vmatrix} -1&3\\2&-2 \end{vmatrix} + (-1)^{2+2} \cdot 1 \cdot \begin{bmatrix} -2&-3\\2&-1 \end{bmatrix} + (-1)^{3+2} \cdot 0 \cdot \begin{bmatrix} -2&-3\\-1&3 \end{bmatrix} \\
      &= (-2) \cdot ((-1) \cdot (-1) - 2 \cdot 3) + 1 \cdot ((-2) \cdot (01) -2 \cdot (-3)) \\
      &= (-2) \cdot (-5) + 8 = 18
    \end{align*}\]

+ Sylvester's determinant theorem
  + for matrices $A_{m\times n}$ and $B_{n \times m}$

    \[ \det(I+m + AB) = \det(I_n + BA) \]

    where $I_m$ and $I_n$ are $m \times m$ and $n \times n$ identity matrices, respectively

  + general results
    + quick calculation of the determinant of a matrix differs from the identity matrix by a matrix of rank 1 w/ column vector $c$ and row vector $r$

      \[ \det(I_m + cr) = 1 + rc \]

    + more generally, $\forall$ invertible matrix $X_{m \times m}$

      \[\begin{align*}
        \det(X + AB) &= \det(X)\det(I_n + BX^{-1}A) \\\\
        \det(X + cr) &= \det(X)\det(1 + rX^{-1}c) = \det(X) + r {\operatorname{X}} c
    \end{align*}\]

    + $\forall\;$ square matrices $A, B$ w/ the same size, $AB$ and $BA$ w/ the same characteristic polynomials

## Properties of determinant in relation to other notions

+ Eigenvalues
  + $A$: an arbitrary $n \times n$ matrix of complex numbers w/ eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$
  + pseudo determinant: the determinant of $A$ as the product of all eigenvalues

    \[ \det(A) = \prod_{i=1}^n \lambda_i = \lambda_1 \lambda_2 \cdots \lambda_n \]

  + conversely, determinant used to find the eigenvalues of matrix $A$: the solution of the characteristic equation

    \[ \det(A - \lambda I) = 0 \]

  + Hermitian matrix as positive definite: all eigenvalues positive
  + Sylvester's criterion asserts that positive definite Hermitian matrix equivalent to determinants of the sub-matrices

    \[ A_k = \begin{bmatrix} a_{1,1}&a_{1,2}&\cdots&a_{1,k}\\a_{2,1}&a_{2,2}&\cdots&a_{2,k}\\ \vdots&\vdots&\ddots&\vdots\\a_{k,1}&a_{k,2}&\cdots&a_{k,k} \end{bmatrix} \hspace{1em} \forall\; k \in [1, n] \]

+ Trace
  + $\operatorname{tr}(A)$: trace of $A$
  + the sum of the diagonal entries of $A$ and also equals the sum of the eigenvalues
  + complex matrices

    \[ \det(\exp(A)) = \exp(\operatorname{tr}(A)) \]

  + real matrices

    \[ \operatorname{tr}(A) = \log(\det(\exp(A))) \]

  + $\exp(A)$: matrix exponential of $A$
    + $\forall\;$ eigenvalue $\lambda$ of $A$ corresponding to the eigenvalue $\exp(\lambda)$ of $\exp(A)$
    + $\forall\; L$ as logarithm of $A$ satisfying $\exp(L) = A$, the determinant of $A$

      \[ \det(A) = \exp(\operatorname{tr}(L)) \]

  + example:

    \[\begin{align*}
      \det(A) &= \frac{1}{2} \left((\operatorname{tr}(A))^2 - \operatorname{tr}(A^2) \right) \hspace{1em} n = 2\\
      \det(A) &= \frac{1}{6} \left((\operatorname{tr}(A))^3 - 3\operatorname{tr}(A)\operatorname{tr}(A^2) + 2 \operatorname{tr}(A^3) \right) \hspace{1em} n = 3\\
      \det(A) &= \frac{1}{24} \left((\operatorname{tr}(A))^4 - 6\operatorname(A^2)(\operatorname{tr}(A))^2 + 3 (\operatorname{tr}(A^2))^2 + 8 \operatorname{tr}(A^3) \operatorname{tr}(A) - 6\operatorname{tr}(A^4) \right) \hspace{1em} n = 4\\
    \end{align*}\]

+ Upper and lower bounds
  + $A$: a positive definite matrix
  + trace operator providing the following tight lower and upper bounds on the log determinant

    \[\begin{align*}
      &\operatorname{tr}(I - A^{-1}) \le \log\det(A) \le \operatorname{tr}(A - I) \\\\
      &\frac{n}{\operatorname{tr}(A^{-1})} \le \det(A)^{\frac{1}{n}} \le \frac{1}{n} \operatorname{tr}(A) \leq \sqrt{\frac{1}{n} \operatorname{tr}(A^2)}
    \end{align*}\]

    + equality $\iff A = I$
    + able to be derived via the formula for the KL-divergence btw two multivariate normal distributions

+ Cramer's rule
  + matrix equations: $Ax = b$
  + $A$: nonzero determinant
  + solution: 

    \[ x_i = \frac{\det(A_i)}{\det(A)}} \hspace{1em} i=1, 2, 3, \dots, n \]

    + $A_i$: the matrix formed by replacing the $i$th column of $A$ by the column vector $b$
  + column expansion of the determinant w/ vectors $a_j$ as the columns of $A$

    \[ \det(A_i) = \det [a_1 \cdots b \cdots a_n] = \sum_{j=1}^n x_j \det[a_1 \cdots a_{i-1} a_j a_{i+1} \cdots a_n] = x_i \det(A) \]

+ Block matrices
  + matrices: $A_{n \times n}, B_{n \times m}, C_{m \times n}, D_{m \times m}$
  
    \[ \det \begin{pmatrix} A&0\\C&D \end{pmatrix} = \det(A) \det(D) = \det \begin{pmatrix} A&B\\0&D \end{pmatrix} \]

  + $A$ invertible

    \[ \det \begin{pmatrix} A&B\\C&D \end{pmatrix} = \det(A)\det(D - CA^{-1}B) \]

  + $D$ invertible

    \[ \det \begin{pmatrix} A&B\\C&D \end{pmatrix} = \det(D)\det(A - BD^{-1}C) \]

  + All blocks are square matrices w/ same order, $C$ and $D$ commute $\implies$

    \[ \det \begin{pmatrix} A&B\\C&D \end{pmatrix} = \det (AD - BC) \]

  + Generalization: all pairs of $n \times n$ matrices of the $np \times np$ block matrices commute $\implies$ the determinant of the block matrices equal to the determinant of the matrix obtained by computing the determinant of the block matrix considering its entries as the entries of a $p \times p$ matrix
  + All blocks are square matrices w/ same order, $A = D$ and $B = C$ $\implies$

    \[ \det \begin{pmatrix} A&B\\B&A \end{pmatrix} = \det(A-B) \det(A+B) \]

  + $D_{1 \times 1}$, $B$ a column vector, and $C$ as a row vector $\implies$ 

    \[ \det\begin{pmatrix} A&B\\C&D \end{pmatrix} = (D - CA^{-1}B) \det(A) \]

  + $s$ as a scalar complex number, a square block matrix w/ characteristics polynomial factored w/

    \[\begin{align*}
      \det\begin{pmatrix} A-sI&B\\C&D-sI \end{pmatrix} &= \det(A - sI) \det\left(D - \frac{C\operatorname{adj}(A-sI)B}{\det(A-sI)} - sI \right) \hspace{1em} \text{if } s \notin \operatorname{eig}(A) \\
      &= \det(A-sI)^{1-n-m} \det(\det(A-sI)D - c\operatorname{adj}(A-sI)B -\det(A-sI) sI)
    \end{align*}\]

+ Derivative
  + the determinant of real square matrices is a polynomial function $\Bbb{R}^{n\times n} \to \Bbb{R}$
  + the polynomial function differentiable
  + the derivative expressed using Jacobi's formula

    \[ \frac{d \det(A)}{d \alpha} = \operatorname{tr}\left(\operatorname{adj}(A) \frac{d A}{d \alpha} \right) \]

  + $A$ invertible $\implies$

    \[ \frac{d \det(A)}{d \alpha} = \det(A) \operatorname{tr}\left(A^{-1}\frac{d A}{d \alpha} \right) \]

  + expressed in terms of the entries of $A$

    \[ \frac{\partial \det(A)}{\partial A_{ij}} = \operatorname(A)_{ji} = \det(A)(A^{-1})_{ji} \]

  + equivalent formulation

    \[ \det(A+\epsilon X) - \det(A) = \operatorname{tr}(\operatorname{adj}(A)X) \epsilon + O(\epsilon^2) = \det(A)\operatorname(A^{-1}X)\epsilon + O(\epsilon^2) \]


## Applications

+ Linear independence
  + determinant = 0 $\iff$ the column vectors (or the row vectors) of the matrix linearly dependent
  + determinants used to characterize linear dependent vectors
  + differential equations: given $n$ functions $f_1(x), \dots, f_n(x)$, the Wronskian (showing linear independence in a set of solutions)

    \[ W(f_1, \dots, f_n)(x) - \begin{vmatrix} f_1(x)&f_2(x)&\cdots&f_n(x)\\f_1^\prime(x)&f_2^\prime(x)&\cdots&f_n^\prime(x)\\ \vdots&\vdots&\ddots&\vdots\\ f_1^{(n-1)}(x)&f_2^{(n-1)}(x)&\cdots&f_n^{(n-1)}(x)\end{vmatrix} \]

  + $x \neq 0$ in a specific interval $\iff$ the given functions and all their derivatives up to order $n-1$ linearly independent



