# Invertible Matrix

Organization: Wikipedia

[Origin](https://en.wikipedia.org/wiki/Invertible_matrix)


## Overview

+ Invertible matrix
  + a.k.a., nonsingular, nondegenerate
  + Def: $\forall\; A_{n \times n}, \exists\; B_{n \times n} \text{ s.t. } AB = BA = I_n$
  + $B$: the (multiplicative) inverse of $A$, uniquely determined by $A$
  + __matrix invsion:__ the process of finding the matrix $B$ satisfying the prior equation for a given invertible matrix $A$

+ Singular/degenerate matrix
  + a square matrix not invertible
  + singular matrix $A \iff \det(A) = 0$
  + rare in probability: almost never singular
  + no inverse matrix w/ non-square matrix
  + some non-square matrices w/ left or right inverse
    + left inverse: $A_{m \times n}$ w/ $\operatorname{rank} = n \;(m \le n) \implies \exists\; B_{n \times m} \text{ s.t. } BA = I_n$
    + right inverse: $A_{m \times n}$ w/ $\operatorname{rank} = m \; (m \le n) \implies \exists\; B_{n \times m} \text{ s.t. } AB = I_m$

## Properties

+ The invertible matrix theorem: $A_{n \times n}$ over a field $K$, the following statements equivalent
  + invertible $A \implies$ nonsingular or non-degenerate
  + $A$ row-equivalent to $I_n$
  + $A$ column-equivalent to $I_n$
  + $A$ w/ n pivot positions
  + $\det(A) \ne 0$
  + $A$ w/ full rank, i.e., $\operatorname{rank}(A) = n$
  + $A\bf{x} = 0$ w/ only the trivial solution $\bf{x} = 0$
  + the kernel of $A$ is trivial, i.e., it contains only the null vector as anb element, $\operatorname{ker}(A) = \{0\}$
  + the equation $A\bf{x} = \bf{b}$ w/ exactly one solution $\forall\; \bf{b} \in K^n$
  + the columns of $A$ linearly independent
  + the columns of $A$ span $K^n$
  + $\operatorname{Col} A = K^n$
  + the columns of $A$ form a basis of $K^n$
  + the linear transformation mapping $\bf{x}$ to $A\bf{x}$ is bijection from $K^n$ to $K^n$
  + $\exists\; B_{n \times n}$ s.t. $AB = I_n=BA$
  + transpose $A^T$ invertible ($\therefore$ rows of $A$ linear independent, span $K^n$, and form a basis of $K^n$)
  + the number 0 not an eigenvalue of $A$
  + $A$ expressed as a finite product of elementary matrices
  + $A$ w/ left or right inverse, which case both left and right inverse and $B = C = A^{-1}$




## Examples





## Methods of matrix invertible





### Gaussian elimination





### Newton's method





### Cayley-Hamilton method





### Eigendecomposition





### Cholesky decomposition





### Analytic solution





### Blockwise inversion





### By Neumann series





### p-adic approximation





### Reciprocal basis vectors method






## Derivative of the matrix inverse






## Generalized inverse





## Applications




### Regression/least square





### Matrix inverses in real-time simulations





### Matrix inverses in MIMO wireless communication







