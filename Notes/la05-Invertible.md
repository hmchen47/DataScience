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







