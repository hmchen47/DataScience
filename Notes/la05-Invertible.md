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

+ Properties of invertible matrices: $\exists\; A$ invertible
  + $(A^{-1})^{-1} = A$
  + $(kA)^{-1} = k^{-1}A^{-1}, \;\forall\; k \ne 0$
  + $A$ w/ orthonormal columns $\implies (A\bf{x})^+ = \bf{x}^+A^{-1}$, where ${}^+$ denotes the Moore-Penrose inverse and $\bf{x}$ is a vector
  + $(A^T)^{-1} = (A^{-1})^T$
  + $\forall\; \text{ invertible } A_{n \times n}, B_{n \times n} \text{ s.t. } (AB)^{-1} = B^{-1} A^{-1}$.  More generally, invertible $A_1, \dots, A_k \implies$ $(A_1 A_2 \cdots A_k)^{-1} = A_k^{-1} A_{k-1}^{-1} \cdots A_2^{-1} A_1^{-1}$
  + $\det A^{-1} = (\det A)^{-1}$

+ Orthogonal vectors
  + the rows of the inverse matrix $V$ of a matrix $U \to$ orthonormal to the columns of $U$
  + suppose that $UV = VU = I$
    + $v_i^T$: the rows of $V, 1 \le i \le n$
    + $u_j$: the columns of $U, 1 \le j \le n$
  + Euclidean inner product: $v_i^T u_j = \delta_{i,j}$
    + useful in constructing the inverse of a square matrix in some instances
    + a set of orthogonal vectors (not necessarily orthonormal vectors) to the columns of $U$ are known
  + the iterative Gram-Schmidt process applied to the initial set to detemine the rows of the inverse $V$

+ Adjugate matrix
  + used to find the inverse of $A$
  + invertible $A \implies$

    \[ A6{-1} = \frac{1}{\det(A)} \operatorname{adj}(A) \]

+ Identity matrix
  + the associativity of matrix multiplication
  + finite square matrices $A, B \text{ s.t. } AB = I \implies BA = I$


## Examples

+ 2x2 matrix
  + consider invertible $A_{2 \times 2}$

    \[ A = \begin{bmatrix} -1&\frac{32}\\1&-1 \end{bmatrix} \hspace{1em}\to\hspace{1em} \det A = -\frac12 \ne 0 \]

  + consider non-invertible, or singular $B_{2 \times 2}$

    \[ B = \begin{bmatrix} -1&\frac32\\\frac23&-1 \end{bmatrix} \hspace{1em}\to\hspace{1em} \det B = 0 \]


## Methods of matrix invertible

+ Gaussian elimination
  + an algorithm used to 
    + determine whether a given matrix is invertible
    + find the inverse
  + alternative: the UL decomposition, generating upper and lower triangular matrices

+ Newton's method
  + used for a multiplicative inverse algorithm
  + convenient to find a suitable starting seed: $X_{k+1} = 2 X_k - X_k A X_k$
  + particularly useful when dealing w/ families of related matrices
  + probably a good starting point for refining an approximation for the new inverse
  + example: the pair of sequences of inverse matrices used in obtaining matrix square roots by Denman-Beavers iteration
  + probably more than one pass of the iteration to obtain closely enough new matrix
  + useful for "touch up" corrections to Gauss-Jordan algorithm wontainated by small errors due to imperfect computer arithmetic




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







