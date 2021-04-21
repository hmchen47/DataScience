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

+ Cayley-Hamilton method
  + $\det(A), \operatorname{tr}(A),$ and power of $A$ used to express the inverse of $A$

    \[ A^{-1} = \frac{1}{\det(A)} \sum_{s=0}^{n-1} A^s \sum_{k_1, k_2, \dots, k_{n-1}} \prod_{l=1}^{n-1} \frac{(-1)^{k_l + 1}}{l^{k_l} k_l!} \operatorname{tr}(A^l)^{k_l} \]

  + $\operatorname{tr}(A)$: traces of $A$
    + given by the sum of the main diagonal
    + taken over $s$ and the set of all $k_l \ge 0$ satisfying the linear Diophantine equartion

    \[ s + \sum_{l=1}^{n-1} l K_l = n -1 \]

  + rewritten w/ complete Bell polynomials of the arguments $t_l = -(l - 1)!\operatorname{tr}(A^l)$

    \[ A^{-1} = \frac{1}{\det(A)} \sum_{s=1}^n A^{s-1} \,\frac{(-1)^{n-1}}{(n-s)!} \,B_{n-s}(t_1, t_2, \dots, t_{n-s}) \]

+ Eigendecomposition
  + $\exists\; \text{ eigendecomposed } A, \forall\; \lambda \ne 0 \implies \text{ invertible } A$ w/

    \[ A^{-1} = Q\Lambda^{-1}Q^{-1} \]

    + $Q$: square $(n \times n)$ matrix
    + $q_i$: the eigenvector, $i$th column of $Q$
    + $\Lambda$: the diagonal matrix w/ diagonal elements as the corresponding eigenvalues, i.e., $\Lambda_{ii} = \lambda_i$
  + symmetric $A$ and orthogonal $Q$ s.t. $Q^{-1} = Q^T$
  + inverse of diagonal $\Lambda$

    \[ [\Lambda^{-1}]_{ii} = \frac{1}{\lambda_i} \]

+ Cholesky decomposition
  + positive definite $A \implies$ the inverse

    \[ A^{-1} = (L^\ast)^{-1} L^{-1} \]

  + $L$: the lower triangular Cholesky decomposition of $A$
  + $L^\ast$: the conjugate transpose of $L$

+ Analytic solution
  + adjugate matrix: the transpose of the matrix of cofactor
    + efficient way to compute the inverse of small matrix
    + inefficient way to calculate the inverse of large matrix
  + the inverse

    \[ A^{-1} = \frac{1}{|A|} C^T = \frac{1}{|A|} \begin{pmatrix} C_11&C_{21}&\cdots&C_{n1}\\C_{12}&C_{22}&\cdots&C_{n2}\\ \vdots&\vdots&\ddots&\vdots\\C_{1n}&C_{2n}&\cdots&C_{nn} \end{pmatrix} \]

    so that

    \[ (A^{-1})_{ij} = \frac{1}{|A|}\left(C^T\right)_{ij} = \frac{1}{|A|} (C_{ji}) \]

    + $|A|$: the determinant of $A$
    + $C$: the matrix of cofactors
    + $C^T$: transpose matrix

+ Inversion of 2x2 matrices
  + the inverse of $A$

    \[ A^{-1} = \begin{bmatrix} a&b\\c&d \end{bmatrix}^{-1} = \frac{1}{\det A} \begin{bmatrix} d&-b\\-c&a \end{bmatrix} = \frac{1}{ad-bc} \begin{bmatrix} d&-b\\-c&a \end{bmatrix} \]

  + the Cayley-Hamilton method

    \[ A^{-1} = \frac{1}{\det A} [(\operatorname{tr} A)I - A] \]

+ Inversion of 3x3 matrices
  + the inverse of $A$

    \[ A^{-1} = \begin{bmatrix} a&b&c\\d&e&f\\g&h&i \end{bmatrix} = \frac{1}{\det A} \begin{bmatrix} A&B&C\\D&E&F\\G&H&I \end{bmatrix}^T = \frac{1}{\det A} \begin{bmatrix} A&D&G\\B&E&H\\C&F&I \end{bmatrix} \]

  + the elements of the intermediary matrix above (RHS)

    \[\begin{array}{crcrcr}
      A = &  (ei-fh), & D = & -(bi - ch), & G = &  (bf - ce), \\
      B = & -(di-fg), & E = &  (ai - cg), & H = & -(af - cd), \\
      C = &  (dh-eg), & F = & -(ah - bg), & I = &  (ae - bd). 
    \end{array}\]

  + applying the rule of Sarrus to get $\det(A)$

    \[ \det(A) = aA + bB + cC \]

  + the Cayley-Hamilton decomposition

    \[ A^{-1} = \frac{1}{\det(A)} \left(\frac12 \left[(\operatorname{tr}A)^2 - \operatorname{tr} A^2 \right] I - A \operatorname{tr} A + A^2 \right) \]

  + general 3x3 inverse expressed concisely in terms of the cross product and triple product
    + $\exists\; A = [\bf{x}_0 \bf{x}_1 \bf{x}_2]$ is invertible

      \[ A^{-1} = \frac{1}{\det (A)} \begin{bmatrix} (\bf{x}_1 \times \bf{x}_2)^T\\(\bf{x}_2 \times \bf{x}_0)^T\\(\bf{x}_0 \times \bf{x}_1)^T \end{bmatrix}\]

    + $\det (A)$ equal to the triple product of $\bf{x}_0, \bf{x}_1, \bf{x}_2$
    + the volume of the parallelpiped of formed by the rows or columns

      \[ \det(A) = \bf{x}_0  \cdot (\bf{x}_1  \times \bf{x}_2) \]

    + correctness of the formula checked by using cross- and triple-product properties
    + left and right inverse always coincide
    + each row of $A^{-1}$: orthogonal to the non-corresponding two columns of $A$
    + $\det(A)$ causing the diagonal elements of $I = A^{-1}A$ to be unitary

+ Inversion of 4x4 matrices
  + expressions for the inverse of $A$ get complicated
  + the Cayley-Hamilton method w/ $n = 4$

    \[ A^{-1} = \frac{1}{\det (A)} \left( \frac16 \left[(\operatorname{tr} A)^3 - 3 \operatorname{tr} A \operatorname{tr} A^2 + 2 \operatorname{tr} A^3 \right] I - \frac12 A \left[(\operatorname A)^2 - \operatorname{tr} A^2 \right] + A^2 \operatorname{tr} A - A^3 \right) \]

+ Blockwise inversion
  + analytic inversion formula

    \[ \begin{bmatrix} A&B\\C&D \end{bmatrix} = \begin{bmatrix} A^{-1} + A^{-1} B(D - CA^{-1} B)^{-1} & -A^{-1}B(D-CA^{-1}B)^{-1}\\ -(D-CA^{-1}B)^{-1}CA^{-1} & (D - CA^{-1}B)^{-1} \end{bmatrix} \tag{1} \]

    + $A, B, C, D$: matrix sub-blocks of arbitrary size
    + $A, D - CA^{-1}B$: non-singular
    + advantageous w/ diagonal $A$ and small $(D - CA^{-1}B)$
  + nullity theorem
    + the nullity of $A$ = the nullity of the sub-block in the lower right of the inverse matrix
    + the nullity of $B$ = the nullity of the sub-block in the upper right of the inverse matrix
  + the inversion procedure
    + Eq. (1) performed matrix block operations operated on $C$ and $D$ first
    + $A$ and $B$ operated on first, and provided $D$ and $A - BD^{-1}C$ are non-singular

      \[ \begin{bmatrix}A&B\\C&D \end{bmatrix}^{-1} = \begin{bmatrix} (A - BD^{-1}C)^{-1}&-(A - BD^{-1}C)^{-1}BD^{-1}\\-D^{-1}C(A - BD^{-1}C)^{-1}&D^{-1}+D^{-1}C(A - BD^{-1}C)^{-1}BD^{-1} \end{bmatrix} \tag{2} \]

    + Eq. (1) = Eq. (2) $\implies$

      \[\begin{array}{rcl} \tag{3}
        (A - BD^{-1}C)^{-1} &=& A^{-1} + A^{-1}B(D - CA^{-1}B)^{-1}CA^{-1}\\
        (A - BD^{-1}C)^{-1}BD^{-1} &=& A^{-1}B(D - CA^{-1}B)^{-1}\\
        D^{-1}C(A - BD^{-1}C)^{-1} &=& (D - CA^{-1}B)^{-1}CA^{-1}\\
        D^{-1}+D^{-1}C(A - BD^{-1}C)^{-1}BD^{-1} &=& (D - CA^{-1}B)^{-1}
      \end{array}\]

    + binomial inverse theorem: Eq. (3) = the Woodbury matrix identity
  + invertible $A, D \implies$

    \[ \begin{bmatrix} A&B\\C&D \end{bmatrix}^{-1} = \begin{bmatrix} (A - BD^{-1}C)^{-1}&0\\0&(D - CA^{-1}B)^{-1} \end{bmatrix} \begin{bmatrix} I&-BD^{-1}\\-CA^{-1}&I \end{bmatrix} \tag{2} \]

  + the Weinstein-Aronszajn identity: one of the two matrices in the block-diagonal matrix is invertible exactly when the other is
  + inversion of $A_{n \times n}$ requiring inversion of two half-sized matrices and 6 multiplications btw 2 half-sized matrices
    + divided and conquer algorithm: using blockwise inversion to invert a matrix
    + matrix multiplication algorithm w/ same time complexity as divided and conquer algorithm
  + matrix multiplication algorithm:
    + time complexity: $\mathcal{O}(n^{2.3727})$ operations
    + the best proven lower bound: $\Omega(n^2 \log n)$
  + zero upper right block matrix $B \to$ simplify the formula significantly

    \[ \begin{bmatrix} A&0\\C&D \end{bmatrix}^{-1} = \begin{bmatrix} A^{-1}&0\\-D^{-1}CA^{-1}&D^{-1} \end{bmatrix} \]

+ Neumann series
  + $\exists\; A$ w/ $\displaystyle\lim_{n \to \infty} (I - A)^n = 0 \implies$ inverse $A$ expressed by a Neumann series

    \[ A^{-1} = \sum_{n=0}^\infty (I - A)^n \]

  + preconditioner: truncating the sum results in an "approximate" inverse
  + geometric sum: truncated Neumann series accelerated exponentially by satisfying

    \[ \sum_{n=0}^{2^L -1} (I -A)^n = \prod_{l=0}^{L-1} \left( I + (I - A)^{2^l} \right) \]

  + $\therefore\;$ only $2L -2$ matrix multiplication required to compute $2^L$ terms of the sum
  + generalization: $A$ near the invertible matrix $X$ in the sense that

      \[ \lim_{n \to \infty} (I - X^{-1} A)^n = 0 \hspace{1em}\text{ or }\hspace{1em} \lim_{n \to \infty} (I - A X^{-1})^n = 0 \]

    $\implies A$ nonsingular w/ inverse

      \[ A^{-1} = \sum_{n \to \infty} \left(X^{-1}(X - A)\right)^n X^{-1} \]

  + $A - X$ w/ rank 1 $\implies$

    \[ A^{-1} = X^{-1} = \frac{X^{-1} (A - X) X^{-1}}{1 + \operatorname{tr}\left(X^{-1}(A - X)\right)} \]

+ p-adic approximation
  + $A$ w/ integer and rational coefficients and $\exists\;$ a solution in arbitrary-precision rationals $\implies$ a $p$-adic approximation methods converges to an exact solution in $\mathcal{O}(n^4 \log^2 n)$
  + relying on solving $n$ linear system via Dixon's method of $p$-adic approximation



### Reciprocal basis vectors method






## Derivative of the matrix inverse






## Generalized inverse





## Applications




### Regression/least square





### Matrix inverses in real-time simulations





### Matrix inverses in MIMO wireless communication







