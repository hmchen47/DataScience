# Eigenvalue algorithm

Organization: Wikipedia

[Origin](https://en.wikipedia.org/wiki/Eigenvalue_algorithm)


## Eigenvalues and eigenvectors

+ Eigenvalue and eigenvector
  + $\exists\; A \in \Bbb{R}^{n \times n}$ or $\Bbb{C}^{n \times n}$
  + $\lambda$: eigenvalue
  + $\bf{v}$: generalized eigenvector
  
    \[ (A - \lambda I)^k = 0 \]

    + $\bf{v}$: aa non-xero $n \times 1$ column vector
    + $I$: the $n \times n$ matrix
    + $k \in \Bbb{R}^+$
    + $\lambda, \bf{v} \in \Bbb{C}$
  + $k=1$: $\bf{v}$ as eigenvector
  + eigenpair: the pair of $\lambda$ and $\bf{v}$ s.t. $A\bf{v} = \lambda \bf{v}$
  + any eigenvalue $\lambda$ of $A$ w/ ordinary eigenvectors associate to it
  + $k$ is the smallest integer s.t. $(A - \lambda I)^k \bf{v} = 0$ for a generalized eigenvector $\bf{v} \implies$ $(A - \lambda I)^{k-1} \bf{v}$ as an ordinary eigenvector, $\forall\; k \le n$
  + $(A - \lambda I)^n \bf{v} = 0 \;\;\forall$ generalized eigenvectors $\bf{v}$ associated w/ $\lambda$

+ Eigenspace
  + __eigenspace__: the kernel $\operatorname{ker}(A - \lambda I)$ consisting of all eigenvectors associated w/ $\lambda$ (along w/ 0) $\forall\; \lambda$ of $A$
  + __generalized eigenspace__: the vector space $\operatorname{ker}((A - \lambda I)^n)$ consisting of all generalized eigenvectors
  + __geometric multiplicity__ of $\lambda$: the dimension of its eigenspace
  + __algebraic multiplicity__ of $\lambda$:
    + the dimension of its generalized eigenspace
    + the characteristic polynomial of $A$

      \[ p_A(z) = \det(zI - A) = \prod_{i=1}^k (z - \lambda_i)^{\alpha_i} \]

      + $\lambda_i$: all the distinct eigenvalues of $A$
      + $\alpha_i$: the corresponding algebraic multiplicities
    + the multiplicity of the eigenvalue as a zero of the characteristic polynomial
    + $\sum_{i=1}^k \alpha_i = n$: the degree of the characteristics polynomial
  + the geometric multiplicity $\le$ the algebraic multiplicity
  + characteristic equation
    + $p_A(z) = 0$
    + roots exactly the eigenvalues of $A$
  + Cayley-Hamilton theorem:
    + $A$ obeying $P_A(A) = 0$
    + the columns of the matrix $\prod_{i \ne j}(A - \lambda_i I)^{\alpha_i} = 0 \text{or }\lambda_j$
    + the column space = the generalized eigenspace of $\lambda_j$

+ Eigenbasis
  + collection of generalized eigenvectors of distinct eigenvalues $\to$ linear independent
  + a basis of all of $\Bbb{C}^n$ chosen consisting of generalized eigenvectors
  + basis $\{\bf{v}_i\}_{i=1}^n$ chosen and organized s.t. 
    + $\bf{v}_i$ and $\bf{v}_j$ w/ the same eigenvalue $\implies \bf{v} \forall\; k \in [i, j]$
    + $\bf{v}_i$ not ordinary eigenvector w/ eigenvalue $\lasmbda_i \implies {A-\lambda_i I) \bf{v}_i = \bf{v}_{i-1}$ 
  + basis vectors placed as the column vectors of a matrix $V = [\bf{v}_1 \bf{v}_2 \cdots \bf{v}_n] \implies V$ used to convert $A$ to its Jorfan normal form

    \[ V^{-1}AV = \begin{bmatrix} \lambda_1&\beta_1&0& \cdots & 0\\ 0&\lambda_2&\beta_2&\cdots&0\\0&0&\lambda_3&\cdots&0\\\vdots&\vdots&\vdots&\ddots&\vdots\\0&0&0&\cdots&\lambda_n \end{bmatrix} \]

    + $\lambda_i$: the eigenvalues
    + $\beta_i = \begin{cases} 1& \text{ if } (A - \lambda_{i+1}) \bf{v}_{i+1} = \bf{v}_i \\ 0 & \text{ otherwise} \end{cases}$
  + generalization:
    + any invertible matrix $W$ and $\lambda$ as an eigenvalue $A$ w/ generalized eigenvector $\bf{v} \implies (W^{-1}AW - \lambda I)^k W^{-k} \bf{v} = 0$
    + $\lambda$: an eigenvalue of $W^{-1}AW$ w/ generalized eigenvector $W^{-1} \bf{v}$
    + similar matrices w/ the same eigenvalues

+ Normal, Hermitian, and real-symmetric matrices
  + adjoint matrix
    + the matrix of cofactors of the transpose
    + $M^\ast$ of a complex matrix $M$ = transpose of the conjugate of $M$: $M^\ast = \overline{M}^T$
  + normal matrix
    + a square matrix $A$ if it commutes w/ its adjoint
    + $A^\ast A = A A^\ast$
  + Hermitian matrix
    + $A^\ast = A$
    + normal matrix
    + $A \in \Bbb{R}^{n \times n} \implies$ the adjoint = its transpose
    + $\iff$ symmetric
  + column vectors: the adjoint used to define the canonical inner product in $\Bbb{C}^n: \bf{w} \cdot \bf{v} = \bf{w}^\ast \bf{v}$
  + properties of normal, Hermitian, and real-symmetric matrices
    + every generalized eigenvector of a normal matrix $\implies$  ordinary eigenvector
    + Jordan normal form w/ diagonal $\implies$ any normal matrix similar to a diagonal matrix
    + eigenvectors of distinct eigenvalues of a normal matrix $\implies$ orthogonal
    + null space and the image (or column space) of a normal matrix $\implies$ orthogonal to each other
    + $\forall$ normal matrix $A, \;\Bbb{C}^n$ w/ an orthonormal basis consisting of eigenvectors of $A \implies$ corresponding matrix of eigenvectors __initary__
    + eigenvalues of a Hermitian matrix $\in \Bbb{R} \implies (\overline{\lambda} - \lambda) \bf{v} = (A^\ast - A) \bf{v} = (A - A)\bf{v} = 0 \;\;\forall\; \bf{v} \ne \bf{0}$
    + $A \in \Bbb{R}^n, \exists$ an orthonormal basis for $\Bbb{R}^n$ consisting of eigenvectors of $A \iff A$ symmetric


## Condition number

+ Condition number
  + the evaluation of some function $f$ for some input $x$
  + $\kappa(f, x)$:
    + the ratio of the relative error in the function's output to the relative error in the input
    + varying w/ both the function and the input
    + how error growing during the calculation
  + base-10 logarithm: how many fewer digits of accuracy exist in the result than existed in the input
  + a best-case scenario: reflecting the instability built into the problem, regardless if how it solved
  + no algorithms able to produce more accurate results than indicated by the condition number, except by chance
  + possibly very ill-conditioned problem to find the roots of a polynomial

+ Operator normal
  + solving the linear equation $A\bf{v} = \bf{b}$ w/ invertible $A$
  + the condition number $\kappa(A^{-1}, \bf{b}) = \|A\|_{\operatorname{op}}\|A^{-1}\|_{\operatorname{op}}$
  + $\|\;\|_{\operatorname{op}}$: the __operator norm__ subordinate to the normal Euclidean nor on $\Bbb{C}^n$
  + $\kappa(A)$ of the matrix $A$
    + $\kappa(A, \bf{b})$ independent of $\bf{b}$ and the same for $A$ and $A^{-1}$
    + the absolute value of the ratio of the largest eigenvalue of $A$ to its smallest
  + unitary $A \implies \|A\|_{\operatorname{op}} = \|A^{-1}\|_{\operatorname{op}} = 1 \implies \kappa(A) = 1$
  + operator norm difficult to calculate for general matrix
  + matrix norm commonly used to estimate the condition number

+ Bauer-Fike theorem
  + $\exists\; A_{n \times n}$ diagonalizable w/ eigenvector matrix $V$
  + $\lambda$ as eigenvalue of $A \implies$ the absolute error in calculating $\lambda$ bounded by the product $\kappa(V)$ and the absolute error in $A$
  + the condition number for finding $\lambda$: $\kappa(\lambda, A) - \|V\|_{\operatorname{op}}\|V^{-1}\|_{\operatorname{op}}$
  + normal $A \implies$ unitary $V$ and $\kappa(\lambda, A) = 1$

+ Eigenspace
  + finding the eigenspace of a normal matrix $A$ corresponding to an eigenvalue $\lambda$
  + condition number inversely proportional to the minimum distance btw $\lambda$ and the other distinct eigenvalues of $A$
  + the eigenspace problem for normal matrices: well-conditioned for isolated eigenvalues
  + not-isolated eigenvalues: identifying the span of all eigenvectors of nearby eigenvalues


## Algorithms

+ General algorithm idea
  + algorithm for finding eigenvalues used to find the roots of polynormials
  + Abel-Ruffini theorem: algorithm w/ dimensions $\ge 4$
    + infinite space
    + involving functions of greater complexity than elementary arithmetic operations and fraction powers
  + algorithms exactly calculating eigenvalues in a finite number of steps only exist for a few special classes of matrix
  + iterative algorithms producing better approximate solutions w/ each iteration for general matrix
  + different algorithms producing different number of eigenvalues: all, a few or even only one
  + identified $\lambda$ used to
    + direct the algorithm toward a different solution next time
    + reduce the problem to one that no longer w/ $\lambda$ as a solution

+ Redirection methods
  + accomplished by shifting
  + replacing $A$ w/ $(A - \mu I) , \exists\; \text{ constant } \mu$
  + eigenvalue found for $(A - \mu I)$ must add back  in to get an eigenvalue for $A$
  + power iteration, $\mu = \lambda$
    + finding the largest eigenvalue in absolute value
    + $\lambda$ only an approximate eigenvalue $\implies$ unlikely to find it a second time
  + inverse iteration
    + finding the lowest eigenvalue
    + $\mu$ chosen well from $\lambda$
    + hopefully closer to some other eigenvalue





## Hessenberg and tridiagonal matrices





## Iterative algorithms





## Direct calculation




