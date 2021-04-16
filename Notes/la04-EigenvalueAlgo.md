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
    + the column space = the generalized eigenspace $\lambda_j$






## Condition number




## Algorithms





## Hessenberg and tridiagonal matrices





## Iterative algorithms





## Direct calculation




