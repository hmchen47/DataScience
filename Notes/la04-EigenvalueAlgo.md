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




## Condition number




## Algorithms





## Hessenberg and tridiagonal matrices





## Iterative algorithms





## Direct calculation




