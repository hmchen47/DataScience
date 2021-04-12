# Eigenvalues and eigenvectors

Organization: Wikipedia

[Origin](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)


## Formal definition

+ Formal definition
  + $T$: a linear transformation from a vector space $V$ over a field $F$ into itself
  + $\bf{v}$: a non-zero vector in $V$
  + $T(\bf{v})$ a scalar multiple of $\bf{v} \implies \bf{v}$ is an eigenvector of $T$

    \[ T(\bf{v}) = \lambda v \]

  + $\lambda$: a scalar in $F$
    + eigenvalue
    + characteristic value
    + characteristic root
  + $A_{n \times n} \leftrightarrow$ linear transformations from same $n$-dim vector space into itself, given any basis of the vector space

+ Finite-dim vector space
  + eigenvalues and eigenvectors in matrices = eigenvalues and eigenvectors in linear transformation
  + $V$: finite-dimensional
  + $A$: the matrix representation of $T$
  + $\bf{u}$: the coordinate vector of $\bf{v}$

  \[ T(\bf{v}) = \lambda v \hspace{1em}\longrightarrow\hspace{1em} A\bf{u} = \lambda\bf{u} \]


## Overview

+ Name origin
  + the prefix `eigen-`: proper, characteristic, own
  + originally used to study principal axes of the rotational motion of rigid bodies
  + applications: stability analysis, vibration analysis, atomic orbitals, facial recognition, and matrix diagonalization

+ Mathematical representation
  + an eigenvector $\bf{v}$ of a linear transformation $T$: a nonzero vector not changing direction when $T$ applied to it
  + applying $T$ tot the eigenvector only scales the eigenvector by the scalar value $\lambda$
  + $\lambda$: eigenvalue, any scalar
  + eigenvalue equation or eigenequation: $T(\bf{v}) = \lambda \bf{v}$

+ Linear transformation
  + mapping vectors in a variety of vector spaces $\implies$ eigenvectors tabling many forms
  + differential operator, $\frac{d}{dx}$
    + eigenvectors as function, called eigenfucntions
    + scaled by the differential operator

      \[ \frac{d}{dx} e^{\lambda x} = \lambda e^{\lambda x} \]

  + linear transformation of differential operator
    + the form of and $n \times n$ matrix
    + eigenvectors: $n \times 1$ matrix
    + eigenvalue equation expressed as the matrix multiplication w/ $\bf{v}$ as an $n \times 1$ matrix

      \[ A \bf{v} = \lambda \bf{v} \]

    + eigenvalues and eigenvectors used to decompose the matrix

+ Related math concepts
  + __eigensystem:__ the set of all eigenvectors of a linear transformation, each paired w/ its corresponding eigenvalue
  + __eigenspace__ or __characteristic space__ of $T$: the set of all eigenvectors of $T$ corresponding to the same eigenvalue, together w/ the zero vector
  + __eigenbasis:__ a set of eigenvectors of $T4 forms a basis f the domain of $T$


## Eigenvalues and eigenvector of matrices

+ Eigenvalues and eigenvectors of matrices
  + consider $n$-dim vectors formed as a list of $n$ scalar, e.g., 

    \[ \bf{x} = \begin{bmatrix} 1&-3&4 \end{bmatrix} \hspace{0.5em}\text{and}\hspace{0.5em} \bf{y} = \begin{bmatrix} -20&60&-80 \end{bmatrix} \]

  + scalar multiplier of each other, or parallel, or colinear
    + $\bf{x}$ and $\bf{y}$ of the above example
    + $\exists\; \lambda$ s.t. $\bf{x} = \lambda \bf{y}$

+ Linear transformation, eigenvalues and eigenvectors
  + linear transformation of $n$-dimensional vectors defined by $A_}n \times n}$

    \[ A \bf{v} = \bf{w} \hspace{0.5em}\longleftrightarrow\hspace{0.5em} \begin{bmatrix} A_{11}&A_{12}&\cdots&A_{1n}\\A_{21}&A_{22}&\cdots&A_{2n}\\ \vdots&\vdots&\ddots&\vdots\\A_{n1}&A_{n2}&\cdots&A_{nn} \end{bmatrix} \begin{bmatrix} v_1\\v_2\\ \vdots \\v_n \end{bmatrix} = \begin{bmatrix} w_1\\w_2\\\vdots\\w_n \end{bmatrix} \tag{1}\]

    + each row: $w_i = A_{i1}v_1 + A_{i2}v_2 + \cdots + A_{in}v_n = \sum_{j=1}^n A_{ij}v_j$

  + $v, w$: scalar multiplies
  + $A\bf{v} = \bf{w} = \lambda\bf{v} \implies \bf{v}$ as an __eigenvector__ of the linear transformation $A$
  + $\lambda$: __eigenvalue__ corresponding to that eigenvector
  + __eigenvalue equation:__ $A\bf{v} = \bf{w} = \lambda\bf{v} \hspace{0.5em}\longrightarrow\hspace{0.5em} (A - \lambda I)\bf{v} = \bf{0}$


## Eigenvalues and characteristic polynomial

+ Characteristic polynomial
  + eigenvalue equation:

    \[ (A - \lambda I)\bf{v} = \bf{0} \tag{2} \]

  + nonzero solution $v \iff \det((A - \lambda I)) = 0$
  + $\lambda$: the eigenvalues of $A$
  + determinant of Eq.(2)
  
    \[ \det(A - \lambda I) = 0 \tag{3}\]

    + polynomial function of the variable $\lambda$ 
    + characteristics polynomial of $A$
    + characteristic equation or the secular equation of $A$
  + $n$: degree of the polynomial = the order of the matrix $A$

+ Fundamental theorem of algebra
  + characteristic polynomial of $A_{n \times n}$ factored into the product of $n$ linear terms

    \[ |A - \lambda I| = (\lambda_1 - \lambda)(\lambda_2 - \lambda) \cdots (\lambda_n - \lambda) \tag{4}\]

  + $\lambda_i \in \Bbb{R}/\Bbb{C}$
  + all entries $a_{ij} \in \Bbb{R} \implies$ the coefficients of the polynomial $\in \Bbb{R}$ and eigenvalues $\lambda_i \in \Bbb{C}$
  + example: consider

    \[ A = \begin{bmatrix} 2&1\\1&2 \end{bmatrix} \]

    + the characteristic polynomial of $A$

      \[ |A - \lambda I| = \begin{vmatrix} 2-\lambda&1\\1&2-\lambda \end{vmatrix} = 3 - 4\lambda + \lambda^2 \]

    + characteristic polynomial = 0 $\implies \lambda = 1$ and $\lambda = 3$
    + eigenvectors: any nonzero multiplies of

      \[ \bf{v}_{\lambda=1} = \begin{bmatrix} 1\\-1 \end{bmatrix}, \hspace{1em} \bf{v}_{\lambda=3} = \begin{bmatrix} 1\\1 \end{bmatrix} \]


## Algebraic multiplicity

+ Algebraic multiplicity
  + $\lambda_i$: an eigenvalue of $A_{n \times n}$
  + definition:
    + $\mu_A(\lambda_i)$ of the eigenvalue
    + its multiplicity as a root of the characteristic polynomial
    + the largest integer $k$ s.t. $(\lambda - \lambda_i)^k$ divides evenly that polynomial
  + suppose $A$ w/ dimension $n$ and $d \le n$ distinct eigenvalues
  + the characteristic polynomial raised to the power of the algebraic multiplicity

    \[ |A - \lambda I| = (\lambda_1 -\lambda)^{\mu_A(\lambda_1)} (\lambda_2 -\lambda)^{\mu_A(\lambda_2)} \cdots (\lambda_d -\lambda)^{\mu_A(\lambda_d)} \]
  
  + $d = n$
    + same as Eq. (4)
    + the size of each eigenvalue's algebraic multiplicity related to the dimensional $n$ as $1 & \le \mu_A(A\lambda_i) \le n$

      \[ \mu_A = \sum_{i=1}^d \mu_A(\lambda_i) = n \]
  
  + simple eigenvalue: $\mu_A(\lambda_i) = 1$
  + semisimple eigenvalue $\nu_A(\lambda_i)$: $\mu_A(\lambda_i)$ equals to geometric multiplicity of $\lambda_i$


## Eigenspaces, geometric multiplicity, and the eigenbasis for matrices

+ Eigenspaces
  + given a particular eigenvalue $\lambda$ of $A_{n \times n}$
  + __eigensoace__ or __characteristic space__ of A associated w/ $\lambda$: define the set $E$ to be all vectors $\bf{v}$ satisfying Eq. (2)

    \[ E = \{\bf{v} : (A - \lambda I) \bf{v} = 0 \} \]

  + properties of $E$
    + precisely the kernel or nullspace of the matrix $(A - \lambda I)$
    + any non-zero vector satisfying the condition is an eigenvector of $A$ associated w/ $\lambda$
    + the union of the zero vector w/ the set of all eigenvectors of $A$ associated w/ $\lambda$
    + equal to nullspace of $(A-\lambda I)$
  + generalization:
    + $\lambda \in \Bbb{C}$ and eigenvectors $\bf{\lambda}_{n \times 1}$ w/ $\lambda_i \in \Bbb{C}$
    + nullspaces as a linear subspace $\implies$ $E$ as a linear subspace of $\Bbb{C}^n$
  + close under addition
    + eigenspace $E$ as a linear subspace
    + $\bf{u, v} \in E \implies (\bf{u + v}) \in E$
    + equivalently $A(\bf{u + v)} = \lambda(\bf{u + v})$
    + proved w/ distributive property of matrix multiplication
  + closed under scalar multiplication
    + $\bf{v} \in E$ and $\alpha \Bbb{C}$
    + $(\alpha \bf{v}) \in E$
    + equivalently $A(\alpha \bf{v}) = \lambda(\alpha \bf{v})$
    + proved by multiplication of complex matrices by complex number is commutative
    + $\bf{u + v}$ and $\alpha \bf{v}$ not zero, eigenvectors of $A$ associated w/ $\lambda$

+ Geometric multiplicity
  + $\nu_A(\lambda)$:
    + the eigenvalue's geometric multiplicity
    + the dimension of eigenspace $E$ associated w/ $\lambda$
    + equivalently the maximum number of linear independent eigenvectors associated w/ $\lambda$
  + $E$: the nullspace of $(A - \lambda I)$
    + the geometric multiplicity of $\lambda$ =  the dimension of the nullspace of $(A - \lambda I)$
    + the nullity of $(A - \lambda I)$
    + rank of $(A - \lambda I)$ as $\gamma_A(\lambda) = n - \text{rank}(A - \lambda I)$
  + an eigenvalue's geometric multiplicity must be at least one, i.e., each eigenvalue has at least one associated eigenvector
  + an eigenvalue's geometric multiplicity unable to exceed its algebra multiplicity
  + an eigenvalue's geometric multiplicity unable to exceed $n$

    \[ 1 \le \gamma_A(\lambda) \le \mu_A(\lambda) \le n \]

  + total geometric multiplicity
    + suppose $A$ w/ $d \le n$ distinct eigenvalues $\lambda_1, \dots, \lambda_d$
    + $\gamma_A(\lambda_i)$: the geometric multiplicity
    + total geometric multiplicity

      \[ \gamma_A = \sum_{i=1}^d \gamma_A(\lambda)i), \hspace{1em} d \le \gamma_A \le n \]

    + the dimension of the sum of all he eigenspaces of $A$'s eigenvalues
    + equivalently the maximum number of linearly independent eigenvectors of $A$
  + $\gamma_A = n \implies$
    + the direct sum of the eigenspaces of all of $A$'s eigenvalues is the entire vector space $\Bbb{C}^n$
    + eigenbasis: a basis of $\Bbb{C}^n$ able to be formed from $n$ linearly independent eigenvectors of $A$
    + any vector in $\Bbb{C}^n$ able to written as a linear combination of eigenvectors of $A$

## Properties of eigenvalues

+ Properties of eigenvalues
  + $A_{n \times n}$ an arbitrary matrix of complex numbers w/ eigenvalues $\lambda_1, \dots, \lambda_n$
  + $\mu_A(\lambda_i)$: the eigenvalues's algebraic multiplicity
  + trace and eigenvalues: the sum of its diagonal elements = the sum of eigenvalues

    \[ \operatorname{tr}(A) = \sum_{i=1}^n a_ii = \sum_{i=1}^n \lambda_i = \lambda_1 + \lambda_2 + \cdots + \lambda_n \] 

  + the determinant pf $A$ = the product of all its eigenvalues

    \[ \det(A) = \prod_{i=1}^n \lambda_i = \lambda_i \lambda_2 \cdots \lambda_n \]

  + the eigenvalues of the $k^{th}$ power of $A$, i.e., the eigenvalues of $A^k, \;\forall k > 0$ are $\lambda_1^k, \dots, \lambda_n^k$
  + invertible $A \iff \lambda_i \ne 0, \;\forall i$
  + invertible $A \implies$ eigenvalues of $A^{-1}$ are $\frac{1}{\lambda_1}, \cdots, \frac{1}{\lambda_n}$ and each eigenvalues's geometric multiplicity coincides
  + $A = A^\star$ or $A$ is Hermitian $\implies$ $\lambda_i \in \Bbb{R}$, where $A^\star$ is conjugate transpose of $A$, also applied to any symmetric real matrix
  + $A$ is Hermitian and positive-definite, positive-semi-definite, negative-definite, or negative-semi-definite $\implies$ $\lambda_i > 0, \ge 0, < 0, \le 0$, respectively
  + $A$ is unitary $\implies |\lambda_i| = 1$
  + $A_{n \times n}$ with eigenvalues $\{\lambda_1, \dots, \lambda_k\} \implies$ $I + A$ w/ eigenvalues $\{\lambda_1+1, \dots, \lambda_k+1\}$
    + $\exists\;\alpha \in \Bbb{C}$, $(\alpha I + A)$ w/ eigenvalues $\{\lambda_1+\alpha, \dots, \lambda_k+\alpha\}$
    + generalization: $\exists\; P \implies P(A)$ w/ eigenvalues $\{P(\lambda_1), \dots, P(\lambda_k)\}$


## Left and right eigenvectors

+ Eigenvectors
  + vector: matrix w/ a single column 
  + right eigenvector
    + eigenvector always refers to a __right eigenvector__
    + a column vector that right multiplies $A_{n \times n}$ in the defining equation, Eq.(1), $A\\bf{v} = \lambda\bf{v}$
  + left eigenvector
    + eigenvalue and eigenvector able to be defined for _row_ vectors
    + left multiply matrix $A$
    + the defining equation

      \[\bf{u}A = \kappa\bf{u} \]

      + $\kappa$: a scalar
      + $\bf{u}$: a $1 \times n$ matrix
    + __left eigenvecctor__: any row vector satisfying the above equation
    + equivalently, $A^T \bf{u}^T = \kappa \bf{u}^T$

## Diagonalization and the eigendecomposition

+ Eigendecomposition
  + the eigenvectors of $A$ form a basis
  + $A$ w/ $n$ linearly independent eigenvectors $\bf{v}_1, \bf{v}_2, \dots, \bf{v}_n$ w/ associated eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$
  + define a square matrix $Q$ w/ columns as the $n$ linearly independent eigenvectors of $A$

    \[ A = [ \bf{v}_1, \bf{v}_2, \dots, \bf{v}_n] \]

  + each column of $Q$ as an eigenvector of $A$, right multiplying $A$ by $Q$ scales each column of $Q$ by its associated eigenvalue

    \[ AQ = [\lambda_1\bf{v}_1, \lambda_2\bf{v}_2, \dots, \lambda_n \bf{v}_n] \]

  + define a diagonal matrix $\Lambda$ w/ diagonal element $\Lambda_{ii}$ as the eigenvalue associated w/ the $i$th column of $Q \implies AQ = QA$
  + the columns of $Q$ as linearly independent $\implies$ $Q$ invertible

    \[ A = Q\Lambda Q^{-1}, \hspace{1em} Q^{-1}AQ = \Lambda \]

  + __decomposition__
    + $A$ decomposed into a matrix composed of its eigenvectors
    + the inverse of the matrix of eigenvectors
    + a similarity transformation
  + such a matrix $A$ to be similar to the diagonal matrix or diagonalizable
  + the matrix $Q$ as the change of basis matrix of the similarity transformation
  + the matrices $A$ and $\Lambda$ representing the same linear transformation expressed into two different bases
  + the eigenvectors used as the basis when representing the linear transformation as $\Lambda$


## Variational characterization

+ Min-max theorem
  + $A$: Hermitian matrix
  + eigenvalue given a variational characterization
  + the largest eigenvalue of $H$ = the maximum value of the quadratic from $\bf{x}^TH\bf{x}/\bf{x}^T\bf{x}$
  + the maximum value of $\bf{x}$ = an eigenvector

## Eigenvector-eigenvalue identity

+ Eigenvalue-eigenvector identity
  + Hermitian matrix
  + the norm square of the $j$th component of a normalized eigenvector
    + calculated by only the matrix eigenvalues and the eigenvalues of the corresponding minor matrix
    + $M_j$: the submatrix formed by removing the $j$th row and column from the original matrix

    \[ |v_{i, j}|^2 = \frac{\prod_k (\lambda_i - \lambda_k(M_j))}{\prod_{k\ne i} (\lambda_i - \lambda_k)} \]

## Examples

+ Two-dimensional matrix
  + consider the matrix

    \[ A = \begin{bmatrix} 2&2\\1&2 \]

  + the effect of transformation on point coordinates in the plane
    + the eigenvectors $v$ of the transformation satisfy Eq. (1)
    + the values of $\lambda$ for which $\det(A - \lambda I) = 0$ as eigenvalues

    <figure style="margin: 0.5em; text-align: center;">
      <img style="margin: 0.1em; padding-top: 0.5em; width: 15vw;"
        onclick= "window.open('https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Eigenvector-eigenvalue_identity')"
        src    = "https://upload.wikimedia.org/wikipedia/commons/0/06/Eigenvectors.gif"
        alt    = "The transformation matrix A = {\displaystyle \left[{\begin{smallmatrix}2&1\\1&2\end{smallmatrix}}\right]}{\displaystyle \left[{\begin{smallmatrix}2&1\\1&2\end{smallmatrix}}\right]} preserves the direction of purple vectors parallel to vλ=1 = [1 −1]T and blue vectors parallel to vλ=3 = [1 1]T. The red vectors are not parallel to either eigenvector, so, their directions are changed by the transformation. The lengths of the purple vectors are unchanged after the transformation (due to their eigenvalue of 1), while blue vectors are three times the length of the original (due to their eigenvalue of 3)."
        title  = "The transformation matrix A = {\displaystyle \left[{\begin{smallmatrix}2&1\\1&2\end{smallmatrix}}\right]}{\displaystyle \left[{\begin{smallmatrix}2&1\\1&2\end{smallmatrix}}\right]} preserves the direction of purple vectors parallel to vλ=1 = [1 −1]T and blue vectors parallel to vλ=3 = [1 1]T. The red vectors are not parallel to either eigenvector, so, their directions are changed by the transformation. The lengths of the purple vectors are unchanged after the transformation (due to their eigenvalue of 1), while blue vectors are three times the length of the original (due to their eigenvalue of 3)."
      />
    </figure>

  + the determinant of $(A - \lambda I)$

    \[\begin{align*}
      |A - \lambda I| &= \left| \begin{bmatrix} 2&1\\1&2 \end{bmatrix} - \lambda \begin{bmatrix} 1&0\\0&1 \end{bmatrix} \right| = \begin{vmatrix} 2-\lambda&1\\1&2-\lambda \end{vmatrix} \\\\
      &= 2 - 4\lambda + \lambda^2
    \end{align*}\]

  + the characteristic polynomial = 0 $\implies \lambda = 1$ and $\lambda = 3$ as eigenvalues
  + $\lambda = 1$, Eq. (2)

    \[\begin{align*}
      &(A - \lambda I) \bf{v}_{\lambda=1} = \begin{bmatrix} 1&1\\1&1 \end{bmatrix} \begin{bmatrix} v_1\\v_2 \end{bmatrix} = \begin{bmatrix} 0\\0 \end{bmatrix} \\
      & 1v_1 + 1 v_2 = 0; \hspace{0.5em} 1v_1 + 1v_2 = 0 \\
      & \therefore\; \bf{v}_{\lambda=1} = \begin{bmatrix} v_1\\v_2 \end{bmatrix} = \begin{bmatrix} 1\\-1 \end{bmatrix}
    \end{align*}\]

  + $\lambda = 3$, Eq. (2)

    \[\begin{align*}
      &(A - 3\lambda I) \bf{v}_{\lambda=1} = \begin{bmatrix} -1&1\\1&-1 \end{bmatrix} \begin{bmatrix} v_1\\v_2 \end{bmatrix} = \begin{bmatrix} 0\\0 \end{bmatrix} \\
      & -1v_1 + 1 v_2 = 0; \hspace{0.5em} 1v_1 - 1v_2 = 0 \\
      & \therefore\; \bf{v}_{\lambda=3} = \begin{bmatrix} v_1\\v_2 \end{bmatrix} = \begin{bmatrix} 1\\1 \end{bmatrix}
    \end{align*}\]

+ Three-dimensional matrix
  + consider the matrix

    \[ A = \begin{bmatrix} 2&0&0\\0&3&4\\0&4&9 \end{bmatrix} \]

  + the characteristic polynomial of $A$

    \[\begin{align*}
      |A - \lambda I| &= \left| \begin{bmatrix} 2&0&0\\0&3&4\\0&4&9 \end{bmatrix} - \lambda \begin{bmatrix} 1&0&0\\0&1&0\\0&0&1 \end{bmatrix} \right| = \begin{vmatrix} 2-\lambda&0&0\\0&3-\lambda&p\\0&4&9-\lambda \end{vmatrix} \\
      &= (2-\lambda) [(3 - \lambda)(9-\lambda) - 16] = -\lambda^3+14\lambda^2-35\lambda+22
    \end{align*}\]

  + the roots of the characteristic polynomial = eigenvalues: 2, 1, and 11
  + the corresponding to the eigenvectors

    \[ \bf{v}_{\lambda = 2} = \begin{bmatrix} 1\\0\\0 \end{bmatrix}, \hspace{1em} \bf{v}_{\lambda = 1} = \begin{bmatrix} 0\\-2\\1 \end{bmatrix}, \hspace{1em} \bf{v}_{\lambda = 11} = \begin{bmatrix} 0\\1\\2 \end{bmatrix} \]

+ Three-dimensional matrix w/ complex eigenvalues
  + consider the cyclic permutation matrix

    \[ A = \begin{bmatrix} 0&1&0\\0&0&1\\1&0&0 \end{bmatrix} \]

  + the characteristic polynomial: $1 - \lambda^3$
  + the roots = eigenvalues: $\lambda_1 = 1, \hspace{1em} \lambda_2 = -\frac12 + i\frac{\sqrt{3}}{2}, \hspace{1em} \lambda_3 = \lambda_2^\star = -\frac12 + -i \frac{\sqrt{3}}{2}$
  + the real eigenvalue $\lambda_1 = 1$, any vector w/ 3 equal nonzero entries is an eigenvector
  + the complex conjugate pair of imaginary eigenvalues

    \[\begin{align*}
      &\lambda_2 \lambda_3 = 1, \hspace{1em} \lambda_2^2 = \lambda_3, \hspace{1em} \lambda_3^2 = \lambda_2\\\\
      &A \begin{bmatrix} 1\\ \lambda_2\\ \lambda_3 \end{bmatrix} = \begin{bmatrix} \lambda_2\\ \lambda_3 \\1 \end{bmatrix} = \lambda_2 \cdot \begin{bmatrix} 1\\ \lambda_2 \\ \lambda_3 \end{bmatrix}, \hspace{1em} A\begin{bmatrix} 1\\ \lambda_3\\ \lambda_2 \end{bmatrix} = \begin{bmatrix} \lambda_3\\ \lambda_2\\ 1\end{bmatrix} = \lambda_2 \cdot \begin{bmatrix}1 \\ \lambda_3 \\ \lambda_2 \end{bmatrix}
    \end{align*}\]

  + the other two eigenvectors of $A$ are complex
  
    \[\bf{v}_{\lambda_2} = \begin{bmatrix} 1\\ \lambda_2 \\ \lambda_3 \end{bmatrix}, \hspace{1em} \bf{v}_{\lambda_3} = \begin{bmatrix} 1\\ \lambda_3 \\ \lambda_2 \end{bmatrix} \]

+ Diagonal matrix
  + diagonal matrices: matrices w/ entries only along the main diagonal
  + eigenvalues of a diagonal matrix = the diagonal elements themselves
  + consider the matrix

    \[ A = \begin{bmatrix} 1&0&0\\0&2&0\\0&0&3 \end{bmatrix} \]
  
  + the characteristic polynomial of $A$: $|A - \lambda I| = (1 - \lambda)(2 - \lambda)(3 - \lambda)$
  + the roots of characteristic polynomial = eigenvalues: $\lambda_1 = 1, \lambda_2 = 2, \lambda_3$
  + each diagonal element corresponds to an eigenvector whose only nonzero component is in the same row as the diagonal element

+ Triangular matrix
  + lower triangular matrix: a diagonal matrix w/ elements above main diagonal = 0
  + upper triangular matrix: a diagonal matrix w/ elements below main diagonal = 0
  + the eigenvalues of triangular matrices = the elements of the main diagonal
  + consider the lower triangular matrix

    \[ A = \begin{bmatrix} 1&0&0\\1&2&0\\2&3&3 \end{bmatrix} \]

  + the characteristic polynomial of $A$

    \[ |A - \lambda I| = (1 - \lambda)(2 - \lambda)(3-\lambda) \]

  + the roots of the characteristic polynomial: $\lambda_1 = 1, \lambda_2 = 2, \lambda_3 = 3$
  + the corresponding eigenvectors

    \[ \bf{v}_{\lambda_1} = \begin{bmatrix} 1\\-1\\1/2 \end{bmatrix}, \hspace{1em} \bf{v}_{\lambda_2} = \begin{bmatrix} 0\\1\\-3 \end{bmatrix}, \hspace{1em} \hspace{1em} \begin{bmatrix} 0\\0\\1 \end{bmatrix} \]

+ Matrix w/ repeated eigenvalues
  + consider the lower triangular matrix

    \[ A = \begin{bmatrix} 2&0&0&0\\1&2&0&0\\0&1&3&0\\0&0&1&3 \end{bmatrix} \]

  + the characteristic polynomial of $A$

    \[ |A - \lambda I | = \begin}vmatrix} 2-\lambda&0&0&0\\1&2-\lambda&0&0\\1&2-\lambda&0&0\\0&1&3-\lambda&0\\0&0&1&3-\lambda \end{vmatrix} = (2-\lambda)^2(3-\lambda)^2 \]

  + the roots of the characteristic polynomial: $\lambda_1 = 2, \lambda_2 = 3$
  + the algebraic multiplicity of each eigenvalue: 2
  + both double roots $\implies$ the sum of the algebraic multiplicity of all eigenvalues: $\mu_A = 4 = n$, the order of the characteristic polynomial and the dimension of $A$
  + the geometric multiplicity of the eigenvalue 2: 1 
    + the only eigenvector in the eigenspace: $[0 1 -1 1]^T$
    + one dimension
  + the geometric multiplicity of the eigenvalues 3: 1
    + the only eigenvector in eigenspace: $[0 0 0 1]^T$
    + one dimension
  + total geometric multiplicity: $\gamma_A = 2$


## General formula for eigenvalues of a 2-dim matrix

Consider a real matrix A. The eigenvalues $\lambda$ of $A$ are

\[ A = \begin{bmatrix} a&b\\c&d \end{bmatrix} \implies \lambda = \frac{1}{2} (a + d) \pm \sqrt{\left(\frac12(a - d)\right)^2 + bc} = \frac12 \left(\operatorname{tr}(A) \pm \sqrt{(\operatorname{tr}(A^2)) - 4 \det(A)} \right) \]


## General definitions

+ Eigenvector
  + $V$: any vector space over some field $K$ of scalar
  + $T$: a linear transformation mapping from $V$ into $V$

    \[ T: V \to V \]

  + __eigenvector:__ a nonzero vector $\bf{v} \in V$ of $T \iff \exists\; \lambda \in K$ s.t. 
  
    \[ T(\bf{v}) = \lambda \bf{v} \tag{5} \]

  + eigenvalue equation for $T$: $T(\bf{v}) = \lambda \bf{v}$
    + $\lambda$: the eigenvalue of $T$ corresponding to the eigenvector $\bf{v}$
    + $T(\bf{v})$: the result of the applying the transformation $T4 tot the vector $\bf{v}$
    + $\lambda\bf{v}$: the product of the scalar $\lambda$ w/ $\bf{v}$

+ Eigenspaces
  + $\exists\; \lambda$, an eigenvalue, consider the set

    \[ E = \{\bf{v}: T(\bf{v}) = \lambda \bf{v}\} \]

  + eigenspace / characteristic space of $T$, $E$: the union of the zero vector w/ the set of all eigenvectors associated w/ $\lambda$
  + a linear transformation: $\forall\; \bf{u, v} \in V, \alpha \in K$

    \[\begin{align*}
      T(\bf{x} + \bf{y}) &= T(\bf{x}) + T(\bf{y}) \\
      T(\alpha \bf{x}) &= \alpha T(\bf{x})
    \end{align*}\]

  + $\exists\; \bf{u}, \bf{v} \in E$, eigenvectors of $T$ associated w/ eigenvalue $\lambda \implies$

    \[\begin{align*}
      T(\bf{u} + \bf{v}) &= T(\bf{u}) + T(\bf{v}) \\
      T(\alpha \bf{v}) &= \alpha T(\bf{v})
    \end{align*}\]

  + $\bf{u} + \bf{v} \in E$ w/ $E$ close under addition and scalar multiplication
  + the eigenspace $E$ w/ $\lambda \implies$ a linear subspace of $V$
  + eigenline: subspace w/ dimension 1

+ Geometric multiplicity
  + $\gamma_T(\lambda)$: the geometric multiplicity of an eigenvalue $\lambda$, the dimension of the eigenspace associated w/ $\lambda$
  + the maximum number of linearly independent eigenvectors associated w/ that eigenvalue
  + every eigenvalue at least one eigenvector $\implies \gamma_T(\lambda) \ge 1$
  + the eigenspace of $T$ forming a direct sum
    + eigenvectors of different eigenvalues always linearly independent
    + the sum of the dimensions of the eigenspaces unable to exceed the dimension $n$ of the vector space on which $T$ operators

+ Eigenbasis
  + invariant subspace: any subspace spanned by eigenvectors of $T$
  + restriction of $T$: diagonalizable
  + the entire vector space $V$ spanned by the eigenvectors of $T$
  + the direct sum of the eigenspaces associated w/ all the eigenvalues of $T$ is the entire vector space $V \implies$ a basis of $V$, called an __eigenbasis__, formed from linearly independent eigenvectors of $T$
  + $T$ admits an eigenbasis $\implies$ $T$ diagonalizable

+ Zero vector as an eigenvector
  + let an eigenvalue to be any scalar $\lambda \in K$ s.t. $\exists\; \bf{v} \in V$ satifing Eq. (5)
  + the vector must be nonzero $\because$ zero vector allows any scalar in $K$ to be an eigenvalue

+ Spectral theory
  + $\lambda$ as an eigenvalue of $T \implies$ the operator $(T - \lambda I)$ not one-to-one
  + $(T - \lambda I)^{-1}$ not existed
  + the operator $(T - \lambda I)$ w/o inverse even if $\lambda$ not an eigenvalue
  + functional analysis
    + eigenvalues generalized tot the spectrum of a linear operator $T$ as the set of all scalars $\lambda$
    + the spectrum of an operator containing all its eigenvalues but not limited to them


## Dynamic equations

+ Dynamic equations
  + simplest difference equations

    \[ x_t = a_1 x_{t-1} + a_2 x_{t-2} + \cdots + a_k x_{t-k} \]

  + solution w/ characteristic equation

    \[ \lambda^k - a_1 \lambda^{k-1} - a_2 \lambda^{k-2} - \cdots - a_{k-1} \lambda - a_k = 0 \]

  + stacking the solution into matrix form $\to$ a set of equation consisting of the above difference equation
  + the $k -1$ equations:
    + $x_{t-1} = x_{t-1}, \dots, x_{t-k+1} = x_{t-k+1}$
    + $k$-dim system of the first order in the stacked variable vector $[x_t \cdots x_{t-k+1}]$ in terms of its once-lagged value
    + taking the characteristic equation of the system's matrix
  + $k$ characteristic roots $\lambda_1, \dots, \lambda_k$ as the solution equation

    \[ x_t = c_1 \lambda_1^t + \cdots + c_k \lambda_k^t \]

  + similar procedure used for solving a differentiable equation of the form

    \[ \frac{d^k x}{dt^k} + a_{k-1} \frac{d^{k-1} x}{dt^{k-1}} + \cdots + a_ \frac{d x}{dt} + a_0 x = 0 \]


## Calculation

+ 




## Applications

+ Eigenvalues of geometric transformations

  <table style="text-align:center; margin:1em auto 1em auto; width: 60vw;">
  <tbody><tr>
  <th>
  </th>
  <th><a href="https://en.wikipedia.org/wiki/Scaling_(geometry)" title="Scaling (geometry)">Scaling</a>
  </th>
  <th>Unequal scaling
  </th>
  <th><a href="https://en.wikipedia.org/wiki/Rotation_(geometry)" title="Rotation (geometry)">Rotation</a>
  </th>
  <th><a href="https://en.wikipedia.org/wiki/Shear_mapping" title="Shear mapping">Horizontal shear</a>
  </th>
  <th><a href="https://en.wikipedia.org/wiki/Hyperbolic_rotation" title="Hyperbolic rotation">Hyperbolic rotation</a>
  </th></tr>
  <tr>
  <th>Illustration
  </th>
  <td><a href="https://en.wikipedia.org/wiki/File:Homothety_in_two_dim.svg"><img alt="Equal scaling (homothety)" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Homothety_in_two_dim.svg/100px-Homothety_in_two_dim.svg.png" decoding="async" width="100" height="100" srcset="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Homothety_in_two_dim.svg/150px-Homothety_in_two_dim.svg.png 1.5x, https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Homothety_in_two_dim.svg/200px-Homothety_in_two_dim.svg.png 2x" data-file-width="1100" data-file-height="1100"></a>
  </td>
  <td><a href="https://en.wikipedia.org/wiki/File:Unequal_scaling.svg"><img alt="Vertical shrink and horizontal stretch of a unit square." src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Unequal_scaling.svg/100px-Unequal_scaling.svg.png" decoding="async" width="100" height="75" srcset="https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Unequal_scaling.svg/150px-Unequal_scaling.svg.png 1.5x, https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Unequal_scaling.svg/200px-Unequal_scaling.svg.png 2x" data-file-width="800" data-file-height="600"></a>
  </td>
  <td><a href="https://en.wikipedia.org/wiki/File:Rotation.png"><img alt="Rotation by 50 degrees" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Rotation.png/100px-Rotation.png" decoding="async" width="100" height="99" srcset="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Rotation.png/150px-Rotation.png 1.5x, https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Rotation.png/200px-Rotation.png 2x" data-file-width="303" data-file-height="299"></a>
  </td>
  <td><div><div><a href="https://en.wikipedia.org/wiki/File:Shear.svg"><img alt="Horizontal shear mapping" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Shear.svg/100px-Shear.svg.png" decoding="async" width="100" height="75" srcset="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Shear.svg/150px-Shear.svg.png 1.5x, https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Shear.svg/200px-Shear.svg.png 2x" data-file-width="800" data-file-height="600"></a></div></div>
  </td>
  <td><a href="https://en.wikipedia.org/wiki/File:Squeeze_r%3D1.5.svg"><img alt="Squeeze r=1.5.svg" src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Squeeze_r%3D1.5.svg/100px-Squeeze_r%3D1.5.svg.png" decoding="async" width="100" height="67" srcset="https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Squeeze_r%3D1.5.svg/150px-Squeeze_r%3D1.5.svg.png 1.5x, https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Squeeze_r%3D1.5.svg/200px-Squeeze_r%3D1.5.svg.png 2x" data-file-width="820" data-file-height="550"></a>
  </td></tr>
  <tr style="vertical-align:top">
  <th>Matrix
  </th>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{bmatrix}k&amp;0\\0&amp;k\end{bmatrix}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mi>k</mi></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mi>k</mi></mtd></mtr></mtable><mo>]</mo></mrow>    </mrow>  </mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle {\begin{bmatrix}k&amp;0\\0&amp;k\end{bmatrix}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/e53e684e250ff477f3d46652383cae556aaf8eb8" aria-hidden="true" style="vertical-align: -2.505ex; width:7.951ex; height:6.176ex;" alt="{\displaystyle {\begin{bmatrix}k&amp;0\\0&amp;k\end{bmatrix}}}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{bmatrix}k_{1}&amp;0\\0&amp;k_{2}\end{bmatrix}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><msub><mi>k</mi><mrow><mn>1</mn></mrow></msub></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd>2</mn></mrow></msub></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle {\begin{bmatrix}k_{1}&amp;0\\0&amp;k_{2}\end{bmatrix}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/7bb5fc953bd28772a8946d5010c987f604198456" aria-hidden="true" style="vertical-align: -2.505ex; width:10.06ex; height:6.176ex;" alt="{\displaystyle {\begin{bmatrix}k_{1}&amp;0\\0&amp;k_{2}\end{bmatrix}}}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{bmatrix}c&amp;-s\\s&amp;c\end{bmatrix}}}">
    <semantics>
      <mrow>  <mstyle displaystyle="true" scriptlevel="0"><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mi>c</mi></mtd><mtd><mo>−<!-- − --></mo><mi>s</mi></mtd></mtr><mtr><mtd><mi>s</mi></mtd><mtd><mi>c</mi></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle {\begin{bmatrix}c&amp;-s\\s&amp;c\end{bmatrix}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/fe419a75a755a4061ffb54342fd7da682da59d10" aria-hidden="true" style="vertical-align: -2.505ex; width:9.518ex; height:6.176ex;" alt="{\displaystyle {\begin{bmatrix}c&amp;-s\\s&amp;c\end{bmatrix}}}"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle c=\cos \theta }">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mi>c</mi><mo>=</mo><mi>cos</mi><mo>⁡<!-- ⁡ --></mo><mi>θ<!-- θ --></mi></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle c=\cos \theta }</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/6baa53af54b1d724a7245a024f7b79172005a757" aria-hidden="true" style="vertical-align: -0.338ex; width:8.694ex; height:2.176ex;" alt="{\displaystyle c=\cos \theta }"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle s=\sin \theta }">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mi>s</mi><mo>=</mo><mi>sin</mi><mo>⁡<!-- ⁡ --></mo><mi>θ<!-- θ --></mi></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle s=\sin \theta }</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2a780f95c74931076ff3a877ec63685ce848744e" aria-hidden="true" style="vertical-align: -0.338ex; width:8.522ex; height:2.176ex;" alt="{\displaystyle s=\sin \theta }"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{bmatrix}1&amp;k\\0&amp;1\end{bmatrix}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>1</mn></mtd><mtd><mi>k</mi></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle {\begin{bmatrix}1&amp;k\\0&amp;1\end{bmatrix}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/91500c98e17498645a476b428553f65bc739829b" aria-hidden="true" style="vertical-align: -2.505ex; width:7.903ex; height:6.176ex;" alt="{\displaystyle {\begin{bmatrix}1&amp;k\\0&amp;1\end{bmatrix}}}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{bmatrix}c&amp;s\\s&amp;c\end{bmatrix}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em">
      <annotation encoding="application/x-tex">{\displaystyle {\begin{bmatrix}c&amp;s\\s&amp;c\end{bmatrix}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/08688e17a8535881d176d96693fecc1a5829432e" aria-hidden="true" style="vertical-align: -2.505ex; width:7.71ex; height:6.176ex;" alt="{\displaystyle {\begin{bmatrix}c&amp;s\\s&amp;c\end{bmatrix}}}"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle c=\cosh \varphi }">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mi>c</mi><mo>=</mo><mi>cosh</mi><mo>⁡<!-- ⁡ --></mo><mi>φ<!-- φ --></mi></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle c=\cosh \varphi }</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/080f3b967e77d7e4fe86de9610473ecdf13c70be" aria-hidden="true" style="vertical-align: -0.838ex; width:10.416ex; height:2.676ex;" alt="{\displaystyle c=\cosh \varphi }"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle s=\sinh \varphi }">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mi>s</mi><mo>=</mo><mi>sinh</mi><mo>⁡<!-- ⁡ --></mo><mi>φ<!-- φ --></mi></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle s=\sinh \varphi }</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1d47465eee3a66306293f395a6b720e1e63f89bf" aria-hidden="true" style="vertical-align: -0.838ex; width:10.244ex; height:2.676ex;" alt="{\displaystyle s=\sinh \varphi }"></span>
  </td></tr>
  <tr>
  <th>Characteristic<br>polynomial
  </th>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \ (\lambda -k)^{2}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mtext>&nbsp;</mtext><mo stretchy="false">(</mo><mi>λ<!-- λ --></mi><mo>−<!-- − --></mo><mi>k</mi><msup><mo stretchy="false">)</mo><mrow><mn>2</mn></mrow></msup></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \ (\lambda -k)^{2}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/0fdb919512cf0ff0f4009cf5c2e5628d036154cb" aria-hidden="true" style="vertical-align: -0.838ex; width:8.851ex; height:3.176ex;" alt="\ (\lambda -k)^{2}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle (\lambda -k_{1})(\lambda -k_{2})}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mo stretchy="false">(</mo><mi>λ<!-- λ --></mi><mo>−<!-- − --></mo><msub><mi>k</mi><mrow><mn>1</mn></mrow></msub><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mi>λ<!-- λ --></mi><mo>−<!-- − --></mo><msub><mi>k</mi><mrow><mn>2</mn></mrow></msub><mo stretchy="false">)</mo></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle (\lambda -k_{1})(\lambda -k_{2})}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/83039dceee60853050759e4ad606ca8a90068292" aria-hidden="true" style="vertical-align: -0.838ex; width:16.541ex; height:2.843ex;" alt="(\lambda -k_{1})(\lambda -k_{2})"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda ^{2}-2c\lambda +1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msup><mi>λ<!-- λ --></mi><mrow><mn>2</mn></mrow></msup><mo>−<!-- − --></mo><mn>2</mn><mi>c</mi><mi>λ<!-- λ --></mi><mo>+</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda ^{2}-2c\lambda +1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/f7a70690556dbacded9c75a6c2cf0901abddf3e7" aria-hidden="true" style="vertical-align: -0.505ex; width:12.777ex; height:2.843ex;" alt="\lambda ^{2}-2c\lambda +1"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \ (\lambda -1)^{2}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mtext>&nbsp;</mtext><mo stretchy="false">(</mo><mi>λ<!-- λ --></mi><mo>−<!-- − --></mo><mn>1</mn><msup><mo stretchy="false">)</mo><mrow><mn>2</mn></mrow></msup></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \ (\lambda -1)^{2}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c87e8c25e09c5cdd0b21a172effb3f27efce3db9" aria-hidden="true" style="vertical-align: -0.838ex; width:8.802ex; height:3.176ex;" alt="\ (\lambda -1)^{2}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda ^{2}-2c\lambda +1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msup><mi>λ<!-- λ --></mi><mrow><mn>2</mn></mrow></msup><mo>−<!-- − --></mo><mn>2</mn><mi>c</mi><mi>λ<!-- λ --></mi><mo>+</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda ^{2}-2c\lambda +1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/f7a70690556dbacded9c75a6c2cf0901abddf3e7" aria-hidden="true" style="vertical-align: -0.505ex; width:12.777ex; height:2.843ex;" alt="\lambda ^{2}-2c\lambda +1"></span>
  </td></tr>
  <tr>
  <th>Eigenvalues, <span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{i}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mi>i</mi></mrow></msub></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{i}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/72fde940918edf84caf3d406cc7d31949166820f" aria-hidden="true" style="vertical-align: -0.671ex; width:2.155ex; height:2.509ex;" alt="\lambda _{i}"></span>
  </th>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{1}=\lambda _{2}=k}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>λ<!-- λ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mi>k</mi></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{1}=\lambda _{2}=k}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/326cbde7a4ba5352e8b638532df84753d1ec4b4f" aria-hidden="true" style="vertical-align: -0.671ex; width:12.227ex; height:2.509ex;" alt="\lambda _{1}=\lambda _{2}=k"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{1}=k_{1}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>k</mi><mrow><mn>1</mn></mrow></msub></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{1}=k_{1}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/78ae6617a5303fafd2bae47632d9d9a7dc81642e" aria-hidden="true" style="vertical-align: -0.671ex; width:7.773ex; height:2.509ex;" alt="\lambda _{1}=k_{1}"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{2}=k_{2}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><msub><mi>k</mi><mrow><mn>2</mn></mrow></msub></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{2}=k_{2}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ac5003f99ebb5285d2afe3c30927f1ea3a18f32e" aria-hidden="true" style="vertical-align: -0.671ex; width:7.773ex; height:2.509ex;" alt="\lambda _{2}=k_{2}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{1}=e^{i\theta }=c+si}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><msup><mi>e</mi><mrow><mi>i</mi><mi>θ<!-- θ --></mi></mrow></msup><mo>=</mo><mi>c</mi><mo>+</mo><mi>s</mi><mi>i</mi></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{1}=e^{i\theta }=c+si}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1fa44786d4f2cc95c4b24e4329a322e2bc9b5dcd" aria-hidden="true" style="vertical-align: -0.671ex; width:17.001ex; height:3.009ex;" alt="{\displaystyle \lambda _{1}=e^{i\theta }=c+si}"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{2}=e^{-i\theta }=c-si}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><msup><mi>e</mi><mrow><mo>−<!-- − --></mo><mi>i</mi><mi>θ<!-- θ --></mi></mrow></msup><mo>=</mo><mi>c</mi><mo>−<!-- − --></mo><mi>s</mi><mi>i</mi></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{2}=e^{-i\theta }=c-si}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/c5b7a54f4fa6bae6e57d3343c8bec0c25c6867d7" aria-hidden="true" style="vertical-align: -0.671ex; width:18.279ex; height:3.009ex;" alt="{\displaystyle \lambda _{2}=e^{-i\theta }=c-si}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{1}=\lambda _{2}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>λ<!-- λ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{1}=\lambda _{2}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/8110a57bf3475c2bfba8d4af326160500b7bef6f" aria-hidden="true" style="vertical-align: -0.671ex; width:12.178ex; height:2.509ex;" alt="\lambda _{1}=\lambda _{2}=1"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{1}=e^{\varphi }}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><msup><mi>e</mi><mrow><mi>φ<!-- φ --></mi></mrow></msup></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{1}=e^{\varphi }}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/66978ed4f5bfb501db97a3b645b8e34ad901dc72" aria-hidden="true" style="vertical-align: -0.671ex; width:7.899ex; height:2.676ex;" alt="\lambda _{1}=e^{\varphi }"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \lambda _{2}=e^{-\varphi }}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>λ<!-- λ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><msup><mi>e</mi><mrow><mo>−<!-- − --></mo><mi>φ<!-- φ --></mi></mrow></msup></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \lambda _{2}=e^{-\varphi }}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2321e6824b3067d46abe87853d4b436e3868c2ff" aria-hidden="true" style="vertical-align: -0.671ex; width:9.177ex; height:2.843ex;" alt="\lambda _{2}=e^{-\varphi }"></span>,
  </td></tr>
  <tr>
  <th>Algebraic <abbr title="multiplicity">mult.</abbr>,<br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{i}=\mu (\lambda _{i})}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mi>i</mi></mrow></msub><mo>=</mo><mi>μ<!-- μ --></mi><mo stretchy="false">(</mo><msub><mi>λ<!-- λ --></mi><mrow><mi>i</mi></mrow></msub><mo stretchy="false">)</mo></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{i}=\mu (\lambda _{i})}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/44a86bfbea85072f96882d88c90d4a1d8d42a7e5" aria-hidden="true" style="vertical-align: -0.838ex; width:10.666ex; height:2.843ex;" alt="{\displaystyle \mu _{i}=\mu (\lambda _{i})}"></span>
  </th>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{1}=2}">
    <semantics><mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>2</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{1}=2}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/37b804debefab063745106268754e2ad8f83a033" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{1}=2"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{1}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{1}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/093269c0f47ef7efeb987e03733c11183738bea1" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{1}=1"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{2}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{2}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/037818fb3de56a2ed84255e914be2f61db842133" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{2}=1"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{1}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{1}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/093269c0f47ef7efeb987e03733c11183738bea1" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{1}=1"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{2}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{2}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/037818fb3de56a2ed84255e914be2f61db842133" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{2}=1"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{1}=2}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>2</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{1}=2}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/37b804debefab063745106268754e2ad8f83a033" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{1}=2"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{1}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{1}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/093269c0f47ef7efeb987e03733c11183738bea1" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{1}=1"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mu _{2}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>μ<!-- μ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mu _{2}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/037818fb3de56a2ed84255e914be2f61db842133" aria-hidden="true" style="vertical-align: -0.838ex; width:6.717ex; height:2.676ex;" alt="\mu _{2}=1"></span>
  </td></tr>
  <tr>
  <th>Geometric <abbr title="multiplicity">mult.</abbr>,<br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{i}=\gamma (\lambda _{i})}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mi>i</mi></mrow></msub><mo>=</mo><mi>γ<!-- γ --></mi><mo stretchy="false">(</mo><msub><mi>λ<!-- λ --></mi><mrow><mi>i</mi></mrow></msub><mo stretchy="false">)</mo></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{i}=\gamma (\lambda _{i})}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ce3237dd9f6f5daba876c3e39baefcfbcca24504" aria-hidden="true" style="vertical-align: -0.838ex; width:10.329ex; height:2.843ex;" alt="\gamma _{i}=\gamma (\lambda _{i})"></span>
  </th>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{1}=2}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>2</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{1}=2}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/6e807587733866aeb774d767add51a536fe647f9" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{1}=2"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{1}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{1}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/79b71e25e13c46dad7b2f6e07c3b6cf6e7a739d9" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{1}=1"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{2}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{2}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/55749b967103dbda8fffc595eccad3afce20cbec" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{2}=1"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{1}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{1}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/79b71e25e13c46dad7b2f6e07c3b6cf6e7a739d9" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{1}=1"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{2}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{2}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/55749b967103dbda8fffc595eccad3afce20cbec" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{2}=1"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{1}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{1}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/79b71e25e13c46dad7b2f6e07c3b6cf6e7a739d9" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{1}=1"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{1}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>1</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{1}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/79b71e25e13c46dad7b2f6e07c3b6cf6e7a739d9" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{1}=1"></span><br><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \gamma _{2}=1}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mi>γ<!-- γ --></mi><mrow><mn>2</mn></mrow></msub><mo>=</mo><mn>1</mn></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \gamma _{2}=1}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/55749b967103dbda8fffc595eccad3afce20cbec" aria-hidden="true" style="vertical-align: -0.838ex; width:6.519ex; height:2.676ex;" alt="\gamma _{2}=1"></span>
  </td></tr>
  <tr>
  <th>Eigenvectors
  </th>
  <td>All nonzero vectors
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\0\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}0\\1\end{bmatrix}}\end{aligned}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mrow><mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0em 2em 0em 2em 0em 2em 0em 2em 0em 2em 0em" displaystyle="true"><mtr><mtd><msub><mrow><mi mathvariant="bold">u</mi></mrow><mrow><mn>1</mn></mrow></msub></mtd><mtd><mi></mi><mo>=</mo><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mtd></mtr><mtr><mtd><msub><mrow><mi mathvariant="bold">u</mi></mrow><mrow><mn>2</mn></mrow></msub></mtd><mtd><mi></mi><mo>=</mo><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mtd></mtr></mtable></mrow></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\0\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}0\\1\end{bmatrix}}\end{aligned}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/7a1069ff8e7733cfde92189c7fff9b12ebd734e3" aria-hidden="true" style="vertical-align: -5.671ex; width:10.758ex; height:12.509ex;" alt="{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\0\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}0\\1\end{bmatrix}}\end{aligned}}}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\-i\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}1\\+i\end{bmatrix}}\end{aligned}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mrow><mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0em 2em 0em 2em 0em 2em 0em 2em 0em 2em 0em" displaystyle="true"><mtr><mtd><msub><mrow><mi mathvariant="bold">u</mi></mrow><mrow><mn>1</mn></mrow></msub></mtd><mtd><mi></mi><mo>=</mo><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mo>−<!-- − --></mo><mi>i</mi></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mtd></mtr><mtr><mtd><msub><mrow><mi mathvariant="bold">u</mi></mrow><mrow><mn>2</mn></mrow></msub></mtd><mtd><mi></mi><mo>=</mo><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mo>+</mo><mi>i</mi></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mtd></mtr></mtable></mrow></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\-i\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}1\\+i\end{bmatrix}}\end{aligned}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ebfcdafb92833dd29fcabb84de5349c305bf4626" aria-hidden="true" style="vertical-align: -5.671ex; width:12.207ex; height:12.509ex;" alt="{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\-i\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}1\\+i\end{bmatrix}}\end{aligned}}}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle \mathbf {u} _{1}={\begin{bmatrix}1\\0\end{bmatrix}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><msub><mrow><mi mathvariant="bold">u</mi></mrow><mrow><mn>1</mn></mrow></msub><mo>=</mo><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle \mathbf {u} _{1}={\begin{bmatrix}1\\0\end{bmatrix}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/de1f984884c937803d30931a83c5310fefb97b5d" aria-hidden="true" style="vertical-align: -2.505ex; width:10.007ex; height:6.176ex;" alt="{\displaystyle \mathbf {u} _{1}={\begin{bmatrix}1\\0\end{bmatrix}}}"></span>
  </td>
  <td><span><span style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\1\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}1\\-1\end{bmatrix}}.\end{aligned}}}">
    <semantics>
      <mrow><mstyle displaystyle="true" scriptlevel="0"><mrow><mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0em 2em 0em 2em 0em 2em 0em 2em 0em 2em 0em" displaystyle="true"><mtr><mtd><msub><mrow><mi mathvariant="bold">u</mi></mrow><mrow><mn>1</mn></mrow></msub></mtd><mtd><mi></mi><mo>=</mo><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></mtd></mtr><mtr><mtd><msub><mrow><mi mathvariant="bold">u</mi></mrow><mrow><mn>2</mn></mrow></msub></mtd><mtd><mi></mi><mo>=</mo><mrow><mrow><mo>[</mo><mtable rowspacing="4pt" columnspacing="1em"><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mo>−<!-- − --></mo><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow><mo>.</mo></mtd></mtr></mtable> </mrow> </mstyle></mrow>
      <annotation encoding="application/x-tex">{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\1\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}1\\-1\end{bmatrix}}.\end{aligned}}}</annotation>
    </semantics>
  </math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b925c7647d284eec56cd2b47431cd9392945d4cf" aria-hidden="true" style="vertical-align: -5.671ex; width:13.213ex; height:12.509ex;" alt="{\displaystyle {\begin{aligned}\mathbf {u} _{1}&amp;={\begin{bmatrix}1\\1\end{bmatrix}}\\\mathbf {u} _{2}&amp;={\begin{bmatrix}1\\-1\end{bmatrix}}.\end{aligned}}}"></span>
  </td></tr></tbody></table>




