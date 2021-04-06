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

    \[ E = \{\bf(v}: (A - \lambda I) \bf{v} = 0 \} \]

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
    + $\bf{u, v} \in E \inplies (\bef{u + v} \in E$
    + equivalently $A(\bf{u + v)} = \lambda(\bf{u + v})$
    + proved w/ distributive property of matrix multiplication
  + closed under scalar multiplication
    + $\bf{v} \in E$ and $\alpha \Bbb{C}$
    + $(\alpha \bf{v}) \in E$
    + equivalently $A(\alpha \bf{v}) = \lambda(\alpha \bf{v})$
    + proved by multiplication of complex matrices by complex number is commutative
    + $\bf{u + v}$ and $\alpha \bf{v}$ not zero, eigenvectors of $A$ associated w/ $\lambda$



## Properties of eigenvalues




## Left and right eigenvectors





## Diagonalization and the eigendecomposition





## Variational characterization





## Eigenvector-eigenvalue identity






## Examples






## General definitions





## Calculation






## Applications



