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


## Eigenvalues and characteristic polynomial




## Algebraic multiplicity





## Eigenspaces, geometric multiplicity, and the eigenbasis for matrices





## Properties of eigenvalues




## Left and right eigenvectors





## Diagonalization and the eigendecomposition





## Variational characterization





## Eigenvector-eigenvalue identity






## Examples






## General definitions





## Calculation






## Applications



