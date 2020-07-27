# Linear Algebra

## Vectors

+ [Vector notation](./Stats/ProbStatsPython/12-RegPCA.md#121-review-of-linear-algebra)
  + __vectors__: letters with a little arrow on top, e.g., $\vec{a},\vec{b},\vec{v}_1,\vec{v}_2,\ldots$
  + $\Bbb{R}^d$: vectors grouped by __dimension d__, the set of all $d$ dimensional (Euclidean) vectors
  + $d$ dimensional vector
    + an element of $\Bbb{R}^d$
    + described by a sequence of $d$ real numbers

+ [Operations on vectors](./Stats/ProbStatsPython/12-RegPCA.md#121-review-of-linear-algebra)
  + basic
    + v1= [1 2] v2= [-1  1]
    + v1+v2= [0 3]
    + 4*v2= [-4  4]
    + -v1= [-1 -2]

  + the inner product
    + __inner product__ or __dot product__: an operation taking two vectors w/ same dimension and returning a number (scalar)
    + math notation: $\vec{a} \cdot \vec{b}$

  + the norm of a vector
    + __length__, __magnitude__, or __norm__ of a vector
    + the distance btw origin, where the vector starts, and its tip

      \[\parallel\vec{v}\parallel = \sqrt{\sum_i v_i^2} = \sqrt{\vec{v} \cdot \vec{v}} \]
  
  + unit vectors
    + vectors whose norm is 1
    + normalizing any vector by dividing its length

  + projection
    + taking the dot product of an arbitrary vector w/ a unit vector
    + a simple geometric interpretation

  + orthogonal vectors
    + two vectors w/ zero dot product
    + the angle btw two vectors is 90 degrees


## Orthonormal Basis

+ [Orthonormal basis](./Stats/ProbStatsPython/12-RegPCA.md#121-review-of-linear-algebra)
  + Definition: (orthonormal basis) the vectors $\vec{u}_1, \vec{u}_2, \dots, \vec{u}_d \in \Bbb{R}^d$ form an <span style="color: magenta; font-weight: bold;"> orthonormal basis of $\Bbb{R}^d$, if
    + __normality__: $\vec{u}_1, \vec{u}_2, \dots, \vec{u}_d$ are unit vectors, i.e., $\forall\, 1 \le i \le d: \vec{u}_i \cdot \vec{u}_i = 1$
    + __orthogonality__: every pair of vectors are orthogonal, i.e., $\forall\, 1 \le i \ne j \le d: \vec{u}_i \cdot \vec{u}_j = 0$
  + the standard basis
    + $\vec{e}_1 = [1,0,0,\ldots,0], \vec{e}_2 = [0,1,0,\ldots,0], \dots,\vec{e}_d = [0,0,0,\ldots,1]$
    + $v_i$: the $i$th coordinate of $\vec{v} =$ the dot product of a vector $\vec{v}$ w/ a standard basis vector $\vec{e}_i$
  + reconstruction using an orthonormal basis $\vec{u}_1,\ldots,\vec{u}_d$
    + orthonormal basis defining a _coordinate system_
    + allowing to move btw coordinate systems
    + represented as a list of $d$ dot products: $[\vec{v}\cdot\vec{u}_1,\vec{v}\cdot\vec{u}_2,\ldots,\vec{v}\cdot\vec{u}_d]$
    + reconstructing by summing its projections on the basis vectors: $\vec{v} = (\vec{v}\cdot \vec{u}_1)\, \vec{u}_1 + \cdots + (\vec{v}\cdot \vec{u}_d)\, \vec{u}_d$
  + $[v_1, v_2, \dots, v_d]$: representing a vector $\vec{v}$ w/ the standard basis
  + __change of basis__
    + representing $\vec{v}$ using an orthonormal basis
    + demo (see diagram)
      + representing the vector $\vec{v}$ from the standard basis $[\vec{e}_1, \vec{e}_2]$ to a new orthonormal basis $[vec{u}_1, \vec{u}_2]$
      + green arrow: projections of $\vec{v}$ onto the directions defined by $\vec{u}_1$ and $\vec{u}_2$

    <div style="margin: 0.5em; display: flex; justify-content: center; align-items: center; flex-flow: row wrap;">
      <a href="./src/Topic12-Lectures/1.Linear_Algebra_Review.ipynb" ismap target="_blank">
        <img src="./Stats/ProbStatsPython/img/t12-05.png" style="margin: 0.1em;" alt="Example of change of basis" title="Example of change of basis" height=250>
      </a>
    </div>

+ [Orthonormal matrices and change of Basis
  + change of basis using matrix notation
    + let ${\bf u}_i = \begin{bmatrix} u_{i1} & u_{i2} & \cdots & u_{id} \end{bmatrix}$
    + orthonormal matrix

    \[ {\bf U} = \begin{bmatrix} {\bf u}_1 \\ {\bf u}_2 \\ \vdots \\ {\bf u}_d \end{bmatrix} = \begin{bmatrix}  u_{11} & u_{12} & \ldots & u_{1d} \\  u_{21} & u_{22} & \ldots & u_{2d} \\  \vdots\\ u_{d1} & u_{d2} & \ldots & u_{dd}  \end{bmatrix} \]

  + orthonormality: ${\bf UU}^\top = {\bf I}$
  + reconstruction of ${\bf v} = $ ${\bf UU}^\top {\bf v}$






