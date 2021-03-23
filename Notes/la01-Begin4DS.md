# A Comprehensive Beginners Guide to Linear Algebra for Data Scientist

Author: V. K. Yadav

Date: 2017-05-17

[Origin](https://www.analyticsvidhya.com/blog/2017/05/comprehensive-guide-to-linear-algebra/)

## 1. Motivation – Why learn Linear Algebra?

+ Motivation
  + scenario 1: identify a flower image
    + how does a computer store an image?
    + identify the attributes of the image
    + store pixel intensities $\to$ matrix
  + scenario 2: XGBOOST
    + an algorithm employed most frequently by winners of Data Science Competition
    + store numerical data in Matrix form
    + enable to process faster and more accurate
  + scenario 3: deep learning
    + using matrix to store inputs
    + weights learned by NN also store in matrix
  + scenario 4: text processing
    + common techniques: Bag of Words, Terms Document s and frequency in matrix
    + store counts (or other similar attributes) of words in documents and frequency in matrix

## 2. Representation of problems in Linear Algebra

+ Representation of problems in Linear Algebra
  + 2 bats and 1 ball; 2 balls and 1 bat $\to$ \$100
  
    \[\begin{align*}
      2x + y &= 100 \\
      x + 2y &= 100
    \end{align*}\]

  + Linear Algebra: data represented in the form of linear equations $\to$ represented in the form of matrices and vectors
  + matrix used to save a long set of linear equations
  + planes: 4 possible cases $\implies$ difficult w/ higher dimensions
    + no intersection at all
    + planes intersect in a line
    + intersect in a plane
    + all planes intersect at a point

## 3. Matrix

+ Matrix terms
  + __order of matrix__: $\text{rows} \times \text{cols}$
  + __square matrix__: rows = cols
  + __upper triangle matrix__: square matrix w/ all elements below diagonal = 0
  + __lower triangle matrix__: square matrix w/ all elements above diagonal = 0
  + __scalar matrix__: square matrix w/ all the diagonal elements equal to some constant $k$
  + __identity matrix__: diagonal = 1, others = 0
  + __column matrix__: only 1 column; vector
  + __row matrix__: only 1 row
  + __trace__: the sum of all the diagonal elements of a square matrix

+ Basic operations on Matrix
  + addition: $\mathbf{C} = \mathbf{A} + \mathbf{B} \to C_{ij} = A_{ij} + B_{ij}$
  + scalar multiplication: $c \cdot \mathbf{A} \to c \cdot [A_{ij}] = [c \, A_{ij}]$
  + transposition: interchanging row and col index
    + math presentation: $A_{ij}^T = A_{ji}$
    + example: python code

      ```python
      import numpy as np
      import pandas as pd

      A = np.arange(21, 30).reshape(3, 3)
      A.transport()
      ```

  + matrix multiplication:
    + math representation: $\mathbf{C} = \mathbf{AB} \to C_{ij} = A_{ik} \times B_{kj}$
    + example: python code

      ```python
      import numpy as np

      A = np.arange(21, 30).reshape(3, 3)
      B = np.arange(31, 40).reshape(3, 3)
      A.dot(B)
      B.dot(A)
      ```

    + properties
      + associative: $\mathbf{A(BC)} = \mathbf{(AB}C)}$

        ```python
        import numpy as np

        A = np.arange(21, 30).reshape(3, 3)
        B = np.arange(31, 40).reshape(3, 3)
        C = np.arange(41, 50).reshape(3, 3)

        tmp1 = (A.dot(B)).dot(C)
        tmp2 = A.dot(B.dot(C))
        ```

      + NOT commutative: $\mathbf{AB} \neq \mathbf{BA}$

+ Representing equations in Matrix form

    \[ \mathbf{AX} = \mathbf{Z} \]


## 4. Solving the Problem

+ Ways to solve matrix equations
  + row echelon form
  + inverse of a matrix

+ Row Echelon Form
  + two conditions followed by any manipulation to be valid
    + preserve the solution
    + reversible
  + manipulation
    1. swap the order of the equations
    2. multiply both sides of equation by any non-zero constant $c$
    3. multiply an equation by any non-zero constant and add to other equation
  + basic idea: clear variables in successive equation and form an upper trianglar matrix
  + rank of matrix: the maximum number of linearity interdependent row vectors in a matrix

+ Inverse of a matrix
  + determinant of a matrix: only applicable to square matrix

    \[\begin{align*}
      \mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} &\to \det(\mathbf{A}) = ad - bc \\\\
      \mathbf{B} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} &\to \det(\mathbf{B}) = a \begin{bmatrix} e & f \\ h & i \end{bmatrix} + b \begin{bmatrix} d & f \\ g & i \end{bmatrix} + c \begin{bmatrix} d & e \\g & h \end{bmatrix}
    \end{align*}\]

  + example: python code

    ```python
    import numpy as np

    arr = np.arange(100, 116).reshape(4, 4)
    np.linalg.det(arr)
    ```

  + __minor of a matrix:__<br>minor corresponding to an element ($A_{ij}$) is the deteminant of the sub-matrix formed by deleting the $i^{\text{th}}$ row and $j^{\text{th}}$ column of the matrix
  + __cofactor of a matrix:__ minor of a matrix w/ signs
  + __cofactor matrix:__ replacing the original elements w/ corresponding cofactor
  + __adjoint of a matrix__
    1. find the cofactor matrix of $\mathbf{A}$
    2. transpose the cofactor matrix
  + example: adjoint of a matrix

    \[\mathbf{A} = \begin{bmatrix} 1 & 2 7 0 \\ 4 & 2 & 5 \\ 1 & 0 & 2 \end{bmatrix} \;\;\underrightarrow{\text{cofactor}} \;\;\mathbf{C} = \begin{bmatrix} 4 & -3 & -2 \\ -4 & 2 & 2 \\ 10 & -5 & -6 \end{bmatrix} \;\;\underrightarrow{\text{adjoint}}\;\; \mathbf{D} = \begin{bmatrix} 4 & -4 & 10 \\ -3 & 2 & -5 \\ -2 & 2 & -6 \end{bmatrix} \]

  + inverse of a matrix
    1. find the adjoint
    2. multiply the adjoint matrix by the inverse of determinant of the matrix $\mathbf{A}$

      \[\text{inv}(\mathbf{A}) = \frac{-1}{2} \begin{bmatrix} 4 & -4 & 10 \\ -3 & 2 & -5 \\ -2 & 2 & -6 \end{bmatrix} \]

  + singular matrix: $\det(\mathbf{A}) = 0$
  + solving linear equations: 

    \[\mathbf{AX} = \mathbf{Z} \hspace{0.5em}\to\hspace{0.5em} \mathbf{A^{-1}AX} = \mathbf{A^{-1}Z} \hspace{0.5em}\to\hspace{0.5em} (\mathbf{A^{-1}A})\mathbf{X} = \mathbf{A^{-1}Z} \hspace{0.5em}\to\hspace{0.5em} \mathbf{X} = \mathbf{A^{-1}Z}\]

  + example: python code

    ```python
    import numpy as np

    arr = np.arange(5, 21).reshape(4, 5)
    np.linarg.inv(arr)
    ```

+ Applications of inverse in Data Science
  + inverse used to calculate parameter vector by normal equation in linear equation
  + to find the final parameter vector ($\theta$) assuming the initial function is parameterized by $\theta$ and $X \to $ find the inverse of ($\mathbf{X^TX}$)
  + $f_{\theta}(\mathbf{X}) = \theta^T X$
    + $\that$: the parameter to calculate
    + $\mathbf{X}$: the column vector of feature or independent variables
  + example: python code

    ```python
    import numpy as np
    import pandas as pd

    Df = pd.read,csv(".../baseball.csv")
    Df1 = pd.head(14)
    X = Df1[['RS', 'RA', 'W', 'OPB', 'SLG', 'BA']]
    Y = Df['OOBP']
    T = X.dot(X)
    inv = np.linalg.inv(XT).dot(Y)
    ```
  
  + drawback: computationally costly ($n^3$) $\to$ Gradient Descent $\to$ Eigenvectors

## 5. Eigenvalues and Eigenvectors


## 6. Singular Value Decomposition





