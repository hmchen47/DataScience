# A Comprehensive Beginners Guide to Linear Algebra for Data Scientist

Author: V. K. Yadav

Date: 2017-05-17

[Origin](https://www.analyticsvidhya.com/blog/2017/05/comprehensive-guide-to-linear-algebra/)

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

+ Representation of problems in Linear Algebra
  + 2 bats and 1 ball; 2 balls and 1 bat $\to$ \$100
  
    \[\begin{align*}
      2x + y &= 100 \\
      x + 2y &= 100
    \end{align*}\]

  + Linear Algebra: data represented in the form of linear equations $\to$ represented in the form of matrices and vectors
  + matrix used to save a long set of linear equations
  + planes: 4 possible cases $\imp[lies$ difficult w/ higher dimensions
    + no intersection at all
    + planes intersect in a line
    + intersect in a plane
    + all planes intersect at a point

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








