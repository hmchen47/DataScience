# Linear Algebra Review

## Matrices and Vectors

### Lecture Notes

+ __Matrix__: Rectangular array of numbers:

    $$\left[ \begin{array}{cc} 1420 & 191 \\ 1271 & 821 \\         949 & 1437 \\ 147 & 1448 \end{array} \right] \Longrightarrow \mathbb{R}^{4 \times 2}, 4 \times 2 \text{ matrix} \;\;\;\;\;\;\;\; \left[ \begin{array}{ccc} 1 & 2 & 3 \\ 4 & 5 & 6 \end{array} \right] \Longrightarrow \mathbb{R}^{2 \times 3}, 2 \times 3 \text{ matrix}$$

    __Dimension of matrix__: number of rows x number oof 

+ IVQ: Which of the following statements are true? Check all that apply.

    1. $\displaystyle \left[ \begin{array}{cc} 1 & 2 \\ 4 & 0 \\ 0 & 1 \end{array} \right]$ is a $3\times2$ matrix.
    2. $\displaystyle \left[ \begin{array}{cccc} 0 & 1 & 4 & 2 \\ 3 & 4 & 0 & 9 \end{array} \right]$ is a $x \times 2$ matrix
    3. $\left[ \begin{array}{ccc} 0 & 4 & 2 \\ 3 & 4 & 9 \\ 5 & -1 & 0 \end{array} \right]$ is a $3 \times 3$ matrix
    4. $\left[ \begin{array}{c} 1 & 2 \end{array} \right]$ is a $1 \times 2$ matrix

    Ans: 134

+ __Matrix Elements__ (entries of matrix)

    $$A = \left[ \begin{array}{cc} 1420 & 191 \\ 1271 & 821 \\ 949 & 1437 \\ 147 & 1448 \end{array} \right] $$  

    $A_{ij} = \text{"}i, j \text{entry"}$ in the $i^{th}$ row, $j^{th}$ column

    $A_{11} = 1402, A_{12} = 192, A_{32} = 1437, A_{41} = 147$ and $A_{43} =$ undefined (error)

+ IVQ: Let A be a matrix shown below. $A_{32}$ is one of the elements of this matrix.

    $$A = \left[ \begin{array}{cccc} 85 & 76 & 66 & 5 \\ 94 & 75 & 18 & 28 \\ 68 & 40 & 71 & 5 \end{array} \right]$$

    What is the value of $A_{32}$?

    1. 18
    2. 28
    3. 76
    4. 40

    Ans: 4


+ __Vector__: an $n x 1$ matrix

    $$y = \left[ \begin{array}{c} 460 \\ 232 \\ 315 \\ 178 \end{array} \right] \Longrightarrow \mathbb{R}^4, n = 4, 4-\text{dimensional  vector}$$

    + $y_i = i^{th}$ element: $y_1 = 460, y_2 = 232, y_3 = 315$
    + 1-indexed vs 0-indexed:

        $$y = \left[ \begin{array}{c} y_1 \\ y_2 \\y_3 \\ y_4 \end{array} \right] \Longrightarrow y[1], \text{1-indexed} \;\;\;\;\;\;\;\;\;\;\;\; y = \left[ \begin{array}{c} y_0 \\ y_1 \\y_2 \\ y_3 \end{array} \right] \Longrightarrow y[0], \text{0-indexed}$$
    + Notation: $A, B, C, D, \ldots$ as vector, $a, b, c, d, \ldots$ as scalar/number or element 


-------------------------------

Matrices are 2-dimensional arrays:

$$\left [ \begin{array}{ccc} a & b & c \\ d & e & f \\ g & h & i \\ j & k & l \end{array} \right ]$$

The above matrix has four rows and three columns, so it is a $4 \times 3$ matrix.

A vector is a matrix with one column and many rows:

$$\left [ \begin{array}{c}  w \\ x \\ y \\ z \end{array} \right ]$$

So vectors are a subset of matrices. The above vector is a $4 \times 1$ matrix.

Notation and terms:

+ $A_{ij}$ refers to the element in the $i^{th}$ row and $j^{th}$ column of matrix $A$.
+ A vector with 'n' rows is referred to as an 'n'-dimensional vector.
+ $v_i$ refers to the element in the $i^{th}$ row of the vector.
+ In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
Matrices are usually denoted by uppercase names while vectors are lowercase.
+ "Scalar" means that an object is a single value, not a vector or matrix.
+ $\mathbb{R}$ refers to the set of scalar real numbers.
+ $\mathbb{R^n}$ refers to the set of n-dimensional vectors of real numbers.

Run the cell below to get familiar with the commands in Octave/Matlab. Feel free to create matrices and vectors and try out different things.

```matlab
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector
v = [1;2;3]

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
```


### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/03.1-V2-LinearAlgebraReview%28Optional%29-MatricesAndVectors.3248b0c0b22b11e4901abd97e8288176/full/360p/index.mp4?Expires=1552176000&Signature=Q34XxhHAnk-M21exAsvwr7yc3pOziJ3NQrNKm9-RKojjdZowpZq0sTNNByKp0hp6WMungsT6vMZUQfeLHYG~MgjNoy2cgX~cMmFcv6QjHNqf0OL-3KI7qkpX~qehA2b5ssNNVTtphe-duB0T4PeXmmks6YFOEBVI0XHILGDJB30_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/QBUzAug0SeaVMwLoNJnmKg?expiry=1552176000000&hmac=yWMBD-0chszweNWWR9lIo5tbEc1gvkPIMxE-lEYLJXM&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Addition and Scalar Multiplication

### Lecture Notes

+ Matrix Addition

    $$\left [ \begin{array}{cc} 1 & 0 \\ 2 & 5 \\3 & 1 \end{array} \right ] + \left [ \begin{array}{cc} 4 & 0.5 \\ 2 5 \\ 0 & 1 \end{array} \right ] = \left [ \begin{array}{cc} 5 & 0.5 \\ 4 & 3 \\ 3 & 2 \end{array} \right ]$$

    $$3 \times 2 \text{ matrix   } + 3 \times 2 \text{ matrix   } = 3 \times 2 \text{ matrix   }$$

    <br/>

    $$\left [ \begin{array}{cc} 1 & 0 \\ 2 & 5 \\3 & 1 \end{array} \right ] + \left [ \begin{array}{cc} 4 & 0.5 \\ 2 & 5 \end{array} \right ] = \text{ error }$$

    $$3 \times 2 \text{ matrix    } + 2 \times 2 \text{ matrix     } = \text{error}$$

+ IVQ: what is $\left [ \begin{array}{ccc} 8 & 6 & 9 \\ 10 & 1 & 10 \end{array} \right ] + \left [ \begin{array}{ccc} 3 & 10 & 2 \\ 6 & 1 &-1 \end{array} \right ]$?

    Ans: $\left [ \begin{array}{ccc} 11 & 16 & 11 \\ 16 & 2 7 9 \end{array} \right ]$

+ Scalar Multiplication

    $$3 \times \left [ \begin{array}{cc} 1 & 0 \\ 2 & 5 \\3 & 1 \end{array} \right ] = \left [ \begin{array}{cc} 3 & 0 \\ 6 & 15 \\ 9 & 3 \end{array} \right ] = \left [ \begin{array}{cc} 1 & 0 \\ 2 & 5 \\3 & 1 \end{array} \right ] \times 3$$

    <br/>

    $$\left [ \begin{array}{cc} 4 & 0 \\ 6 & 3 \end{array} \right ] / 4 = \frac{1}{4} \times \left [ \begin{array}{cc} 4 & 0 \\ 6 & 3 \end{array} \right ] = \left [ \begin{array}{cc} 1 & 0 \\ 2/3 & 3/4 \end{array} \right ]$$

+ IVQ: What is $2 \times \left [ \begin{array}{cc} 4 & 5 \\ 1 & 7 \end{array} \right ]$?

    Ans: $\left [ \begin{array}{cc} 8 & 10 \\ 2 & 14 \end{array} \right ]$

+ Combination of Operands

    $$3 \times \left [ \begin{array}{c} 1\\ 4 \\ 2 \end{array} \right ] + \left [ \begin{array}{c} 0 \\ 0 \\ 5 \end{array} \right ] - \left [ \begin{array}{c} 3 \\ 0 \\ 2 \end{array} \right ] / 3 = \left [ \begin{array}{c} 3\\12\\6 \end{array} \right ] + \left [ \begin{array}{c} 0\\0\\5 \end{array} \right ] - \left [ \begin{array}{c} 1\\0\\2/3 \end{array} \right ] = \left [ \begin{array}{c} 2\\12\\10 \frac{1}{3} \end{array} \right ]$$
    <br/>
    + Operation order: scalar multiplication & scalar division > matrix/vector addition & subtraction
    + 3 x 1 matrix; 3-dimensional vector

+ IVQ: What is $\left [ \begin{array}{c} 4\\6\\7 \end{array} \right ] / 2 - 3 \times \left [ \begin{array}{c} 3\\1\\0 \end{array} \right ]$?

    Ans: $\left [ \begin{array}{c} -4\\0\\3.5 \end{array} \right ]$


-------------------------------

Addition and subtraction are __element-wise__, so you simply add or subtract each corresponding element:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] + \left [ \begin{array}{cc} w & x \\ y & z \end{array} \right ] = \left [ \begin{array}{cc} a+w & b+x \\ c+y & d+z \end{array} \right ]$$

Subtracting Matrices:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] - \left [ \begin{array}{cc} w & x \\ y & z \end{array} \right ] = \left [ \begin{array}{cc} a-w & b-x \\ c-y & d-z \end{array} \right ]$$

To add or subtract two matrices, their dimensions must be __the same__.

In scalar multiplication, we simply multiply every element by the scalar value:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] * x = \left [ \begin{array}{cc} a*x & b*x \\ c*x & d*x \end{array} \right ]$$

In scalar division, we simply divide every element by the scalar value:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] / x = \left [ \begin{array}{cc} a/x & b/x \\ c/x & d/x \end{array} \right ]$$

Experiment below with the Octave/Matlab commands for matrix addition and scalar multiplication. Feel free to try out different commands. Try to write out your answers for each command before running the cell below.

```matlab
% Initialize matrix A and B
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s
```


### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/03.2-V2-LinearAlgebraReview%28Optional%29-AdditionAndScalarMultiplication.0c7f88a0b22b11e4960bf70a8782e569/full/360p/index.mp4?Expires=1552176000&Signature=VKo8DujcBmnBkF9KGkX1rZLIcZTGIr~R3bu3fjZa7rYU~EwOVkWbqdBjczdaLg-pJe49qQrZjJwAlET1LlC9YBs6yclGHlCgcafKaDHSkhxIGf128Ens-jt21qpF4xx7-58NbhJmQV3oQrQmqMCCeYRXL8bTYJdgy-mR2Qgt9~c_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/WfU8T8Y5RCC1PE_GObQg8Q?expiry=1552176000000&hmac=LEb3Y6ja93I-sBpOOv3jar40XHl0iRhstrRjePxafe0&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Matrix Vector Multiplication

### Lecture Notes


-------------------------------


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Matrix Matrix Multiplication

### Lecture Notes


-------------------------------


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Matrix Multiplication Properties

### Lecture Notes


-------------------------------


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Inverse and Transpose

### Lecture Notes


-------------------------------


### Lecture Video

<video src="url" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="subtitle" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Review

### Lecture Slides

#### ML:Linear Algebra Review

Khan Academy has excellent [Linear Algebra Tutorials](https://www.khanacademy.org/#linear-algebra)

#### Matrices and Vectors

Matrices are 2-dimensional arrays:

$$\left [ \begin{array}{ccc}
    a & b & c \\ d & e & f \\ g & h & i \\ j & k & l
\end{array} \right ]$$

The above matrix has four rows and three columns, so it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:

$$\left [ \begin{array}{c} w \\ x\\ y\\ z \end{array} \right ]$$

So vectors are a subset of matrices. The above vector is a 4 x 1 matrix.

__Notation and terms:__

+ $A_{ij}$ refers to the element in the ith row and jth column of matrix $A$.
+ A vector with 'n' rows is referred to as an 'n'-dimensional vector
+ $v_i$ refers to the element in the ith row of the vector.
+ In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
+ Matrices are usually denoted by uppercase names while vectors are lowercase.
+ "Scalar" means that an object is a single value, not a vector or matrix.
+ $\mathbb{R}$ refers to the set of scalar real numbers
+ $\mathbb{R^n}$ refers to the set of n-dimensional vectors of real numbers


#### Addition and Scalar Multiplication

Addition and subtraction are __element-wise__, so you simply add or subtract each corresponding element:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] + \left [ \begin{array}{cc} w & x \\ y & z \end{array} \right ] = \left [ \begin{array}{cc} a+w & b+x \\ c+y & d+z \end{array} \right ]$$

To add or subtract two matrices, their dimensions must be __the same__.

In scalar multiplication, we simply multiply every element by the scalar value:

$$\left [ \begin{array}{cc} a & b \\ c & d \end{array} \right ] * x = \left [ \begin{array}{cc} a*x & b*x \\ c*x & d*x \end{array} \right ]$$


#### Matrix-Vector Multiplication

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$$ \left [ \begin{array}{cc} a & b \\ c & d \\ e & f \end{array} \right ] * \left [ \begin{array}{c} x \\ y \end{array} \right ] = \left [ \begin{array}{cc} a*x + b*y \\ c*x + d*y \\ e*x+f*y \end{array} \right ]$$

The result is a __vector__. The vector must be the __second__ term of the multiplication. The number of __columns__ of the matrix must equal the number of __rows__ of the vector.

An __m x n matrix__ multiplied by an __n x 1__ vector results in an __m x 1__ vector.


#### Matrix-Matrix Multiplication

We multiply two matrices by breaking it into several vector multiplications and concatenating the result

$$\left [ \begin{array}{cc} a & b \\ c & d \\ e & f \end{array} \right ] * \left [ \begin{array}{cc} w & x \\ y & z \end{array} \right ] = \left [ \begin{array}{cc} a*w + b*y & a*x+b*z \\ c*w+d*y & dc*x+d*z \\ e*w+f*y & e*x+f*z \end{array} \right ] $$

An __m x n matrix__ multiplied by an __n x o matrix__ results in an __m x o__ matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

To multiply two matrices, the number of __columns__ of the first matrix must equal the number of __rows__ of the second matrix.

#### Matrix Multiplication Properties

+ Not commutative. $A∗B \neq B∗A$
+ Associative. $(A∗B)∗C=A∗(B∗C)$

The __identity matrix__, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$$\left [ \begin{array}{ccc} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{array} \right ] $$

When multiplying the identity matrix after some matrix $(A∗I)$, the square identity matrix should match the other matrix's __columns__. When multiplying the identity matrix before some other matrix $(I∗A)$, the square identity matrix should match the other matrix's __rows__.


#### Inverse and Transpose

The __inverse__ of a matrix A is denoted A−1. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the pinv(A) function [1] and in matlab with the inv(A) function. Matrices that don't have an inverse are singular or degenerate.

The __transposition__ of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. We can compute transposition of matrices in matlab with the transpose(A) function or A':

$$A = \left [ \begin{array}{cc} a & b \\ c & d \\ e & f \end{array} \right ]$$

$$ A^T = \left [ \begin{array}{ccc} a & c & e \\ b & d & f \end{array} \right ]$$



### Errata

#### Linear Algebra Review

+ Matrix-Matrix Multiplication: 7:14 to 7:33 - While exploring a matrix multiplication, Andrew solved the problem correctly below, but when he tried to rewrite the answer in the original problem, one of the numbers was written incorrectly. The correct result was (matrix 9 15) and (matrix 7 12), but when it was rewritten above it was written as (matrix 9 15) and (matrix 4 12). The 4 should have been a 7. (Thanks to John Kemp and others). This has been partially corrected in the video - third subresult matrix shows 7 but the sound is still 4 for both subresult and result matrices. Subtitle at 6:48 should be “two is seven and two”, and subtitle at 7:14 should be “seven twelve and you”.
+ 3.4: Matrix-Matrix Multiplication: 8:12 - Andrew says that the matrix on the bottom left shows the housing prices, but those are the house sizes as written above
+ 3.6: Transpose and Inverse: 9:23 - While demonstrating a transpose, an example was used to identify B(subscript 12) and A(subscript 21). The correct number 3 was circled in both cases above, but when it was written below, it was written as a 2. The 2 should have been a 3. (Thanks to John Kemp and others)
Addition and scalar multiplication video
Spanish subtitles for this video are wrong. Seems that those subtitles are from another video.



## Practice Quiz: Linear Algebra




