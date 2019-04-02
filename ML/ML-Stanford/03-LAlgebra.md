# Linear Algebra Review

## Matrices and Vectors

### Lecture Notes

+ __Matrix__: Rectangular bmatrix of numbers:

    $$\begin{bmatrix} 1420 & 191 \\ 1271 & 821 \\         949 & 1437 \\ 147 & 1448 \end{bmatrix} \Longrightarrow \mathbb{R}^{4 \times 2}, 4 \times 2 \text{ matrix} \qquad \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \Longrightarrow \mathbb{R}^{2 \times 3}, 2 \times 3 \text{ matrix}$$

    __Dimension of matrix__: number of rows x number oof 

+ IVQ: Which of the following statements are true? Check all that apply.

    1. $\displaystyle \begin{bmatrix} 1 & 2 \\ 4 & 0 \\ 0 & 1 \end{bmatrix}$ is a $3\times2$ matrix.
    2. $\displaystyle \begin{bmatrix} 0 & 1 & 4 & 2 \\ 3 & 4 & 0 & 9 \end{bmatrix}$ is a $x \times 2$ matrix
    3. $\begin{bmatrix} 0 & 4 & 2 \\ 3 & 4 & 9 \\ 5 & -1 & 0 \end{bmatrix}$ is a $3 \times 3$ matrix
    4. $\begin{bmatrix} 1 & 2 \end{bmatrix}$ is a $1 \times 2$ matrix

    Ans: 134

+ __Matrix Elements__ (entries of matrix)

    $$A = \begin{bmatrix} 1420 & 191 \\ 1271 & 821 \\ 949 & 1437 \\ 147 & 1448 \end{bmatrix} $$  

    $A_{ij} = \text{"}i, j \text{entry"}$ in the $i^{th}$ row, $j^{th}$ column

    $A_{11} = 1402, A_{12} = 192, A_{32} = 1437, A_{41} = 147$ and $A_{43} =$ undefined (error)

+ IVQ: Let A be a matrix shown below. $A_{32}$ is one of the elements of this matrix.

    $$A = \begin{bmatrix} 85 & 76 & 66 & 5 \\ 94 & 75 & 18 & 28 \\ 68 & 40 & 71 & 5 \end{bmatrix}$$

    What is the value of $A_{32}$?

    1. 18
    2. 28
    3. 76
    4. 40

    Ans: 4


+ __Vector__: an $n x 1$ matrix

    $$y = \begin{bmatrix} 460 \\ 232 \\ 315 \\ 178 \end{bmatrix} \Longrightarrow \mathbb{R}^4, n = 4, 4-\text{dimensional  vector}$$

    + $y_i = i^{th}$ element: $y_1 = 460, y_2 = 232, y_3 = 315$
    + 1-indexed vs 0-indexed:

        $$y = \begin{bmatrix} y_1 \\ y_2 \\y_3 \\ y_4 \end{bmatrix} \Longrightarrow y[1], \text{1-indexed} \qquad\qquad y = \begin{bmatrix} y_0 \\ y_1 \\y_2 \\ y_3 \end{bmatrix} \Longrightarrow y[0], \text{0-indexed}$$
    + Notation: $A, B, C, D, \ldots$ as vector, $a, b, c, d, \ldots$ as scalar/number or element


-------------------------------

Matrices are 2-dimensional matrix:

$$\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \\ j & k & l \end{bmatrix}$$

The above matrix has four rows and three columns, so it is a $4 \times 3$ matrix.

A vector is a matrix with one column and many rows:

$$\begin{bmatrix}  w \\ x \\ y \\ z \end{bmatrix}$$

So vectors are a subset of matrices. The above vector is a $4 \times 1$ matrix.

Notation and terms:

+ $A_{ij}$ refers to the element in the $i^{th}$ row and $j^{th}$ column of matrix $A$.
+ A vector with 'n' rows is referred to as an 'n'-dimensional vector.
+ $v_i$ refers to the element in the $i^{th}$ row of the vector.
+ In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the bmatrixs are 0-indexed.
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

    $$\begin{bmatrix} 1 & 0 \\ 2 & 5 \\3 & 1 \end{bmatrix} \quad + \quad\begin{bmatrix} 4 & 0.5 \\ 2 5 \\ 0 & 1 \end{bmatrix} \quad = \quad \begin{bmatrix} 5 & 0.5 \\ 4 & 3 \\ 3 & 2 \end{bmatrix}$$

    $$3 \times 2 \text{ matrix   } + 3 \times 2 \text{ matrix   } = 3 \times 2 \text{ matrix   }$$

    <br/>

    $$\begin{bmatrix} 1 & 0 \\ 2 & 5 \\3 & 1 \end{bmatrix} \quad + \quad \begin{bmatrix} 4 & 0.5 \\ 2 & 5 \end{bmatrix} \quad = \quad \text{ error }$$

    $$3 \times 2 \text{ matrix    } + 2 \times 2 \text{ matrix     } = \text{error}$$

+ IVQ: what is $\begin{bmatrix} 8 & 6 & 9 \\ 10 & 1 & 10 \end{bmatrix} + \begin{bmatrix} 3 & 10 & 2 \\ 6 & 1 &-1 \end{bmatrix}$?

    Ans: $\begin{bmatrix} 11 & 16 & 11 \\ 16 & 2 7 9 \end{bmatrix}$

+ Scalar Multiplication

    $$3 \times \begin{bmatrix} 1 & 0 \\ 2 & 5 \\3 & 1 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 6 & 15 \\ 9 & 3 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2 & 5 \\3 & 1 \end{bmatrix} \times 3$$

    <br/>

    $$\begin{bmatrix} 4 & 0 \\ 6 & 3 \end{bmatrix} / 4 = \frac{1}{4} \times \begin{bmatrix} 4 & 0 \\ 6 & 3 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2/3 & 3/4 \end{bmatrix}$$

+ IVQ: What is $2 \times \begin{bmatrix} 4 & 5 \\ 1 & 7 \end{bmatrix}$?

    Ans: $\begin{bmatrix} 8 & 10 \\ 2 & 14 \end{bmatrix}$

+ Combination of Operands

    $$3 \times \begin{bmatrix} 1\\ 4 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 5 \end{bmatrix} - \begin{bmatrix} 3 \\ 0 \\ 2 \end{bmatrix} / 3 = \begin{bmatrix} 3\\12\\6 \end{bmatrix} + \begin{bmatrix} 0\\0\\5 \end{bmatrix} - \begin{bmatrix} 1\\0\\2/3 \end{bmatrix} = \begin{bmatrix} 2\\12\\10 \frac{1}{3} \end{bmatrix}$$
    <br/>
    + Operation order: scalar multiplication & scalar division > matrix/vector addition & subtraction
    + 3 x 1 matrix; 3-dimensional vector

+ IVQ: What is $\begin{bmatrix} 4\\6\\7 \end{bmatrix} / 2 - 3 \times \begin{bmatrix} 3\\1\\0 \end{bmatrix}$?

    Ans: $\begin{bmatrix} -4\\0\\3.5 \end{bmatrix}$


-------------------------------

Addition and subtraction are __element-wise__, so you simply add or subtract each corresponding element:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} + \begin{bmatrix} w & x \\ y & z \end{bmatrix} = \begin{bmatrix} a+w & b+x \\ c+y & d+z \end{bmatrix}$$

Subtracting Matrices:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} - \begin{bmatrix} w & x \\ y & z \end{bmatrix} = \begin{bmatrix} a-w & b-x \\ c-y & d-z \end{bmatrix}$$

To add or subtract two matrices, their dimensions must be __the same__.

In scalar multiplication, we simply multiply every element by the scalar value:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} * x = \begin{bmatrix} a*x & b*x \\ c*x & d*x \end{bmatrix}$$

In scalar division, we simply divide every element by the scalar value:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} / x = \begin{bmatrix} a/x & b/x \\ c/x & d/x \end{bmatrix}$$

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

+ Matrix-Vector Manipulation

    $$\qquad\quad\; A \qquad\quad\; \times \quad x \quad\; = \quad y$$

    <br/>

    $$\begin{bmatrix} a_{11} & \cdots & a_{1n} \\ \vdots &  \ddots & \vdots \\ a_{m1} & \cdots & a_{mn} \end{bmatrix} \times \begin{bmatrix} x_1 \\ \vdots \\x_{n} \end{bmatrix} = \begin{bmatrix} y_1 \\ \vdots \\y_{m} \end{bmatrix}$$

    $m \times n$ matrix ($m$ rows, $n$ columns) $\times n \times 1$ matrix ($n$-dimensional vector) = $m$-dimensional vector

    To get $y_i$, multiply $A$'s $i^{th}$ row with elements of vector $x$, and add them up, i.e.,

    $$y_i = \sum_{j=1}^n a_{ij} * x_j, \;\; \text{for } i = 1, 2, \ldots, m$$

+ Example 1

    $$\begin{bmatrix} 1 & 3 \\ 4 & 0 \\2 & 1 \end{bmatrix} \quad \times \quad \begin{bmatrix} 1 \\ 5 \end{bmatrix} \quad = \quad \begin{bmatrix} 16\\4\\7 \end{bmatrix}$$

    $$3 \times 2 \text{ matrix   }  * 2 \times 1 \text{ vector   } = 3 \times 1 \text{ vector}$$

    $1 * 1 + 3 * 5 = 16$; $4 * 1 + 0 * 5 = 4$; $2 * 1 + 1 * 5 = 7$

+ Example 2
    + IVQ: Consider the product of these two matrices:$\begin{bmatrix} 1 & 2 & 1 & 5 \\ 0 & 3 & 0 & 4 \\-1 & -2 & 0 & \end{bmatrix} \begin{bmatrix} 1\\ 3\\2\\1 \end{bmatrix}$

        What is the dimension of the product?

        Ans: $3 \times 1$
    + Calculation

        $\begin{bmatrix} 1&2&1&5 \\ 0&3&0&4 \\ -1 & -2 & 0 & 0 \end{bmatrix} \; \begin{bmatrix} 1 \\ 3\\2\\1 \end{bmatrix}  = \begin{bmatrix} 1*1+2*3+1*2+5*1 \\ 0*1+3*3+0*2+4*1 \\ (-1)*1+(-2)*3+0*2+0*1 \end{bmatrix} = \begin{bmatrix} 14\\13\\-7 \end{bmatrix}$
    + IVQ: What is $\begin{bmatrix} 1 & 0 & 3 \\ 2 & 1 & 5 \\ 3 & 1 7 2 \end{bmatrix} \times \begin{bmatrix} 1\\6\\2 \end{bmatrix}$?

        Ans: $\begin{bmatrix} 7 \\ 18 \\ 13 \end{bmatrix}$

+ Example: House price
    + Hourse size: 2014, 1416, 1534, 853
    + Hypothesis function: $h_\theta (x) = -40 + 0.25x$
    + Predicted house price?

        $$\begin{bmatrix} h_\theta(2014) \\ h_\theta(1416) \\ h_\theta(1534) \\ h_\theta(852) \end{bmatrix} = \begin{bmatrix} 1 & 2104 \\ 1 & 1416 \\ 1 & 1534 \\ 1 & 852 \end{bmatrix} \begin{bmatrix} -40 \\ 0.25 \end{bmatrix} = \begin{bmatrix} -40 + 0.25 * 2104 \\ -40 + 0.25 * 1416 \\ -40 + 0.25 * 1534 \\ -40 + 0.25 * 852 \end{bmatrix}$$
    + Prediction ($4$-dimensional vector) = Data matrix ($4 \times 1$ matrix) * parameters ($2 \times 1$ matrix, $2$-dimsional vector)


-------------------------------

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$$\begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix} * \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} a*x+b*y \\ c*x+d*y \\ e*x+f*y \end{bmatrix}$$

The result is a __vector__. The number of __columns__ of the matrix must equal the number of __rows__ of the vector.

An __m x n matrix__ multiplied by an __n x 1__ vector results in an __m x 1 vector__.

Below is an example of a matrix-vector multiplication. Make sure you understand how the multiplication works. Feel free to try different matrix-vector multiplications.

```matlab
% Initialize matrix A 
A = [1, 2, 3; 4, 5, 6;7, 8, 9] 

% Initialize vector v 
v = [1; 1; 1] 

% Multiply A * v
Av = A * v
```

### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/03.3-V2-LinearAlgebraReview%28Optional%29-MatrixVectorMultiplication.20caffb0b22b11e4901abd97e8288176/full/360p/index.mp4?Expires=1552176000&Signature=ABvoiuScr9H2CZEp69aeZrBF-afCrM54rbgwfKpYl-5qBNnsyBWehvUHydq6jo3sb3vpuV7T5RpRjq03Vui0WJCur0TY6pWZ39NcVFlAKxrVpx5M2D2XiZQPjD-2YpnAV4WBHQOoo0cbbQb2KYfTlhkYqeYNXyFde5oXKdF~Xr4_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/P654rbDkTI-ueK2w5AyPRg?expiry=1552176000000&hmac=lSYPRja0K2aBRvQLGdBInbAJ7vyFLldn8a_7J92TM8Q&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Matrix Matrix Multiplication

### Lecture Notes

+ Matrix-Matrix Multiplication

    $$A \qquad\qquad \times \qquad\qquad B \qquad\qquad = \qquad\qquad C$$

    <br/>

    $$\begin{bmatrix} a_{11} & \cdots & a_{1n} \\ \vdots &  \ddots & \vdots \\ a_{m1} & \cdots & a_{mn} \end{bmatrix} \times \begin{bmatrix} b_{11} & \cdots & b_{n1} \\ \vdots &  \ddots & \vdots \\ b_{o1} & \cdots & b_{on} \end{bmatrix} = \begin{bmatrix} c_{11} & \cdots & c_{1o} \\ \vdots &  \ddots & \vdots \\ c_{m1} & \cdots & c_{mo} \end{bmatrix}$$

    The $i^{th}$ column of the matrix $C$ is obtained by multiplying $A$ with the $i^{th}$ column of $B$. (for $i=1, 2, \ldots, o$), i.e., 

    $$c_{ij} = \sum_{k=1}^n a_{ik} * b_{lj}, \; \text{ for } i = 1, 2, \ldots, m, j = 1, 2, \ldots, o$$

+ Example 1

    $\begin{bmatrix} 1&3&2\\0&4&1 \end{bmatrix} * \begin{bmatrix} 1&3\\0&1\\5&2 \end{bmatrix} = \begin{bmatrix} 11&10\\9&14 \end{bmatrix}$

    $\begin{bmatrix} 1&3&2\\0&4&1 \end{bmatrix} * \begin{bmatrix} 1\\0\\5 \end{bmatrix} = \begin{bmatrix} 11\\9 \end{bmatrix}$

    $\begin{bmatrix} 1&3&2\\0&4&1 \end{bmatrix} * \begin{bmatrix} 3\\1\\2 \end{bmatrix} = \begin{bmatrix} 10\\14 \end{bmatrix}$

+ Example 2

    $\begin{bmatrix} 1&3\\2&5 \end{bmatrix} * \begin{bmatrix} 0&1\\3&2 \end{bmatrix} = \begin{bmatrix} 9&7\\15&12 \end{bmatrix}$

    $\begin{bmatrix} 1&3\\2&5 \end{bmatrix} * \begin{bmatrix} 0\\3 \end{bmatrix} = \begin{bmatrix} 9\\15 \end{bmatrix}$

    $\begin{bmatrix} 1&3\\2&5 \end{bmatrix} * \begin{bmatrix} 1\\2 \end{bmatrix} = \begin{bmatrix} 7\\12 \end{bmatrix}$

+ IVQ: In the equation $\begin{bmatrix} 1&3\\2&4\\0&5 \end{bmatrix} \begin{bmatrix} 1&0\\2&3 \end{bmatrix} = \begin{bmatrix} 7&9\\a&b\\c&d \end{bmatrix}$, what are $a, b, c, \text{ and } d$?

    Ans: 10, 12, 10, 15


+ Example: House Price with Different Hypothesis Function
    + Hourse size: 2014, 1416, 1534, 853
    + Hypothesis functions: 
        1. $h_\theta (x) = -40 + 0.25x$
        2. $h_\theta (x) = 200 + 0.1x$
        3. $h_\theta (x) = -150 + 0.4x$
    + Predicted house price?

        $\begin{bmatrix} 1 & 2104 \\ 1 & 1416 \\ 1 & 1534 \\ 1 & 852 \end{bmatrix} \times \begin{bmatrix} -40 & 200 & -150 \\ 0.25 & 0.1 & 0.4 \end{bmatrix} = \begin{bmatrix} 486 & 410 & 692 \\ 314 & 342 & 416 \\ 344 & 353 & 464 \\ 172 & 284 & 191 \end{bmatrix}$
    + Prediction of the $h_\theta$: <br/>
        First = $\begin{bmatrix} 486 \\ 314 \\ 344 \\ 172 \end{bmatrix} \qquad$ 
        Second = $\begin{bmatrix} 410 \\ 342 \\ 353 \\ 285 \end{bmatrix} \qquad$
        Third = $\begin{bmatrix} 692 \\ 416 \\ 464 \\ 191 \end{bmatrix}$


-------------------------------

We multiply two matrices by breaking it into several vector multiplications and concatenating the result.

$$\begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix} * \begin{bmatrix} w & x \\ y & z \end{bmatrix} = \begin{bmatrix} a*w+b*x & a*x + b*z \\ c*w +d*y & c*x+d*z \\ e*w+f*y & e*x+f*z \end{bmatrix}$$

An __m x n matrix__ multiplied by an __n x o matrix__ results in an __m x o matrix__. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

To multiply two matrices, the number of __columns__ of the first matrix must equal the number of __rows__ of the second matrix.

For example:

```matlab
% Initialize a 3 by 2 matrix
A = [1, 2; 3, 4; 5, 6]

% Initialize a 2 by 1 matrix
B = [1; 2]

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1)
mult_AB = A*B

% Make sure you understand why we got that result
```

### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/03.4-V2-LinearAlgebraReview%28Optional%29-MatrixMatrixMultiplication.0db57090b22b11e4bb7e93e7536260ed/full/360p/index.mp4?Expires=1552176000&Signature=kkEkt7ArgNPx8PAYNLMVp1GQ8wyJxVS9gkLNIgKGdeNJtE2lvTlQPTQdRN5H2B7rzlxiksDSDrYgXpOnkx3XiQv1LmtFIxyldYmgIk2~iqvczR5jvIdg5qpSRbNYPlHUMZ~2rLXnFHi~NyfK8fE~CZWtQvc7eba~M4XKOwZf1ho_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/HSCcS8wTEeabaQqy7nURwA?expiry=1552176000000&hmac=7psfifRZxPmZVthuwMx_reKjPC9cLL5mSaRAX-mjaao&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Matrix Multiplication Properties

### Lecture Notes

+ Commutative property
    + Let $A$ and $B$ be matrices.  Then in general $A \times B \neq B \times A \Longrightarrow \text{not commutative}$
    + Ssame dimension: Counter example

        $$\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 0 & 0 \\ 2 & 0 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 0 \end{bmatrix} \neq \begin{bmatrix} 0 & 0 \\ 2 & 2 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 2 & 0 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$$
    + Different dimensions: $A_{m \times n}$  and $B_{n \times m}$

        $$\begin{array}{rcl} A_{nm} \times B_{mn} = C_{nn} & \Longrightarrow & n \times n \text{ matrix} \\ B_{mn} \times A_{nm} = D_{mm} & \Longrightarrow & m \times m \text{ matrix} \end{array}$$

+ Associative property
    + Let $A$, $B$ and $C$ be matrices. $A \times (B \times C) = (A \times B) \times C \Longrightarrow \text{ associatve}$
    + Proof (simplified):
        + Let $D = B \times C$. Computer $A \times D$
        + Let $E = A \times B$. Computer $E \times C$
        + $A \times D = E \times C$

+ Identity Matrix
    + Denoted $I$ or $I_{n \times n}$; 1 is identity $\in \mathbb{R}$
    + Examples of identity matrices

        $I_{2\times2} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \quad\;\; I_{3\times 3} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \quad\;\; I_{4\times 4} = \begin{bmatrix} 1 & 0 & 0 & 0\\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0 \\ 0&0&0&1\end{bmatrix} \quad\;\; I_{n\times n} = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1\end{bmatrix}$
    + For any matrix $\displaystyle A_{m \times n}$, $\displaystyle \quad\;A_{m \times n} \cdot I_{n \times n} =  I_{m \times m} \cdot A_{m \times n} = A_{m \times n}$
    + NB: 
        + In general: $AB \neq BA$
        + Identity matrix: $AI = IA$
    + IVQ: what is $\begin{bmatrix} 1&0&0\\0&1&0\\0&0&1 \end{bmatrix} \times \begin{bmatrix} 1\\3\\2 \end{bmatrix}$?

        Ans: $\begin{bmatrix} 1\\3\\2 \end{bmatrix}$

-------------------------------

+ Matrices are not commutative: $A∗B \neq B∗A$
+ Matrices are associative: $(A∗B)∗C=A∗(B∗C)$

The __identity matrix__, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$$\begin{bmatrix} 1&0&0\\0&1&0\\0&0&1 \end{bmatrix}$$

When multiplying the identity matrix after some matrix (A∗I), the square identity matrix's dimension should match the other matrix's __columns__. When multiplying the identity matrix before some other matrix (I∗A), the square identity matrix's dimension should match the other matrix's __rows__.

```matlab
% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 

% Note that IA = AI but AB != BA
```


### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/03.5-V2-LinearAlgebraReview%28Optional%29-MatrixMultiplicationProperties.c0b73ed0b22b11e4960bf70a8782e569/full/360p/index.mp4?Expires=1552176000&Signature=QU2PpmNkGFz8BZL2fcINY2ylYs2Gxh4I4CIDtTYkpSVK2gKie6n-4OK6KkWLaJFPLWrE9HyT1Zeoi4xpfVZIHU5toL4KXCFelvgqDeK5I0UYHEvGVonWpVgtUGlWLboGjX0mfZKqy9xVHxYWc0niOaCYkGAvsc-3qo17FyqIx2U_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/acd9p-bdQjaHfafm3QI2yg?expiry=1552176000000&hmac=zOMAZT8_vPdljvQa-i2Tt-WQkmyJ90-YOy7EVa8w9-w&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Inverse and Transpose

### Lecture Notes

+ Matrix Inverse
    + Inverse in real number $\mathbb{R}$:
        + Identity: 1
        + Inverse: $3 * 3^{-1} = 3 * 1/3 = 1$
        + Not all numbers have an inverse: $0^{-1}$ undefined
    + If $A$ is an $m \times m$ matrix (squared matrix w/ $m$ rows and $m$ columns), and if it has an inverse, $\quad\; AA^{-1} = A^{-1}A = I$
    + Example: 

        $$\begin{bmatrix} 3&4\\2&16 \end{bmatrix} \begin{bmatrix} 0.4&-0.1\\-0.05&0.075 \end{bmatrix} = \begin{bmatrix} 1&0\\0&1 \end{bmatrix} = I_{2\times2}$$
    + __singular matrix__ / __degenerate matrix__: Matrices that don't have an inverse

+ Matrix Transpose
    + Let $A$ ba an $m\times n$ matrix, and let $B = A^T$. Then $B$ is an $n\times m$ matrix, and 

        $$B_{ij} = A_{ji}$$
    + Example: $A = \begin{bmatrix} 1&2&0\\3&5&9 \end{bmatrix} \qquad B = A^T = \begin{bmatrix} 1&3\\2&5\\0&9 \end{bmatrix}$
    + IVQ: What is $\begin{bmatrix} 0&3\\1&4 \end{bmatrix}^T$?

        Ans: $\begin{bmatrix} 0&1\\3&4 \end{bmatrix}$


-------------------------------

The __inverse__ of a matrix A is denoted $A^{-1}$. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the `pinv(A)` function and in Matlab with the `inv(A)` function. Matrices that don't have an inverse are singular or degenerate.

The __transposition__ of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. We can compute transposition of matrices in matlab with the transpose(A) function or A':

$$ A = \begin{bmatrix} a&b\\c&d\\e&f \end{bmatrix} \Longrightarrow A^T = \begin{bmatrix} a&b&c\\d&e&f \end{bmatrix}$$

In other words: $\quad\;\; A_{ij} = A^T_{ji}$

```matlab
% Initialize matrix A
A = [1,2,0;0,5,6;7,0,9]

% Transpose A
A_trans = A'

% Take the inverse of A
A_inv = inv(A)

% What is A^(-1)*A?
A_invA = inv(A)*A
```


### Lecture Video

<video src="https://d3c33hcgiwev3.cloudfront.net/03.6-V2-LinearAlgebraReview%28Optional%29-InverseAndTranspose.114b8fa0b22b11e48803b9598c8534ce/full/360p/index.mp4?Expires=1552176000&Signature=XUKTMESbISzW~31sJq3Iep2iFlecm1EwD2PTc5BBa1Kw1Bp2E7EvzcFYIUZ25f~8ofcl4J2m7wPXLuEt62-j6DdEOcawjgHSOX3GiRWaPyKM8-Tzc7cVRFEW9pjqIPt8wnXcqdZ5BowhBfR-HcF3dj1qf23r9vjWJUt~E36oypE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A" preload="none" loop="loop" controls="controls" style="margin-left: 2em;" muted="" poster="http://www.multipelife.com/wp-content/uploads/2016/08/video-converter-software.png" width="180">
  <track src="https://www.coursera.org/api/subtitleAssetProxy.v1/7AlKE8wVEeax7Q7v7IdiLA?expiry=1552176000000&hmac=z0pFXpTljgJAe0ykTeYhOPmc22boLrjIstBEjTyYRBw&fileExtension=vtt" kind="captions" srclang="en" label="English" default>
  Your browser does not support the HTML5 video element.
</video>

<br/>


## Review

### Lecture Slides

#### ML:Linear Algebra Review

Khan Academy has excellent [Linear Algebra Tutorials](https://www.khanacademy.org/#linear-algebra)

#### Matrices and Vectors

Matrices are 2-dimensional bmatrixs:

$$\begin{bmatrix}
    a & b & c \\ d & e & f \\ g & h & i \\ j & k & l
\end{bmatrix}$$

The above matrix has four rows and three columns, so it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:

$$\begin{bmatrix} w \\ x\\ y\\ z \end{bmatrix}$$

So vectors are a subset of matrices. The above vector is a 4 x 1 matrix.

__Notation and terms:__

+ $A_{ij}$ refers to the element in the ith row and jth column of matrix $A$.
+ A vector with 'n' rows is referred to as an 'n'-dimensional vector
+ $v_i$ refers to the element in the ith row of the vector.
+ In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the bmatrixs are 0-indexed.
+ Matrices are usually denoted by uppercase names while vectors are lowercase.
+ "Scalar" means that an object is a single value, not a vector or matrix.
+ $\mathbb{R}$ refers to the set of scalar real numbers
+ $\mathbb{R^n}$ refers to the set of n-dimensional vectors of real numbers


#### Addition and Scalar Multiplication

Addition and subtraction are __element-wise__, so you simply add or subtract each corresponding element:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} + \begin{bmatrix} w & x \\ y & z \end{bmatrix} = \begin{bmatrix} a+w & b+x \\ c+y & d+z \end{bmatrix}$$

To add or subtract two matrices, their dimensions must be __the same__.

In scalar multiplication, we simply multiply every element by the scalar value:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} * x = \begin{bmatrix} a*x & b*x \\ c*x & d*x \end{bmatrix}$$


#### Matrix-Vector Multiplication

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$$ \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix} * \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} a*x + b*y \\ c*x + d*y \\ e*x+f*y \end{bmatrix}$$

The result is a __vector__. The vector must be the __second__ term of the multiplication. The number of __columns__ of the matrix must equal the number of __rows__ of the vector.

An __m x n matrix__ multiplied by an __n x 1__ vector results in an __m x 1__ vector.


#### Matrix-Matrix Multiplication

We multiply two matrices by breaking it into several vector multiplications and concatenating the result

$$\begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix} * \begin{bmatrix} w & x \\ y & z \end{bmatrix} = \begin{bmatrix} a*w + b*y & a*x+b*z \\ c*w+d*y & dc*x+d*z \\ e*w+f*y & e*x+f*z \end{bmatrix} $$

An __m x n matrix__ multiplied by an __n x o matrix__ results in an __m x o__ matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

To multiply two matrices, the number of __columns__ of the first matrix must equal the number of __rows__ of the second matrix.

#### Matrix Multiplication Properties

+ Not commutative. $A∗B \neq B∗A$
+ Associative. $(A∗B)∗C=A∗(B∗C)$

The __identity matrix__, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} $$

When multiplying the identity matrix after some matrix $(A∗I)$, the square identity matrix should match the other matrix's __columns__. When multiplying the identity matrix before some other matrix $(I∗A)$, the square identity matrix should match the other matrix's __rows__.


#### Inverse and Transpose

The __inverse__ of a matrix A is denoted A−1. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the pinv(A) function [1] and in matlab with the inv(A) function. Matrices that don't have an inverse are singular or degenerate.

The __transposition__ of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. We can compute transposition of matrices in matlab with the transpose(A) function or A':

$$A = \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix}$$

$$ A^T = \begin{bmatrix} a & c & e \\ b & d & f \end{bmatrix}$$



### Errata

#### Linear Algebra Review

+ Matrix-Matrix Multiplication: 7:14 to 7:33 - While exploring a matrix multiplication, Andrew solved the problem correctly below, but when he tried to rewrite the answer in the original problem, one of the numbers was written incorrectly. The correct result was (matrix 9 15) and (matrix 7 12), but when it was rewritten above it was written as (matrix 9 15) and (matrix 4 12). The 4 should have been a 7. (Thanks to John Kemp and others). This has been partially corrected in the video - third subresult matrix shows 7 but the sound is still 4 for both subresult and result matrices. Subtitle at 6:48 should be “two is seven and two”, and subtitle at 7:14 should be “seven twelve and you”.
+ 3.4: Matrix-Matrix Multiplication: 8:12 - Andrew says that the matrix on the bottom left shows the housing prices, but those are the house sizes as written above
+ 3.6: Transpose and Inverse: 9:23 - While demonstrating a transpose, an example was used to identify B(subscript 12) and A(subscript 21). The correct number 3 was circled in both cases above, but when it was written below, it was written as a 2. The 2 should have been a 3. (Thanks to John Kemp and others)
Addition and scalar multiplication video
Spanish subtitles for this video are wrong. Seems that those subtitles are from another video.



## Practice Quiz: Linear Algebra

1. Let two matrices be $A = \begin{bmatrix} 1&−4\\-2&1 \end{bmatrix}, \qquad\quad\; B = \begin{bmatrix} 0&3\\5&8 \end{bmatrix}$. What is $A - B$?

    Ans: $\begin{bmatrix} 1&-7\\-7&-7 \end{bmatrix}$


2. Let $x = \begin{bmatrix} 8\\2\\5\\1 \end{bmatrix}$.  Whatis $2*x$?

    Ans: $\begin{bmatrix} 16\\4\\10\\2 \end{bmatrix}$<br/>
    To multiply the vector x by 2, take each element of x and multiply that element by 2.


3. Let $u$ be a 3-dimensional vector, where specifically $u = \begin{bmatrix} 8\\1\\4 \end{bmatrix}$.  What is $u^T$?

    Ans: $\begin{bmatrix} 8&1&4 \end{bmatrix}$


4. Let $u$ and $v$ be 3-dimensional vectors, where specifically $u = \begin{bmatrix} -3\\4\\3 \end{bmatrix}$ and $v = \begin{bmatrix} 3 \\1\\5 \end{bmatrix}$. What $u^Tv$ is?

    Ans: 10


5. Let $A$ and $B$ be 3x3 (square) matrices. Which of the following must necessarily hold true? Check all that apply.

    1. If $A$ is the 3x3 identity matrix, then $A * B = B * A$
    2. $A * B = B * A$
    3. $A + B = B + A$
    4. $A*B*A = B*A*B$

    Ans: 13
    1. Even though matrix multiplication is not commutative in general ($A*B \neq B*A$ for general matrices $A$, $B$), for the special case where $A=I$, we have $A*B = I*B = B$, and also $B*A = B*I = B$. So, $A*B = B*A$.
    3. We add matrices element-wise. So, this must be true.




