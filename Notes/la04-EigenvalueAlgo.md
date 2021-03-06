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

+ Normal, Hermitian, and real-symmetric matrices
  + adjoint matrix
    + the matrix of cofactors of the transpose
    + $M^\ast$ of a complex matrix $M$ = transpose of the conjugate of $M$: $M^\ast = \overline{M}^T$
  + normal matrix
    + a square matrix $A$ if it commutes w/ its adjoint
    + $A^\ast A = A A^\ast$
  + Hermitian matrix
    + $A^\ast = A$
    + normal matrix
    + $A \in \Bbb{R}^{n \times n} \implies$ the adjoint = its transpose
    + $\iff$ symmetric
  + column vectors: the adjoint used to define the canonical inner product in $\Bbb{C}^n: \bf{w} \cdot \bf{v} = \bf{w}^\ast \bf{v}$
  + properties of normal, Hermitian, and real-symmetric matrices
    + every generalized eigenvector of a normal matrix $\implies$  ordinary eigenvector
    + Jordan normal form w/ diagonal $\implies$ any normal matrix similar to a diagonal matrix
    + eigenvectors of distinct eigenvalues of a normal matrix $\implies$ orthogonal
    + null space and the image (or column space) of a normal matrix $\implies$ orthogonal to each other
    + $\forall$ normal matrix $A, \;\Bbb{C}^n$ w/ an orthonormal basis consisting of eigenvectors of $A \implies$ corresponding matrix of eigenvectors __initary__
    + eigenvalues of a Hermitian matrix $\in \Bbb{R} \implies (\overline{\lambda} - \lambda) \bf{v} = (A^\ast - A) \bf{v} = (A - A)\bf{v} = 0 \;\;\forall\; \bf{v} \ne \bf{0}$
    + $A \in \Bbb{R}^n, \exists$ an orthonormal basis for $\Bbb{R}^n$ consisting of eigenvectors of $A \iff A$ symmetric


## Condition number

+ Condition number
  + the evaluation of some function $f$ for some input $x$
  + $\kappa(f, x)$:
    + the ratio of the relative error in the function's output to the relative error in the input
    + varying w/ both the function and the input
    + how error growing during the calculation
  + base-10 logarithm: how many fewer digits of accuracy exist in the result than existed in the input
  + a best-case scenario: reflecting the instability built into the problem, regardless if how it solved
  + no algorithms able to produce more accurate results than indicated by the condition number, except by chance
  + possibly very ill-conditioned problem to find the roots of a polynomial

+ Operator normal
  + solving the linear equation $A\bf{v} = \bf{b}$ w/ invertible $A$
  + the condition number $\kappa(A^{-1}, \bf{b}) = \|A\|_{\operatorname{op}}\|A^{-1}\|_{\operatorname{op}}$
  + $\|\;\|_{\operatorname{op}}$: the __operator norm__ subordinate to the normal Euclidean nor on $\Bbb{C}^n$
  + $\kappa(A)$ of the matrix $A$
    + $\kappa(A, \bf{b})$ independent of $\bf{b}$ and the same for $A$ and $A^{-1}$
    + the absolute value of the ratio of the largest eigenvalue of $A$ to its smallest
  + unitary $A \implies \|A\|_{\operatorname{op}} = \|A^{-1}\|_{\operatorname{op}} = 1 \implies \kappa(A) = 1$
  + operator norm difficult to calculate for general matrix
  + matrix norm commonly used to estimate the condition number

+ Bauer-Fike theorem
  + $\exists\; A_{n \times n}$ diagonalizable w/ eigenvector matrix $V$
  + $\lambda$ as eigenvalue of $A \implies$ the absolute error in calculating $\lambda$ bounded by the product $\kappa(V)$ and the absolute error in $A$
  + the condition number for finding $\lambda$: $\kappa(\lambda, A) - \|V\|_{\operatorname{op}}\|V^{-1}\|_{\operatorname{op}}$
  + normal $A \implies$ unitary $V$ and $\kappa(\lambda, A) = 1$

+ Eigenspace
  + finding the eigenspace of a normal matrix $A$ corresponding to an eigenvalue $\lambda$
  + condition number inversely proportional to the minimum distance btw $\lambda$ and the other distinct eigenvalues of $A$
  + the eigenspace problem for normal matrices: well-conditioned for isolated eigenvalues
  + not-isolated eigenvalues: identifying the span of all eigenvectors of nearby eigenvalues


## Algorithms

+ General algorithm idea
  + algorithm for finding eigenvalues used to find the roots of polynormials
  + Abel-Ruffini theorem: algorithm w/ dimensions $\ge 4$
    + infinite space
    + involving functions of greater complexity than elementary arithmetic operations and fraction powers
  + algorithms exactly calculating eigenvalues in a finite number of steps only exist for a few special classes of matrix
  + iterative algorithms producing better approximate solutions w/ each iteration for general matrix
  + different algorithms producing different number of eigenvalues: all, a few or even only one
  + identified $\lambda$ used to
    + direct the algorithm toward a different solution next time
    + reduce the problem to one that no longer w/ $\lambda$ as a solution

+ Redirection methods
  + accomplished by shifting
  + replacing $A$ w/ $(A - \mu I) , \exists\; \text{ constant } \mu$
  + eigenvalue found for $(A - \mu I)$ must add back  in to get an eigenvalue for $A$
  + power iteration, $\mu = \lambda$
    + finding the largest eigenvalue in absolute value
    + $\lambda$ only an approximate eigenvalue $\implies$ unlikely to find it a second time
  + inverse iteration
    + finding the lowest eigenvalue
    + $\mu$ chosen well from $\lambda$
    + hopefully closer to some other eigenvalue

+ Reduction methods
  + accomplished by restricting $A$ to the column space of the matrix $(A - \lambda I)$
  + $(A - \lambda I)$ singular $\implies$ the column space w/ lesser dimension
  + applied to the restrict matrix
  + able to be repeated until all eigenvalues found

+ Inverse iteration based algorithms
  + applied to algorithm unable to produce eigenvectors
  + $\mu$ set to a close approximation tot he eigenvalue
  + quickly converging to the eigenvector iof the cloest eigenvalue to $\mu$
  + alternative for small matrix: finding the column space of the product of $(A - \lambda^\prime I), \;\forall\; \lambda^\prime$ (other eigenvalues)

+ Thompson's formula
  + applied to the norm of unit eigenvector compoenets of normal matrices
  + $\exists\; \text {normal } A_{n \times n}$
    + eigenvalues $\lambda_i(A)$
    + corresponding eigenvectors $\bf{v}_i$ w/ component entries $v_{i,j}$
  + $A_j$: an $n-1 \times n-1$ matrix by removing $j$th row and column from $A$
  + $\lambda_k(A_j)$: the $k$th eigenvalue
  + formula

    \[ |v_{i, j}|^2 \prod_{k=1, k\ne i} \big(\lambda_i(A) \lambda_k(A)\big) = \prod_{k=1}^{n-1} \big(\lambda_i(A) - \lambda_k(A_j)\big) \]

  + $p, p_j$: characteristic polynomials of $A$ and $A_j$

    \[ |v_{i, j}|^2 = \frac{p_j\big(\lambda_i (A)\big)}{p^\prime \big(\lambda_j(A)\big)}, \hspace{1em}\forall\;i \text{ s.t. }p^\prime\big(\lambda_i(A)\big) \ne 0 \]

## Hessenberg and tridiagonal matrices

+ Hessenberg matrices
  + the eigenvalues of triangular matrix: the diagonal elements
  + general matrices: $\nexists\;$ finite method, e.g., Gaussian elimination, to convert a matrix to triangular form while preserving eigenvalues
  + possible to reach something close to triangular
  + upper Hessenberg matrix: a square matrix w/ all entries below the subdiagonal are zero
  + lower Hessenberg matrix: a square matrix w/ all entries above the subtriangal are zero
  + upper & lower Hessenberg matrices: tridiagonal
  + Hessenberg and tridiagonal matrices: starting points for many eigenvalue algorithms
  + several methods commonly used to convert a general matrix into a Hessenberg matrix w/ the sam eeigenvalues
  + symmetric or Hermitian matrices $\to$ tridiagonal

+ Similarity matrix
  + eigenvalue only: no similarity matrix required
  + both eigenvalue and eigenvector:
    + similarity matrix required probably
    + transform the eigenvectors of the Hessenberg matrix back into eigenvectors of the original matrix

<table style="text-align: center">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Hessenberg_and_tridiagonal_matrices">Algorithms for Hessenberg & Tridiagonal Matrices</a></caption>
  <tbody><tr>
    <th>Method</th>
    <th>Applies to</th>
    <th>Produces</th>
    <th>Cost without similarity matrix</th>
    <th>Cost with similarity matrix</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Householder_transformation" title="Householder transformation">Householder transformations</a></td>
    <td>General</td>
    <td>Hessenberg</td>
    <td>​<sup>2<i>n</i><sup>3</sup></sup>⁄<sub>3</sub> + <i>O</i>(<i>n</i><sup>2</sup>)</td>
    <td>​<sup>4<i>n</i><sup>3</sup></sup>⁄<sub>3</sub> + <i>O</i>(<i>n</i><sup>2</sup>)</td>
    <td align="left">Reflect each column through a subspace to zero out its lower entries.</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Givens_rotation" title="Givens rotation">Givens rotations</a></td>
    <td>General</td>
    <td>Hessenberg</td>
    <td>​<sup>4<i>n</i><sup>3</sup></sup>⁄<sub>3</sub> + <i>O</i>(<i>n</i><sup>2</sup>)</td>
    <td></td>
    <td align="left">Apply planar rotations to zero out individual entries. Rotations are ordered so that later ones do not cause zero entries to become non-zero again.</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Arnoldi_iteration" title="Arnoldi iteration">Arnoldi iteration</a></td>
    <td>General</td>
    <td>Hessenberg</td>
    <td></td>
    <td></td>
    <td align="left">Perform Gram–Schmidt orthogonalization on Krylov subspaces.
    </td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Lanczos_algorithm" title="Lanczos algorithm">Lanczos algorithm</a></td>
    <td>Hermitian</td>
    <td>Tridiagonal</td>
    <td></td>
    <td></td>
    <td align="left">Arnoldi iteration for Hermitian matrices, with shortcuts.</td>
  </tr>
</tbody></table>

## Iterative algorithms

<table style="text-align: center">
  <caption style="font-size: 1.5em; margin: 0.2em;"><a href="https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Iterative_algorithms">Iterative Algorithms</a></caption>
  <tbody><tr>
    <th>Method</th>
    <th>Applies to</th>
    <th>Produces</th>
    <th>Cost per step</th>
    <th>Converg-ence</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Lanczos_algorithm" title="Lanczos algorithm">Lanczos algorithm</a></td><td>Hermitian</td><td> <i>m</i>  largest/smallest eigenpairs</td><td></td><td></td><td align="left"></td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Power_iteration" title="Power iteration">Power iteration</a></td><td>general</td><td>eigenpair with largest value</td><td><i>O</i>(<i>n</i><sup>2</sup>)</td><td>linear</td><td align="left">Repeatedly applies the matrix to an arbitrary starting vector and renormalizes.</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Inverse_iteration" title="Inverse iteration">Inverse iteration</a></td><td>general</td><td>eigenpair with value closest to μ</td><td></td><td>linear</td><td align="left">Power iteration for (<i>A</i> − <i>μI</i> )<sup>−1</sup></td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration" title="Rayleigh quotient iteration">Rayleigh quotient iteration</a></td><td>Hermitian</td><td>any eigenpair</td><td></td><td>cubic</td><td align="left">Power iteration for (<i>A</i> − <i>μ</i><sub><i>i</i></sub><i>I</i> )<sup>−1</sup>, where <i>μ</i><sub><i>i</i></sub> for each iteration is the Rayleigh quotient of the previous iteration.</td>
  </tr>
  <tr>
    <td width="200"><a href="/w/index.php?title=Preconditioned_inverse_iteration&amp;action=edit&amp;redlink=1" title="Preconditioned inverse iteration (page does not exist)">Preconditioned inverse iteration</a><sup id="cite_ref-14"></sup> or <a href="https://en.wikipedia.org/wiki/LOBPCG" title="LOBPCG">LOBPCG algorithm</a></td><td><a href="https://en.wikipedia.org/wiki/Positive-definite_matrix" title="Positive-definite matrix">positive-definite</a> real symmetric</td><td>eigenpair with value closest to μ</td><td></td><td></td><td align="left">Inverse iteration using a <a href="https://en.wikipedia.org/wiki/Preconditioner" title="Preconditioner">preconditioner</a> (an approximate inverse to <i>A</i>).</td>
  </tr>
  <tr>
    <td><a href="/w/index.php?title=Bisection_eigenvalue_algorithm&amp;action=edit&amp;redlink=1" title="Bisection eigenvalue algorithm (page does not exist)">Bisection method</a></td><td>real symmetric tridiagonal</td><td>any eigenvalue</td><td></td><td>linear</td><td align="left">Uses the <a href="https://en.wikipedia.org/wiki/Bisection_method" title="Bisection method">bisection method</a> to find roots of the characteristic polynomial, supported by the Sturm sequence.</td>
  </tr>
  <tr>
    <td><a href="/w/index.php?title=Laguerre_iteration&amp;action=edit&amp;redlink=1" title="Laguerre iteration (page does not exist)">Laguerre iteration</a></td><td>real symmetric tridiagonal</td><td>any eigenvalue</td><td></td><td>cubic<sup id="cite_ref-15"><a href="#cite_note-15">[12]</a></sup></td><td align="left">Uses <a href="https://en.wikipedia.org/wiki/Laguerre%27s_method" title="Laguerre's method">Laguerre's method</a> to find roots of the characteristic polynomial, supported by the Sturm sequence.</td>
  </tr>
  <tr>
    <td rowspan="2"><a href="https://en.wikipedia.org/wiki/QR_algorithm" title="QR algorithm">QR algorithm</a></td><td rowspan="2">Hessenberg</td><td>all eigenvalues</td><td><i>O</i>(<i>n</i><sup>2</sup>)</td><td rowspan="2">cubic</td><td align="left" rowspan="2">Factors <i>A</i> = <i>QR</i>, where <i>Q</i> is orthogonal and <i>R</i> is triangular, then applies the next iteration to <i>RQ</i>.</td>
  </tr>
  <tr>
    <td>all eigenpairs</td><td>6<i>n</i><sup>3</sup> + <i>O</i>(<i>n</i><sup>2</sup>)</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm" title="Jacobi eigenvalue algorithm">Jacobi eigenvalue algorithm</a></td><td>real symmetric</td><td>all eigenvalues</td><td><i>O</i>(<i>n</i><sup>3</sup>)</td><td>quadratic</td><td align="left">Uses Givens rotations to attempt clearing all off-diagonal entries. This fails, but strengthens the diagonal.</td>
  </tr>
  <tr>
    <td rowspan="2"><a href="https://en.wikipedia.org/wiki/Divide-and-conquer_eigenvalue_algorithm" title="Divide-and-conquer eigenvalue algorithm">Divide-and-conquer</a></td><td rowspan="2">Hermitian tridiagonal</td><td>all eigenvalues</td><td><i>O</i>(<i>n</i><sup>2</sup>)</td><td rowspan="2"></td><td align="left" rowspan="2">Divides the matrix into submatrices that are diagonalized then recombined.</td>
  </tr>
  <tr>
    <td>all eigenpairs</td><td>(​<sup>4</sup>⁄<sub>3</sub>)<i>n</i><sup>3</sup> + <i>O</i>(<i>n</i><sup>2</sup>)</td>
  </tr>
  <tr>
    <td><a href="/w/index.php?title=Homotopy_method&amp;action=edit&amp;redlink=1" title="Homotopy method (page does not exist)">Homotopy method</a></td><td>real symmetric tridiagonal</td><td>all eigenpairs</td><td><i>O</i>(<i>n</i><sup>2</sup>)</td><td></td><td align="left">Constructs a computable homotopy path from a diagonal eigenvalue problem.</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Folded_spectrum_method" title="Folded spectrum method">Folded spectrum method</a></td><td>real symmetric</td><td>eigenpair with value closest to μ</td><td></td><td></td><td align="left">Preconditioned inverse iteration applied to (<i>A</i> − <i>μI</i> )<sup>2</sup></td>
  </tr>
  <tr>
    <td><a href="/w/index.php?title=MRRR&amp;action=edit&amp;redlink=1" title="MRRR (page does not exist)">MRRR algorithm</a><sup id="cite_ref-17"></sup></td><td>real symmetric tridiagonal</td><td>some or all eigenpairs</td><td><i>O</i>(<i>n</i><sup>2</sup>)</td><td></td><td align="left">"Multiple relatively robust representations" – performs inverse iteration on a <a href="https://en.wikipedia.org/wiki/Cholesky_decomposition" title="Cholesky decomposition"><i>LDL</i><sup>T</sup> decomposition</a> of the shifted matrix.</td>
  </tr>
</tbody></table>


## Direct calculation

+ Triangular matrices
  + determinant of a triangular matrix = the product of its diagonal entries
  + triangular $T \implies \det(\lambda I - T) = \prod_i (\lambda  T_{ii})$
  + eigenvalues of $T$ = diagonal entries

+ Factorable polynomial equations
  + $\forall$ polynomial $p, p(A) = 0 \implies \lambda$ satisfies the same equation
  + $\forall\; p$ w/ a known factorization $\implies \lambda$ among its roots
  + projection: a square matrix $P$ satisfying $P^2 = P$
    + roots of the corresponding scalar polynomial equation, $\lambda^2 = \lambda$
    + any projection w/ 0 and 1 for its eigenvalues
    + the multiplicity of 0 as an eigenvalue: the nullity of $P$
    + the multiplicity of 1 as an eigenvalue: the rank of $P$
  + $\exists\; A \text{ and scalar } \alpha$ s.t. $A^2 = \alpha^2 I$
    + the eigenvalues: $\pm \alpha$
    + projection operator

      \[\begin{align*}
        & P_+ = \frac12 \left(I + \frac{A}{\alpha}\right) \hspace{2em} P_- = \frac12 \left(I - \frac{A}{\alpha}\right)\\
        &\text{satisfy}\\
        &AP_+ = \alpha P_+ \hspace{1em} AP_- = -\alpha P_-\\
        &P_+P_+ = P_+ \hspace{1em} P_- P_- = P_- \hspace{1em} P_+P_- = P_-P_+ = 0
      \end{align*}\]

    + column spaces of $P_+$ and $P_-$: the eigenspaces of $A$ corresponding to $+\alpha$ and $-\alpha$, respectively

+ 2-dim matrix
  + dim = 2~4: formula involving radicals used to find the eigenvalues
  + Consider general matrix $A_{2 \times 2}$

    \[ A = \begin{bmatrix} a&b\\c&d \end{bmatrix} \]

  + the characteristic polynomial

    \[ \det\begin{bmatrix} \lambda - a&-b\\-c&\lambda - d \end{bmatrix} = \lambda^2 - (a + d) \lambda + (ad - bc) = \lambda^2 - \lambda \operatorname{tr}(A) + \det(A) \]

  + the eigenvalues via the quadratic formula

    \[ \lambda = \frac{\operatorname{tr}(A) \pm \sqrt{\operatorname{tr}^2(A) - 4\det(A)}}{2} \]

  + $\operatorname{gap}(A) = \sqrt{\operatorname{tr}(A) - 4 \det(A)}$: the distance btw the two eigenvalues
  
    \[ \frac{\partial \lambda}{\partial a} = \frac12 \left(1 \pm \frac{a-d}{\operatorname{gap}(A)}\right), \hspace{2em} \frac{\partial \lambda}{\partial b} = \frac{\pm c}{\operatorname{gap}(A)} \]

  + similar formula for $c$ and $d$
  + isolated eigenvalues $\implies$ well-conditioned calculation
  + finding eigenvectors by the Cayley-Hamilton theorem
    + eigenvalues $\lambda_1, \lambda_2 \implies (A - \lambda_1 I)(A - \lambda_2 I) = (A -\lambda_2 I){A - \lambda_1 I) = 0$ 
    + the columns of $(A - \lambda_2 I)$ annihilated by $(A - \lambda_1 I)$ and vice versa
    + neither matrix zero $\implies$ the columns of each must include eigenvectors for the other eigenvalue

+ Example: $2 \times 2$ matrix
  + consider $A_{2 \times 2}$

    \[ A = \begin{bmatrix} 4&3\\-2&-3 \end{bmatrix} \]

  + $\operatorname{tr}(A) = 4- 3 = 1, \det(A) = 4(-3) -3(-2) = -6, \therefore\;$ the characteristic equation

    \[ 0 = \lambda^2 - \lambda -6 = (\lambda - 3)(\lambda + 2) \]

  + the eigenvalues: 3 and -2

    \[ A-3I = \begin{bmatrix} 1&3\\-2&-3 \end{bmatrix}, \hspace{2em} A + 2I = \begin{bmatrix} 6&3\\-2&-1 \end{bmatrix} \]

  + the eigenvectors: (1, -2) for $\lambda = -2$ and (3, -1) for $\lambda = 3$

+ 3-dim matrix
  + characteristic equation of a symmetric $A_{3 \times 3}$

    \[ \det(\alpha I - A) = \alpha^3 - \alpha^2 \operatorname{tr}(A) -\alpha \frac12 \left(\operatorname{tr}(A^2) - \operatorname{tr}^2(A)\right) - \det(A) = 0 \]

  + solved by the methods of Cardano or Lagrange
    + simplify the expression considerably
    + directly lead to a trigonometric solution
  + $A = pB + qI \implies A, B$ w/ the same eigenvectors
  + $\beta$ as an eigenvalue of $B \iff \alpha = p\beta + q$ an eigenvalue of $A$
  + let $q = \operatorname{tr}(A)/3$ and $p = \left( \operatorname{tr}\left((A - qI)^2\right)/6\right)^{1/2}$

    \[ \det(\beta I - B) = \beta^3 - 3 \beta -\det(B) = 0 \]

  + substitution $\beta = 2 \cos\theta$ and simplifying the identity $\cos 3\theta = 4 \cos^3 \theta - 3 \cos \theta \to \cos 3\theta = \det(B)/2$

    \[ \beta = 2 \cos\left(\frac13 \arccos\left(\det(B)/2\right) + \frac{2k\pi}{3} \right), \;k = 0, 1, 2 \]

  + $|\det(\beta)} > 2 \in \Bbb{C} \implies \arccos$ taken along the same branch for all three values of $k$
  + pseudo code

    ```matlab
    % Given a real symmetric 3x3 matrix A, compute the eigenvalues
    % Note that acos and cos operate on angles in radians

    p1 = A(1,2)^2 + A(1,3)^2 + A(2,3)^2
    if (p1 == 0) 
      % A is diagonal.
      eig1 = A(1,1)
      eig2 = A(2,2)
      eig3 = A(3,3)
    else
      q = trace(A)/3               % trace(A) is the sum of all diagonal values
      p2 = (A(1,1) - q)^2 + (A(2,2) - q)^2 + (A(3,3) - q)^2 + 2 * p1
      p = sqrt(p2 / 6)
      B = (1 / p) * (A - q * I)    % I is the identity matrix
      r = det(B) / 2

      % In exact arithmetic for a symmetric matrix  -1 <= r <= 1
      % but computation error can leave it slightly outside this range.
      if (r <= -1) 
          phi = pi / 3
      elseif (r >= 1)
          phi = 0
      else
          phi = acos(r) / 3
      end

      % the eigenvalues satisfy eig3 <= eig2 <= eig1
      eig1 = q + 2 * p * cos(phi)
      eig3 = q + 2 * p * cos(phi + (2*pi/3))
      eig2 = 3 * q - eig1 - eig3     % since trace(A) = eig1 + eig2 + eig3
    end
    ```

  + via Cayley-Hamilton theorem
    + $\alpha_1, \alpha_2, \alpha_3$ distinct eigenvalues of $A \implies (A -\alpha_1 I)(A - \alpha_2 I)(A - \alpha_3 I) = 0$
    + the product of any two of these matrices containing an eigenvector for the 3rd eigenvalue
  + $\alpha_3 = \alpha_1 \implies (A - \alpha_1 I)^2(A - \alpha_2 I) = 0 \text{ and } (A - \alpha_2 I)(A - \alpha_1 I)^2 = 0$
    + generalized eigenspace of $\alpha_1$ spanned by the columns of $A - \alpha_2 I$
    + ordinary eigenspace of $\alpha_1$ spanned by the columns of $(A - \alpha_1 I)(A - \alpha_2 I)$
    + ordinary eigenspace of $\alpha_2$ spanned by the columns of $(A - \alpha_1 I)^2$

+ Example: 3x3 matrix
  + consider a matrix

    \[ A = \begin{bmatrix} 3&2&6\\2&2&5\\-2&-1&-4 \end{bmatrix} \]

  + the characteristic equation: $0 = \lambda^2 - \lambda^2 - \lambda + 1 = (\lambda -1)^2(\lambda + 1)$
  + eigenvalues: 1 (of multiplicity 2) and -1
  + eigenvectors

    \[\begin{align*}
      A - I = \begin{bmatrix} 2&2&6\\2&1&5\\-2&-1&-5 \end{bmatrix}, &\hspace{2em} A + I = \begin{bmatrix} 4&2&6\\2&3&5\\-2&-1&-3 \end{bmatrix} \\\\
      (A - I)^2 = \begin{bmatrix} -4&0&-8\\-5&0&-8\\4&0&8 \end{bmatrix}, &\hspace{2em} (A - I)(A + I) = \begin{bmatrix} 0&4&4\\0&2&2\\0&-2&-2 \end{bmatrix}
    \end{align*}\]

    + $\lambda = -1 \to (-4, -4, 4)$
    + $\lambda = 1 \to  (4, 2, -2)$
    + generalized eigenvectors w/ $\lambda = 1$: (2, 3, -1) and (6, 5, -3)
    + (2, 3, -1) and (6, 5, -3) combined w/ (-4, -4, 4) and (4, 2, -2) to form a basis of generalized eigenvectors of $A$

+ Eigenvectors of normal 3x3 matrices
  + a normal matrix $A_{3 \times 3} \implies$ the cross-product used to find eigenvectors
  + eigenvalue $\lambda \implies$ the null space of $(A - \lambda I)$ perpendicular to its column space
  + cross-product of two independent columns of $A - \lambda I \in$ null space
  + the column space is 2 dimensional $\to$ eigenspace must be 1 dimensional $\to$ anu other eigenvector parallel to it


## List of eigenvalue algorithms

+ [Eigenvalue algorithm](https://en.wikipedia.org/wiki/List_of_numerical_analysis_topics#Eigenvalue_algorithms) — a numerical algorithm for  locating the eigenvalues of a matrix
  + [Power iteration](https://en.wikipedia.org/wiki/Power_iteration)
  + [Inverse iteration](https://en.wikipedia.org/wiki/Inverse_iteration)
  + [Rayleigh quotient iteration](https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration)
  + [Arnoldi iteration](https://en.wikipedia.org/wiki/Arnoldi_iteration) — based on Krylov subspaces
  + [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) — Arnoldi, specialized for positive-definite matrices
    + [Block Lanczos algorithm](https://en.wikipedia.org/wiki/Block_Lanczos_algorithm) — for when matrix is over a finite field
  + [QR algorithm](https://en.wikipedia.org/wiki/QR_algorithm)
  + [Jacobi eigenvalue algorithm](https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm) — select a small submatrix which can be diagonalized exactly, and repeat
    + [Jacobi rotation](https://en.wikipedia.org/wiki/Jacobi_rotation) — the building block, almost a Givens rotation
    + [Jacobi method for complex Hermitian matrices](https://en.wikipedia.org/wiki/Jacobi_method_for_complex_Hermitian_matrices)
  + [Divide-and-conquer eigenvalue algorithm](https://en.wikipedia.org/wiki/Divide-and-conquer_eigenvalue_algorithm)
  + [Folded spectrum method](https://en.wikipedia.org/wiki/Folded_spectrum_method)
  + [LOBPCG](https://en.wikipedia.org/wiki/LOBPCG) — Locally Optimal Block Preconditioned Conjugate Gradient Method
  + [Eigenvalue perturbation](https://en.wikipedia.org/wiki/Eigenvalue_perturbation) — stability of eigenvalues under perturbations of the matrix
