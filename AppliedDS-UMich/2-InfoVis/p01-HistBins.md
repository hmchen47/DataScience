# Selecting the Number of Bins in a Histogram: A Decision Theoretic Approach

Author: K. He & G.Meeden
Publication: Journal of Statistical Planning and inference, 61(1), 49-59
Year: 1997

## Introduction

+ D. Scott, "On optimal and data-based histograms," Biometrika, 66:605{610, 1979.
    + A formula for the optimal histogram bin width which asymptotically minimizes the integrated mean squared error
    + Since the underlying density is usually unknown, it is not immediately clear how one should apply this in practice.
    + Using the Gaussian density as a reference standard, which leads to the data-based choice for the bin width of $a \times s \times n^{-1/3}$, where $a = 3.49$ and $s$ is an estimate of the standard deviation.
    + For real data sets histograms based on __5-20 bins__ usually suffice.

+ Mats Rudemo, "Empirical choice of histograms and kernel density estimators", Scandinavian J. of Stat., 9:65-78, 1982
    + a cross-validation technique for selecting the number of bins
    + large sampling variation


## Selecting the Number of Bins

+ Problem: decision on a histogram is an estimate of the unknown density $f$ where the bins are all of the same size, a typical decision $d$ consists of two components, the number of bins and the mass assigned to each bin.

+ Notation
    + ${\bf X} = (X_1, \ldots, X_n)$: a random sample from some unknown continuous distribution on the known interval $[a, b]$
    + $\Theta$: parameter space, the class of all probability density functions $f$ on $[a, b]$
    + $d = (k, {\bf m}_k)$: decision function
        + $k \in \mathbb{N}^+$ and $5 \leq k \leq 20$ in general
        + ${\bf m}_k = (m_{k,1}, \ldots, m_{k,k})$: measured vector, ${\bf m}_k > 0$ and $\sum_i m_{k, i} = 1$
    + $L(f, d, {\bf m})$: a loss function when $d = (k, {\bf m}_k)$ is an estimate of $f$
    + ${\bf p}_k (f) = {\bf p}_k = (p_{k,1}, \ldots, p_{k,k})$: predict vector

+ Loss function $L(f, d, {\bf x})$
    + Assume that the loss function depends on $f$ only through the mass it assigns to the $k$ equal subintervals of $[a, b]$, i.e., on ${\bf p}_k (f) = {\bf p}_k = (p_{k,1}, \ldots, p_{k,k})$
    + Assume that the loss function $L(f, d, {\bf x})$ depends on the data ${\bf X} = {\bf x}$ and denote by

        $$\begin{array}{rcl} 
            L(f, d, {\bf m}) & = & L({\bf p}_k, k, {\bf m}_k, {\bf x}) \\
                & = & 
                \left\{
                \begin{array}{ll} 
                    c(k, {\bf m}) \sum_1^k (p_{k,i} - m_{k,i})^2 & \text{if no  } m_{k,i}= 1 \\ 
                    1/k & \text{if  } \exists \text{  } m_{k,i} = 1
                \end{array}
                \right.
        \end{array}$$
    + Consider the second part $1/k$: With simple case of $4$ bins on the unit interval $[0, 1]$ and the unknown $f$ puts all its mass in the first  bin, i.e., $[0, .25]$. Consider two different decisions $d_1 = (2, (1, 0))$ and $d_2 = (4, (1, 0, 0, 0 ))$. Both decisions on the first part equals to $0$.  But $d_1$ is bettern than $d_2$.


## The stepwise Bayes justification




