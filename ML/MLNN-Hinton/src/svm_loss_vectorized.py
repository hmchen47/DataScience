def svm_loss_vectorized(W, X, y, reg):
    """
    URL: https://trongr.github.io/neural-network-course/neuralnetworks.html

    \[
        \frac{\partial L}{\partial W} = \frac{X^T}{N} 
        \begin{pmatrix}
            X & \leftarrow & X(S - S[y] + 1 > 0) \\
            X[y] & \leftarrow & 0 \\
            X[y] & \leftarrow & -\sum_{cols} X
        \end{pmatrix}
        + \lambda W \thicksim D \times C
    \]
    """

    num_train = X.shape[0]
    allrows = np.arange(num_train)
    S = np.dot(X, W)
    Z = S - S[allrows, y][None].T + 1

    margins = np.maximum(0, Z)
    margins[allrows, y] = 0

    loss = 1.0 / num_train * np.sum(margins) + 0.5 * reg * np.sum(W * W)

    margin_indicators = np.where(Z > 0, 1, 0)
    margin_indicators[allrows, y] = 0
    margin_indicators[allrows, y] = -np.sum(margin_indicators, axis=1)

    dW = 1.0 / num_train * np.dot(X.T, margin_indicators) + reg * W

    return loss, dW

