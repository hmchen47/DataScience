def svm_loss_naive(W, X, y, reg):
    """
    URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
    """

    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        # calculate partial derivatives:
        margins = scores - correct_class_score + 1
        margins_indicator = np.where(margins > 0, 1, 0)
        dWi = np.dot(X[i][np.newaxis].T, margins_indicator[np.newaxis])
        dWi[:, y[i]] = 0
        dW += dWi

        # replace one column:
        margins_indicator_count = max(0, np.sum(margins_indicator) - 1)
        dW[:, y[i]] += - margins_indicator_count * X[i]

        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # delta = 1
            if margin > 0:
                loss += margin

    loss /= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW
