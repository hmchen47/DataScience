def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
    
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
        self.W = 0.001 * np.random.randn(dim, num_classes)

    loss_history = []
    for it in xrange(num_iters):
        batch_idx = np.random.choice(num_train, batch_size, replace=False) # todo try replace=True
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        loss, grad = self.loss(X_batch, y_batch, reg)
        loss_history.append(loss)

        self.W = self.W - learning_rate * grad # gradient descent / delta rule

        if verbose and it % 100 == 0:
            print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history
