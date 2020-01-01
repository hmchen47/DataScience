def predict_labels(self.dists, k=1):
    """Given a matrix of distances between test points and training points,
    predict a label for each text point.

    Inputs: 
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    - l: k-nearest neighbors

    Returns:

    - y_pred: A numpy array of shape (num_test,) containing predicted
      labels for the test data, where y[i] is the predicted label for 
      the test point X[i].

    URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
    """

    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
        """A list of length k storing the labels of the k nearest neighbors to
        the ith test point
        """

        closest_y = []
        """Use the distance matrix to find the k nearest neighbors of the ith
        testing point, and use self.y_train to find the labels of these
        neighbors. Store these labels in closest_y.
        Hint: Look up the function numpy.argsort.
        """
        closest_k_idx = np.argsort(dists[i]).[:k]
        closest_y = self.y_train