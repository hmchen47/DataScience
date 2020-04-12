class FullyConnectedNet(object):
  """A fully-connected neural network with an arbitrary number of hidden
  layers, ReLU nonlinearities, and a softmax loss function. This will
  also implement dropout and batch normalization as options. For a
  network with L layers, the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...}
  block is repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in
  the self.params dictionary and will be learned using the Solver
  class.

  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=1, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    predim = input_dim
    for i, dim in enumerate(hidden_dims):
        self.params["W" + str(i + 1)] = weight_scale * np.random.randn(predim, dim)
        self.params["b" + str(i + 1)] = np.zeros(dim)
        if self.use_batchnorm:
            self.params["gamma" + str(i + 1)] = np.ones(dim)
            self.params["beta" + str(i + 1)] = np.zeros(dim)
        predim = dim
    self.params["W" + str(self.num_layers)] = weight_scale * np.random.randn(predim, num_classes)
    self.params["b" + str(self.num_layers)] = np.zeros(num_classes)

    # When using dropout we need to pass a dropout_param dictionary to
    # each dropout layer so that the layer knows the dropout
    # probability and the mode (train / test). You can pass the same
    # dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means
    # and variances, so we need to pass a special bn_param object to
    # each batch normalization layer. You should pass
    # self.bn_params[0] to the forward pass of the first batch
    # normalization layer, self.bn_params[1] to the forward pass of
    # the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since
    # they behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    out = X
    cache = {}
    dropout_cache = {}
    for i in xrange(self.num_layers - 1):
        W = self.params["W" + str(i + 1)]
        b = self.params["b" + str(i + 1)]
        if self.use_batchnorm:
            gamma = self.params["gamma" + str(i + 1)]
            beta = self.params["beta" + str(i + 1)]
            bn_param = self.bn_params[i]
            out, cache[str(i + 1)] = affine_batchnorm_relu_forward(out, W, b, gamma, beta, bn_param)
        else:
            out, cache[str(i + 1)] = affine_relu_forward(out, W, b)

        if self.use_dropout:
            out, dropout_cache[str(i + 1)] = dropout_forward(out, self.dropout_param)

    W = self.params["W" + str(self.num_layers)]
    b = self.params["b" + str(self.num_layers)]
    scores, cache[str(self.num_layers)] = affine_forward(out, W, b)

    if mode == 'test':
      return scores

    grads = {}
    loss, dout = softmax_loss(scores, y)
    dout, dW, db = affine_backward(dout, cache[str(self.num_layers)])
    grads["W" + str(self.num_layers)] = dW
    grads["b" + str(self.num_layers)] = db

    for i in reversed(xrange(self.num_layers - 1)):
        if self.use_dropout:
            dout = dropout_backward(dout, dropout_cache[str(i + 1)])

        if self.use_batchnorm:
            dout, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache[str(i + 1)])
            grads["gamma" + str(i + 1)] = dgamma
            grads["beta" + str(i + 1)] = dbeta
        else:
            dout, dW, db = affine_relu_backward(dout, cache[str(i + 1)])

        W = self.params["W" + str(i + 1)]
        grads["W" + str(i + 1)] = dW + self.reg * W
        grads["b" + str(i + 1)] = db
        loss += 0.5 * self.reg * np.sum(W * W)

    return loss, grads