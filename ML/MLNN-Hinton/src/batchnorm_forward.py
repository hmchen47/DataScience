def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_std Array of shape (D,) giving running std of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-7)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_std = bn_param.get('running_std', np.zeros(D, dtype=x.dtype))

  if mode == 'train':
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    out = gamma * (x - m) / s + beta
    running_mean = momentum * running_mean + (1 - momentum) * m
    running_std = momentum * running_std + (1 - momentum) * s
  elif mode == 'test':
    m = running_mean
    s = running_std
    out = gamma * (x - m) / s + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  bn_param['running_mean'] = running_mean
  bn_param['running_std'] = running_std

  cache = (x, gamma, beta, m, s)
  return out, cache

def batchnorm_backward(dout, cache):
  """Backward pass for batch normalization.

  For this implementation, you should write out a computation graph
  for batch normalization on paper and propagate gradients backward
  through intermediate nodes.

  Let's skip computation graph and go straight to full derivative.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  N, D = dout.shape
  x, gamma, beta, m, s = cache

  xm = x - m
  dx = gamma / s * dout \
     - gamma / (N * s) * np.sum(dout, axis=0) \
     - gamma / (N * s**3) * xm * np.sum(dout * xm, axis=0)

  dgamma = np.sum(dout * xm / s, axis=0)
  dbeta = np.sum(dout, axis=0)

  return dx, dgamma, dbeta
