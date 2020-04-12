def dropout_forward(x, dropout_param):
  """
  URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
  
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
      NOTE. This is the opposite of the lecture slides! Slides said p is the
      probability of keeping a neuron alive.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  if mode == 'train':
    mask = np.random.random(x.shape) > p
    out = x * mask / (1 - p)
  elif mode == 'test':
    out = x
    mask = None

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)
  return out, cache

def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.
  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  p, mode = dropout_param['p'], dropout_param['mode']
  if mode == 'train':
    dx = dout * mask / (1 - p)
  elif mode == 'test':
    dx = dout
  return dx
