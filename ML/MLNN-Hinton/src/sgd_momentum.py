def sgd_momentum(w, dw, config=None):
  """
  URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
  
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  a = config.setdefault('learning_rate', 1e-2)
  m = config.setdefault('momentum', 0.9)
  v = config.setdefault('velocity', np.zeros_like(w))

  v = m * v - a * dw
  next_w = w + v

  config['velocity'] = v
  return next_w, config
