def rmsprop(x, dx, config=None):
  """
  URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
  
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  a = config.setdefault('learning_rate', 1e-2)
  d = config.setdefault('decay_rate', 0.99)
  e = config.setdefault('epsilon', 1e-8)
  c = config.setdefault('cache', np.zeros_like(x))

  c = d * c + (1 - d) * dx ** 2
  next_x = x - a * dx / (np.sqrt(c) + e)

  config["cache"] = c
  return next_x, config
