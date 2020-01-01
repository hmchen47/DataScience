def adam(x, dx, config=None):
  """
  URL: https://trongr.github.io/neural-network-course/neuralnetworks.html

  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  a = config.setdefault('learning_rate', 1e-3)
  b1 = config.setdefault('beta1', 0.9)
  b2 = config.setdefault('beta2', 0.999)
  e = config.setdefault('epsilon', 1e-8)
  m = config.setdefault('m', np.zeros_like(x))
  v = config.setdefault('v', np.zeros_like(x))
  t = config.setdefault('t', 0) + 1

  m = b1 * m + (1 - b1) * dx
  v = b2 * v + (1 - b2) * dx ** 2

  mhat = m / (1 - b1 ** t) # NOTE. Bias corrections not retained by m and v,
  vhat = v / (1 - b2 ** t) # only in temporary variables mhat and vhat.

  next_x = x - a * mhat / (np.sqrt(vhat) + e)

  config["m"] = m
  config["v"] = v
  config["t"] = t

  return next_x, config


