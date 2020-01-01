def conv_backward_naive(dout, cache):
  """
  URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
  
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  (x, w, b, conv_param, c) = cache
  s = conv_param["stride"]
  p = conv_param["pad"]
  N, C, H, W = x.shape
  F, C, hf, wf = w.shape
  Hp = 1 + (H + 2 * p - hf) / s
  Wp = 1 + (W + 2 * p - wf) / s

  dy = dout.reshape(N, -1).T.reshape(F, -1)
  f = w.reshape(F, -1)
  dx = np.dot(f.T, dy)
  dx = ic.col2im_indices(dx, x.shape, hf, wf, p, s)
  dw = np.dot(dy, c.T).reshape(w.shape)
  db = np.sum(dy, axis=1)

  return dx, dw, db
