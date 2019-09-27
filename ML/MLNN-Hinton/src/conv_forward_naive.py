def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height h and width h.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, hf, wf)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - hf) / stride
    W' = 1 + (W + 2 * pad - wf) / stride
  - cache: (x, w, b, conv_param)
  """
  s = conv_param["stride"]
  p = conv_param["pad"]
  N, C, H, W = x.shape
  F, C, hf, wf = w.shape
  Hp = 1 + (H + 2 * p - hf) / s
  Wp = 1 + (W + 2 * p - wf) / s

  f = w.reshape(F, -1)
  c = ic.im2col_indices(x, hf, wf, p, s)
  out = np.dot(f, c) + b[None].T
  out = out.reshape(-1, N).T.reshape(N, F, Hp, Wp)

  cache = (x, w, b, conv_param, c)
  return out, cache
