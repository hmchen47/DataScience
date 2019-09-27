def relu_backward(dout, cache):
  """
  URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
  
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = dout * np.where(x >= 0, 1, 0)
  return dx
