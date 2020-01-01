def affine_backward(dout, cache):
  """
  URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
  
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: bias weights, of shape (M,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
  db = np.sum(dout, axis=0)
  dx = np.dot(dout, w.T).reshape(x.shape)
  return dx, dw, db
