class Function():


  def __init__(self):
    self.grad_fn = [] # the Functions (or Tensors) that created this Function


  def __call__(self, **kwargs):
    """

    * Run operation and return resulting Tensor
    * Maintain operation's gradient function (grad_fn) in DAG
    """
    pass

  def backward(self, voi, **kwargs):
    """

    * Compute gradient for each gradient function (grad_fn)
    * Accumulates gradients in Tensor grad attributes
    * Propagates all the way to Tensors (leafs of DAG)

    Args:
      voi: variable of interest.
    """
    pass

