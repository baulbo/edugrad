from edugrad.function import Function
class Tensor(Function):


  def __init__(self, value: float, grad_fn: Function = None):
    super().__init__()
    self.grad = 0 # keeps accumulation of gradients
    self.value = value
    self.grad_fn = grad_fn


  def __call__(self):
    return self.value


  def backward(self, voi):
    if voi == self:
      self.grad+=1
    elif self.grad_fn != None:
      self.grad_fn.backward(voi)
