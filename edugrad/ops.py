from edugrad.function import Function
from edugrad.tensor import Tensor

class Add(Function):


  def __init__(self):
    super().__init__()


  def __call__(self, a: Function | Tensor, b: Function | Tensor):
    self.grad_fn = [a, b]
    return Tensor(a() + b(), grad_fn=self)
    
    
  def backward(self, voi):
    for t in self.grad_fn: 
      t.backward(voi)


class Mul(Function):


  def __init__(self):
    super().__init__()


  def __call__(self, a: Function | Tensor, b: Function | Tensor):
    self.grad_fn = [a, b]
    return Tensor(a() * b(), grad_fn=self)
    
    
  def backward(self, voi):
    self.grad_fn[0]
    backward(voi)

