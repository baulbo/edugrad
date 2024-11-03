from edugrad.function import function
from edugrad.tensor import Tensor
from edugrad.ops import Add, Mul
import random

def Linear(Function):


  def __init__(self, num_neurons: int):
    super().__init__()
    self.num_neurons = num_neurons
    self.weights = [Tensor(random.random()) for _ in range(num_neurons)]


  def __call__(self, x: list[Tensor]):
    self.grad_fn = [] # list of multiplications


