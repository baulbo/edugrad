from edugrad.tensor import Tensor
from edugrad.ops import Add

if __name__ == "__main__":

  a = Tensor(1)
  b = Tensor(2)

  add = Add()
  c = add(a, b)
  add.backward(a)

  print("a's gradient (1?):", a.grad)
  print()

  add2 = Add()
  d = Tensor(2)
  e = Tensor(3)

  add2(Add()(d, e) , Add()(d, e))
  add2.backward(d)
  print("d's gradient (2?):", d.grad)
  print()

