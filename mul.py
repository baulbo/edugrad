from edugrad.tensor import Tensor
from edugrad.ops import Add

a = Tensor(1)
b = Tensor(2)

add = Add()
c = add(a, b)
add.backward(a)

print("a's gradient (1?):", a.grad)
print()

add2 = Add()
add3 = Add()
add4 = Add()
d = Tensor(2)
e = Tensor(3)

add4(add2(d, e) , add3(d, e))
add4.backward(d)
print("d's gradient (2?):", d.grad)
print()

