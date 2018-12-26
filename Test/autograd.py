import torch
from torch.autograd import Variable

# 创建一个变量：
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

# 对变量做一个运算:
y = x + 2
print(y)

# y作为一个运算的结果被创建，所以它有grad_fn。
print(y.grad_fn) # grad_fn=<AddBackward>

# 在y上做更多的运算：
z = y * y * 3
out = z.mean()

print(z, out) # z: grad_fn=<MulBackward>  out:grad_fn=<MeanBackward1>


## 梯度
# out.backward()等价于out.backward(torch.Tensor([1.0]))。
out.backward()
# 打印梯度 d(out)/dx
print(x.grad)

# 使用autograd做很多疯狂的事情
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
