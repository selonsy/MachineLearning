
import torch
import numpy as np

x = torch.Tensor(5, 3)
y = torch.rand(5, 3)

# 加法
print(x + y)

print(torch.add(x, y))

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y,原地进行（in-place）的加法
y.add_(x)
print(y)

# 注：任何原地改变tensor的运算后边会后缀一个“_”,例如：x.copy_(y),x.t_(),会改变x的值。


# 将torch Tensor转换为numpy array
# torch的Tensor和numpy的array分享底层的内存地址，所以改变其中一个就会改变另一个。
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)
    

