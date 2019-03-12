import torch
import numpy as np
from torch.autograd import Variable

#######################################################################################
# 1.
# 2.
# 3.
########################################################################################
#True支持GPU，False不支持
print(torch.cuda.is_available())

# 生成一个五行四列的二维矩阵，未进行初始化
print(torch.Tensor(5, 4))

# 返回的数组大小是5x4的矩阵，初始化是0~1的均匀分布
x=torch.rand(5, 4)
print(x)

# torch.randn(*sizes, out=None) → Tensor
# 返回一个张量，包含了从标准正态分布(均值为0，方差为1，即高斯白噪声)中抽取一组随机数，形状由可变参数sizes定义。
print(torch.randn(5, 4))

#查看x的形状,是一个tuple
print(x.size())

# numpy 类似的返回5x4大小的矩阵
print(np.ones((5, 4)))

# 类似的返回5x4大小的张量
print(torch.ones(5,4))

# 返回5x4大小的张量 对角线上全1，其他全0
print(torch.eye(5,4))

# 返回一个1维张量，包含从start到end，以step为步长的一组序列值(默认步长为1)
print(torch.arange(1,5,1))  # result：tensor([1, 2, 3, 4])

# 返回一个1维张量，包含在区间start 和 end 上均匀间隔的steps个点。
print(torch.linspace(1,5,2))

# torch.normal(means, std, out=None)
# 返回一个张量，包含从给定参数means,std的离散正态分布中抽取随机数。
print(torch.normal(means=torch.arange(1, 11), std=torch.arange(1, 0, -0.1)))

# 给定参数n，返回一个从0 到n -1 的随机整数排列。
print(torch.randperm(2))

# numpy转换成Tensor
a = torch.ones(5)
b = a.numpy()
print(b)

# Tensor转换成numpy
a= np.ones(5)
b=torch.from_numpy(a)
print(b)


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



########################################################################################
# 1.autograd
# 2.
# 3.
########################################################################################

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
