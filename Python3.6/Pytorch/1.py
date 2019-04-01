import torch
import numpy as np

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
