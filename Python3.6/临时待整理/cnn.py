# 这是一个简单的前馈网络。它接受输入，然后一层一层向前传播，最后输出一个结果。
# 训练神经网络的典型步骤如下：
# （1）  定义神经网络，该网络包含一些可以学习的参数（如权重）
# （2）  在输入数据集上进行迭代
# （3）  使用网络对输入数据进行处理
# （4）  计算loss（输出值距离正确值有多远）
# （5）  将梯度反向传播到网络参数中
# （6）  更新网络的权重，使用简单的更新法则：weight = weight - learning_rate* gradient，即：新的权重=旧的权重-学习率*梯度值。

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        # 1 input image channel, 6 output channels, 5x5 square convolution        
        # 输入图像通道为1，6个输出通道，5×5的卷积核      
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 输入图像通道为6，16个输出通道，5×5的卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        # y = Wx + b 的仿射变换
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    # backward函数（梯度在此函数中计算）就会利用autograd来自动定义
    def forward(self, x):
        # Max pooling over a (2, 2) window
        # 2×2的最大池化窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        # 若是一个正方形，可以用一个数来表示
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension:取得除batch外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# 学习到的参数可以被net.parameters()返回
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight：卷积层1的参数


input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

# 将梯度缓冲区归零，然后使用随机梯度值进行反向传播。
net.zero_grad()
out.backward(torch.randn(1, 10))

# 损失函数
# 损失函数采用输出值和目标值作为输入参数，来计算输出值距离目标值还有多大差距。
# 在nn package中有很多种不同的损失函数，最简单的一个loss就是nn.MSELoss，它计算输出值和目标值之间的均方差。

output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
target = target.float().view(1,-1)  # 先统一数据格式，再统一维度，-1表示由前面的进行推断
criterion = nn.MSELoss()

loss = criterion(output, target.float())
print(loss)

# 没有creator属性，不知道为什么？
# print(loss.creator) # MSELoss
# print(loss.creator.previous_functions[0][0])# Linear
# print(loss.creator.previous_functions[0][0].previous_functions[0][0])# Relu

net.zero_grad() # 清空梯度的缓存
print("反向传播前的conv1.bias.grad：")
print(net.conv1.bias.grad)

loss.backward()

print("反向传播后的conv1.bias.grad：")
print(net.conv1.bias.grad)

# 更新权重
# Stochastic Gradient Descent (SGD): 
# weight=weight−learningrate∗gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
# 当然，PyTorch同样提供了很多的类似函数包括SGD，Nesterov-SGD, Adam, RMSProp等等。所有的这些方法都被封装到包torch.optim中。

import torch.optim as optim
# 创建自己的optimizer
optimizer = optim.SGD(net.parameters(),lr=0.01)
# 在训练的循环中
optimizer.zero_grad() # 清空梯度缓存
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step() # 更新操作


