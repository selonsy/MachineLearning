import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

transform = transforms.Compose(
    [
        # ToTensor是指把PIL.Image(RGB) 或者numpy.ndarray(H x W x C) 从0到255的值映射到0到1的范围内，
        transforms.ToTensor(),
        # 并转化成Tensor格式。
        # Normalize(mean，std)是通过下面公式实现数据归一化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # channel=（channel-mean）/std
    ])

'''torchvision.datasets.CIFAR10
1.root，表示cifar10数据的加载的相对目录  !!!!!windows的话直接使用绝对路径即可。
2.train，表示是否加载数据库的训练集，false的时候加载测试集
3.download，表示是否自动下载cifar数据集
4.transform，表示是否需要对数据进行预处理，none为不进行预处理

torch.utils.data.DataLoader
1、dataset，这个就是PyTorch已有的数据读取接口（比如torchvision.datasets.ImageFolder）或者自定义的数据接口的输出，该输出要么是torch.utils.data.Dataset类的对象，要么是继承自torch.utils.data.Dataset类的自定义类的对象。
2、batch_size，根据具体情况设置即可。
3、shuffle，一般在训练数据中会采用。
4、collate_fn，是用来处理不同情况下的输入dataset的封装，一般采用默认即可，除非你自定义的数据读取输出非常少见。
5、batch_sampler，从注释可以看出，其和batch_size、shuffle等参数是互斥的，一般采用默认。
6、sampler，从代码可以看出，其和shuffle是互斥的，一般默认即可。
7、num_workers，从注释可以看出这个参数必须大于等于0，0的话表示数据导入在主进程中进行，其他大于0的数表示通过多个进程来导入数据，可以加快数据导入速度。
8、pin_memory，注释写得很清楚了： pin_memory (bool, optional): If True, the data loader will copy tensors into CUDA pinned memory before returning them. 也就是一个数据拷贝的问题。
9、timeout，是用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错。
'''

cifar_path = 'D:\workspace\DataSet\cifar-10-python' #'D:\99Workspace\MLDL\DataSet\cifar-10'

trainset = torchvision.datasets.CIFAR10(
    root=cifar_path, train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root=cifar_path, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=0)

# print(len(trainset))
# print(len(testset))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 用于显示图像的函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  # 画图命令

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# 交叉熵损失函数和SGD做为优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model_path = 'D:\workspace\\vot\DaSiamRPN\pytorch\model_529.pkl'

# if __name__ == '__main__'，增加这句话，否则多个进程会报错
if __name__ == '__main__':
    # # 随机的获取一些训练图像
    # dataiter = iter(trainloader)
    # images,labels = dataiter.next()
    # #显示图像
    # imshow(torchvision.utils.make_grid(images))
    # #打印标签
    # print(' '.join('%10s'%classes[labels[j]] for j in range(4)))

    i = 2 # i=1:训练,i=2:测试

    if i==1:
        # 训练神经网络
        # 事情变得有趣起来。我们简单的迭代取得数据集中的数据然后喂给神将网络并进行优化。
        for epoch in range(2):  # 多次循环取出数据
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # 获得输入
                inputs, labels = data
                # 使用Variable包装这些数据
                inputs, labels = Variable(inputs), Variable(labels)
                # 清空缓存的梯度信息
                optimizer.zero_grad()
                #forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 打印统计信息
                running_loss += loss.data[0]
                if i % 2000 == 1999:  # 打印每2000 mini-batches
                    print('[%d,%5d] loss :%.3f' %
                        (epoch+1, i+1, running_loss / 2000))
                    running_loss = 0.0

        print("结束训练")
        torch.save(net.state_dict(),model_path)

    # 仅仅保存和加载模型参数
    # torch.save(the_model.state_dict(), PATH)
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))

    # 保存和加载整个模型
    # torch.save(the_model, PATH)
    # the_model = torch.load(PATH)

    # # 在测试集上测试神经网络
    # dataiter = iter(testloader)
    # images,labels = dataiter.next()

    # # 显示图片
    # imshow(torchvision.utils.make_grid(images))
    # print("GroundTruth:",''.join('%10s'%classes[labels[j]] for j in range(4)))

    # outputs = net(Variable(images))

    # _,predicted = torch.max(outputs.data,1)
    # print('Predicted:',' '.join('%5s'%classes[predicted[j][0]] for j in range(4)))

    if i==2:

        # 加载预训练模型
        net.load_state_dict(torch.load(model_path))

        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1) # 输出每行的最大值以及序号
            total += labels.size(0)
            correct += (predicted == labels).sum()
        # 输出总体正确率
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        # 看起来不错哦，因为它比随机的结果好很多（随机的准确率为10%）
        # 那么，那些类别可以被很好的区分，那些类别却又不能被很好的区分呢？
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i].float() / class_total[i]))
