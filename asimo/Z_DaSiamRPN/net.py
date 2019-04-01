# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from fpn import *

class SiamRPN(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        feat_in = configs[-1]
        super(SiamRPN, self).__init__()
        self.featureExtract = nn.Sequential(                                # torch.nn.Sequential是顺序容器
            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),   # 第一层卷积，输入图像通道为3，192个输出通道，11×11的卷积核，步长为2
            nn.BatchNorm2d(configs[1]),                                     # 第一层归一化，num_features为192
            nn.MaxPool2d(kernel_size=3, stride=2),                          # 第一层池化，最大池化，采样核size=3，步长为2
            nn.ReLU(inplace=True),          
                                            # 非线性激活函数为ReLu，inplace=True：将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )

        self.anchor = anchor # 锚
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)      # 模板分支的回归子分支
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)               # 检测分支的回归子分支
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)    # 模板分支的分类子分支
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)             # 检测分支的分类子分支
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)          # 1x1卷积

        self.r1_kernel = []
        self.cls1_kernel = []

        self.cfg = {}

    # torch.nn.Module.forward 定义每次调用时执行的计算。所有子类应该覆盖该函数。
    # 即使用xx = net(input) 的时候，就会自动的进入到forward函数里面，其中net是网络的实例。
    def forward(self, x): # x:1,3,271,271
        x_f = self.featureExtract(x) # 1,256,24,24

        # fpn output
        # 1,3,127,127
        # torch.Size([1, 256, 32, 32])
        # torch.Size([1, 256, 16, 16])
        # torch.Size([1, 256, 8, 8])
        # torch.Size([1, 256, 4, 4])
        # 1,3,271,271
        # torch.Size([1, 256, 68, 68])
        # torch.Size([1, 256, 34, 34])
        # torch.Size([1, 256, 17, 17])
        # torch.Size([1, 256, 9, 9])


        # 模板的两个卷积结果：
        # self.r1_kernel：20,256,4,4
        # self.cls1_kernel:10,256,4,4

        fpn_net = FPN101().cuda()
        fms = fpn_net(x)
        # torch.Size([1, 256, 68, 68])
        # torch.Size([1, 256, 34, 34])
        # torch.Size([1, 256, 17, 17])
        # torch.Size([1, 256, 9, 9])
        for fm in fms:
            print(fm.size())
            r1 = self.regress_adjust(F.conv2d(self.conv_r2(fm), self.r1_kernel))
            c1 = F.conv2d(self.conv_cls2(fm), self.cls1_kernel)
            print(r1.size())
            print(c1.size())
            # torch.Size([1, 20, 63, 63])
            # torch.Size([1, 10, 63, 63])
            # torch.Size([1, 20, 29, 29])
            # torch.Size([1, 10, 29, 29])
            # torch.Size([1, 20, 12, 12])
            # torch.Size([1, 10, 12, 12])
            # torch.Size([1, 20, 4, 4])
            # torch.Size([1, 10, 4, 4])

        # Todo：想了一下，fpn要加的话应该加在这里，上面提取的特征图x_f是最顶层的，忽略了底层的细节，对小目标的检测不友好。
        # 在这里传入FPN计算后的特征图s，然后分别于模板的结果卷积，得到几组不同的坐标回归量和得分量，后面再进行筛选。


        # 此处返回检测分支的输入（271*271*3），在经过特征提取网络之后，与模板分支的两个输出（r1_kernel：回归核，cls1_kernel：分类核）直接进行卷积。
        # 此处的F.conv2d为torch.nn.functional下的卷积函数，两个参数分别为 输入张量，过滤器 ，可以理解为自定义卷积操作。
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
               F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def temple(self, z): # 1,3,127,127
        z_f = self.featureExtract(z)            # 调用featureExtract提取模板z的特征,1,256,6,6
        r1_kernel_raw = self.conv_r1(z_f)       # 由z_f得到坐标回归量r1_kernel_raw,1,5120,4,4
        cls1_kernel_raw = self.conv_cls1(z_f)   # 由z_f得到分类得分cls1_kernel_raw,1,2560,4,4
        kernel_size = r1_kernel_raw.data.size()[-1] # 模板宽度
        # torch.Tensor.view 返回一个新的张量，其数据与自身张量相同，但大小不同
        # r1_kernel_raw原始的形状：[feature_out∗4∗anchor,kernel_size,kernel_size]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)   # 20,256,4,4
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size) # 10,256,4,4


class SiamRPNBIG(SiamRPN):
    def __init__(self):
        super(SiamRPNBIG, self).__init__(size=2)
        self.cfg = {'lr':0.295, 'window_influence': 0.42, 'penalty_k': 0.055, 'instance_size': 271, 'adaptive': True} # 0.383


class SiamRPNvot(SiamRPN):
    def __init__(self):
        super(SiamRPNvot, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr':0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 271, 'adaptive': False} # 0.355


class SiamRPNotb(SiamRPN):
    def __init__(self):
        super(SiamRPNotb, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'instance_size': 271, 'adaptive': False} # 0.655

