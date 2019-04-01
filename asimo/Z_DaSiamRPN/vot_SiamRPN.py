# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

'''
1.将此作为baseline，改进点记录为 # selonsy，可全局搜索修改
2.可尝试将代码进行优化，以提高运行速度，即FPS

'''

import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net_file = 'D:\\workspace\\vot\\asimo\\other\\DaSiamRPN\\selonsy\\model\\SiamRPNBIG.model' #join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file)) # 加载模型参数
net.eval().cuda() # 对模型进行验证，cuda(device=None) 将所有模型参数和缓冲区移动到GPU

# warm up
# 预热，运行模板和检测分支10次，先屏蔽掉
# for i in range(10):
#     net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda()) # selonsy:cuda()表示使用GPU进行计算,FloatTensor(1, 3, 127, 127)：浮点型四维张量
#     net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
handle = vot.VOT("polygon")
Polygon = handle.region() # region：将配置消息发送到客户端并接收初始化区域和第一个图像的路径。其返回值为初始化区域。
cx, cy, w, h = get_axis_aligned_bbox(Polygon) # get_axis_aligned_bbox：将坐标数据转换成 RPN 的格式

image_file = handle.frame() # frame 函数从客户端获取帧（图像路径）
if not image_file:
    sys.exit(0)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_file)  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker,SiamRPN_init:构造状态结构体并运行模板分支
# 从第一帧开始跟踪，表示很奇怪，难道直接给定的不准确么？   # selonsy：改进点
while True: # 进入跟踪循环
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    state = SiamRPN_track(state, im)  # track,SiamRPN_track:运行检测分支并更新状态变量
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz']) # cxy_wh_2_rect:将坐标转换成矩形框的表示形式

    handle.report(Rectangle(res[0], res[1], res[2], res[3])) # report:将跟踪结果报告给客户端

print(handle.result)
print(handle.frames)

del handle