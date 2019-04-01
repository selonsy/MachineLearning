# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import vot
from vot import Rectangle,convert_region
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
import glob
from net import SiamRPNBIG
from net import SiamRPNvot
from net import SiamRPNotb
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from config import *

def run_Tracker_2_VOT(argv):
    # load net
    #net_file = 'D:\\workspace\\vot\\asimo\\other\\DaSiamRPN\\selonsy\\model\\SiamRPNBIG.model' #join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
    net_file = model_path
    net = SiamRPNBIG()    
    net.load_state_dict(torch.load(net_file)) # 加载模型参数
    net.eval().cuda()                         # 对模型进行验证，cuda(device=None) 将所有模型参数和缓冲区移动到GPU

    # command = map(str,['python3', 'D:\\workspace\\vot\\asimo\\other\\DaSiamRPN\\code\\my_Tracker.py',        
    #         seq.seqName,
    #         seq.stratFrame,
    #         seq.endFrame,
    #         seq.init_rect,
    #         seq.imgFormat,
    #         seq.name
    #         ])
    # print(argv)  # ['David', '300', '770', '[129, 80, 64, 78]', '{0:04d}.jpg', 'David_0']
    # return None

    # matlab 版本
    # python3 D:\workspace\vot\asimo\other\DaSiamRPN\code\my_Tracker.py 
    # matlab 
    # D:\workspace\vot\tracker_benchmark_python\data\Basketball\img\ 
    # 1 
    # 725 
    # [197,213,34,81] 
    # 4 
    # basketball_1

    if argv[0].strip()=='matlab': 
        bench_mark_type='matlab'       
        seq_list_path = argv[1].strip()
        seq_startFrame = argv[2].strip()
        seq_endFrame = argv[3].strip()
        ground_truth_txt = argv[4].strip()
        seq_imgFormat = argv[5].strip()
        seq_name = argv[6].strip()
    else:
        bench_mark_type='python'  
        data_root_path = dataset_path
        seq_list_path = data_root_path + argv[1].strip()+'\\img'
        seq_startFrame = argv[2].strip()
        seq_endFrame = argv[3].strip()
        seq_imgFormat =  argv[5].strip()
        ground_truth_txt = argv[4].strip()
        seq_name = argv[5].strip()
    
    # start to track
    handle = vot.VOT("polygon",bench_mark_type,seq_list_path,seq_startFrame,seq_endFrame,seq_imgFormat,ground_truth_txt,seq_name)
    Polygon = handle.region() # region：将配置消息发送到客户端并接收初始化区域和第一个图像的路径。其返回值为初始化区域。
    
    # Todo:
    # 暂时不知道为什么要求这么一个框，cx，cy为框的中心点，w，h分别代表宽和高
    # 此框将原始的矩形框进行了一定程度的缩小，具体原因暂不清楚
    
    # Polygon:334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41
    
    cx, cy, w, h = get_axis_aligned_bbox(Polygon) # get_axis_aligned_bbox：将坐标数据转换成 RPN 的格式
    # bag 1.jpg cx:365.2075 cy:194.595 w:106.13089774126823 h:96.41442877353934

    # rect_test = cxy_wh_2_rect(np.array([cx, cy]),np.array([w, h]))
    # rect:'312.1420511293659,146.38778561323034,106.13089774126823,96.41442877353934'

    # rect_test = convert_region(Polygon,'rectangle') #互相转换的值在图示中的矩形框重合    
    # rect:'292.23,128.36,145.95999999999998,132.46999999999997'
    
    image_file = handle.frame() # frame 函数从客户端获取帧（图像路径）
    if not image_file:
        print("Image file not found!")
        sys.exit(0)
    # print(2)

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
        # print("rect:{0},{1},{2},{3}".format(res[0], res[1], res[2], res[3]))
        handle.report(Rectangle(res[0], res[1], res[2], res[3])) # report:将跟踪结果报告给客户端

    # print(handle.result)
    # print(handle.frames)

    del handle
    print("跟踪结束:{0}".format(seq_name))
    return None

if __name__=='__main__':
    run_Tracker_2_VOT(sys.argv[1:])
