# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import argparse, torch.optim as optim 

import cv2, torch, json
import numpy as np

import os
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists

from net import SiamRPNotb
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from run_SiamFPN import SiamFPN_init, SiamFPN_track
from utils import rect_2_cxy_wh, cxy_wh_2_rect
from config import *
from eval_otb import *

from fpn import *
REWRITE = False

parser = argparse.ArgumentParser(description='PyTorch SiamRPN OTB Test')
parser.add_argument('--dataset', dest='dataset', default='OTB2015', help='datasets')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')

def train():
    return None
    
def track_video(model, video):
    video_path = join('test', args.dataset, 'SiamRPN_AlexNet_OTB2015')
    result_path = join(video_path, '{:s}.txt'.format(video['name']))
    if os.path.exists(result_path) and not REWRITE:
        return 1 # 跟踪过了的，就1秒咯~
    toc, regions = 0, []
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)  # TODO: batch load
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            # state = SiamRPN_init(im, target_pos, target_sz, model)  # init tracker
            state = SiamFPN_init(im, target_pos, target_sz, model)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(gt[f])
        elif f > 0:  # tracking
            # state = SiamRPN_track(state, im)  # track
            state = SiamFPN_track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
            regions.append(location)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= 0:  # visualization
            if f == 0: cv2.destroyAllWindows()
            if len(gt[f]) == 8:
                cv2.polylines(im, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            else:
                cv2.rectangle(im, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]  #
                cv2.rectangle(im, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(video['name'], im)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    video_path = join('test', args.dataset, 'SiamRPN_AlexNet_OTB2015')
    if not isdir(video_path): makedirs(video_path)
    result_path = join(video_path, '{:s}.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write(','.join([str(i) for i in x])+'\n')

    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f / toc))
    return f / toc

def load_dataset(dataset):
    '''[加载训练数据集，暂时以OTB100来训练]
    
    Arguments:
        dataset {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    '''

    base_path = r'D:\workspace\vot\tracker_benchmark\OTB'
    json_path = r'D:\workspace\vot\tracker_benchmark\OTB\OTB2015.json'
    
    info = json.load(open(json_path, 'r'))
    for v in info.keys():
        path_name = info[v]['name']
        info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
        info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]  # our tracker is 0-index
        info[v]['name'] = v
    return info

def main():
    
    global args, v_id
    args = parser.parse_args()
    epochs = 1  # 训练的轮次
    fps_list = []
    # args.visualization = True  # 开启视觉展示

    net = SiamFPN50().cuda()
    
    dataset = load_dataset(args.dataset)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   # 学习率为0.001
    criterion = nn.CrossEntropyLoss()   # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
        
    for epochi in range(epochs): # 每个epoch要训练所有数据集所有的图片，每训练一个数据集便打印一下训练的效果（loss值）
        running_loss = 0.0
        for v_id, video in enumerate(dataset.keys()):            
            # 此处用于测试，仅训练单个视频
            if v_id==1:
                break
            video = 'Biker'.lower()
            
            # 执行跟踪任务
            fps = track_video(net, dataset[video])

            # 加载跟踪后的结果
            video_path = join('test', args.dataset, 'SiamRPN_AlexNet_OTB2015')
            result_path = join(video_path, '{0}.txt'.format(video))                    
            bb_rect = np.array(np.loadtxt(result_path, delimiter=',').astype(np.float))

            # 加载ground_truth
            gt_rect = np.array(dataset[video]['gt_rect']).astype(np.float)

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            loss = criterion(bb_rect, gt_rect)  # 计算损失值
            loss.backward()                     # loss反向传播
            optimizer.step()                    # 反向传播后参数更新

            print('[%d, %5d] loss: %.3f' % (epoch + 1, v_id + 1, running_loss / gt_rect.shape[0]))

            print("finish training sequence %s" % video)
            fps_list.append(fps)

            # 引入评测函数,输出精度和成功率
            # eval_auc_single(dataset='OTB2015', tracker_reg='S*', start=0, end=1e6,req_name=video)        
            print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))
    torch.save(net.state_dict(),"siamfpn.pkl")

if __name__ == '__main__':
    main()
