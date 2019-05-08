import argparse
import functools
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from tqdm import tqdm

import setproctitle

sys.path.append(os.getcwd())

from net.config import *
from net.run_SiamFPN import run_SiamFPN


def embeded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    return int(pieces[1])


def embeded_numbers_results(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    return int(pieces[-2])


def cal_iou(box1, box2):
    r"""

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou


def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)


# 验证跟踪结果的准确性
def evaluation(_type):
    # ------------ starting evaluation  -----------
    if config.MACHINE_TYPE == Machine_type.Linux:
        data_path = '/home/sjl/dataset/otb/'
        result_path = '/home/selonsy/workspace/SiamFPN/data/results/otb_result_otb{0}.json'.format(_type)
        save_path = '/home/selonsy/workspace/SiamFPN/data/results/otb_eval_result_otb{0}.json'.format(_type)
    else:
        data_path = r'D:\dataset\otb\\' # r'E:\dataset\OTB'
        result_path = r"D:\workspace\MachineLearning\HelloWorld\59version\data\results\result_otb{0}.json".format(_type)    
        save_path = r"D:\workspace\MachineLearning\HelloWorld\59version\data\results\eval_result_otb{0}.json".format(_type)
    with open(result_path, 'r') as f:
        results = json.load(f)    
    results_eval = {}
    for model in sorted(list(results.keys()), key=embeded_numbers_results):
        results_eval[model] = {}
        success_all_video = []
        for video in results[model].keys():
            result_boxes = results[model][video]
            with open(data_path + video + '/groundtruth_rect.txt', 'r') as f:
                result_boxes_gt = f.readlines()
            if ',' in result_boxes_gt[0]:
                result_boxes_gt = [list(map(int, box.split(','))) for box in result_boxes_gt]
            else:
                result_boxes_gt = [list(map(int, box.split())) for box in result_boxes_gt]
            result_boxes_gt = [np.array(box) for box in result_boxes_gt]            
            iou = list(map(cal_iou, result_boxes, result_boxes_gt)) # 计算交并比
            success = cal_success(iou)                              # 计算成功率
            auc = np.mean(success)                                  # 计算AUC
            success_all_video.append(success)
            results_eval[model][video] = auc
        results_eval[model]['all_video'] = np.mean(success_all_video)
        print(model.split('/')[-1] + ' : ', np.mean(success_all_video))
    json.dump(results_eval, open(save_path, 'w'))


# 跟踪
def validation(args):
     # ------------ prepare data  -----------
    if config.MACHINE_TYPE == Machine_type.Linux:
        data_path = '/home/sjl/dataset/otb/'
    else:
        data_path = "D:\\dataset\\otb\\" # r'E:\dataset\OTB'
    if '50' in args.videos:
        direct_file = data_path + 'tb_50.txt'
    elif '100' in args.videos:
        direct_file = data_path + 'tb_100.txt'
    elif '13' in args.videos:
        direct_file = data_path + 'cvpr13.txt'
    else:
        raise ValueError('videos setting wrong')
    with open(direct_file, 'r') as f:
        direct_lines = f.readlines()
    video_names = np.sort([x.split('\t')[0] for x in direct_lines])
    video_paths = [data_path + x for x in video_names]

    # ------------ prepare models  -----------
    # 可以一次性操作多个模型
    input_paths = [os.path.abspath(x) for x in args.model_paths]
    model_paths = []
    for input_path in input_paths:
        if os.path.isdir(input_path):
            input_path = os.path.abspath(input_path)
            model_path = sorted([x for x in os.listdir(input_path) if 'pth' in x], key=embeded_numbers)
            model_path = [input_path + '/' + x for x in model_path]
            model_paths.extend(model_path)
        elif os.path.isfile(input_path):
            model_path = os.path.abspath(input_path)
            model_paths.append(model_path)
        else:
            raise ValueError('model_path setting wrong')

    # ------------ starting validation  -----------
    print("start validation!")
    results = {}
    for model_path in tqdm(model_paths, total=len(model_paths)):
        results[os.path.abspath(model_path)] = {}
        for video_path in tqdm(video_paths, total=len(video_paths)):
            groundtruth_path = video_path + '/groundtruth_rect.txt'
            assert os.path.isfile(groundtruth_path), 'groundtruth of ' + video_path + ' doesn\'t exist'
            with open(groundtruth_path, 'r') as f:
                boxes = f.readlines()
            # 有些是,号分隔;有些是空格分隔
            if ',' in boxes[0]:
                boxes = [list(map(int, box.split(','))) for box in boxes]
            else:
                boxes = [list(map(int, box.split())) for box in boxes]
            # gt的cx,cy需要减1
            boxes = [np.array(box) - [1, 1, 0, 0] for box in boxes]
            # 跟踪代码
            result = run_SiamFPN(video_path, model_path, boxes)

            result_boxes = [np.array(box) + [1, 1, 0, 0] for box in result['res']]
            results[os.path.abspath(model_path)][video_path.split('/')[-1]] = [box.tolist() for box in result_boxes]

    # with Pool(processes=mp.cpu_count()) as pool:
    #     for ret in tqdm(pool.imap_unordered(
    #             functools.partial(worker, video_paths), model_paths), total=len(model_paths)):
    #         results.update(ret)

    json.dump(results, open(args.save_name, 'w'))


if __name__ == '__main__':
    program_name = os.getcwd().split('/')[-1]
    setproctitle.setproctitle('sjl test ' + program_name)
    parser = argparse.ArgumentParser(description='Test some models on OTB2015 or OTB2013')  # 创建一个解析对象
    parser.add_argument('--model_paths', '-ms', dest='model_paths', nargs='+',
                        help='the path of models or the path of a model or folder')
    parser.add_argument('--videos', '-v', dest='videos')                                    # choices=['tb50', 'tb100', 'cvpr2013']
    parser.add_argument('--save_name', '-n', dest='save_name', default='result.json')       # 向该对象中添加你要关注的命令行参数和选项
    args = parser.parse_args()                                                              # 进行解析
    
    # 临时测试,直接给args赋值
    args.videos = "50" # "50,100,13"
    if config.MACHINE_TYPE == Machine_type.Linux:    
        args.model_paths = [r'/home/selonsy/workspace/SiamFPN/data/models/otb_siamfpn_46_trainloss_1.2423_validloss_nan.pth'] # 
        args.save_name = "./data/results/otb_result_otb{0}.json".format(args.videos) 
    else:
        args.model_paths = [r"D:\workspace\MachineLearning\HelloWorld\59version\data\models\siamfpn_50_trainloss_1.1085_validloss_0.9913.pth"]
        args.save_name = r"D:\workspace\MachineLearning\HelloWorld\59version\data\results\result_otb{0}.json".format(args.videos)
    # 跟踪
    validation(args)
    # 验证
    evaluation(args.videos)
