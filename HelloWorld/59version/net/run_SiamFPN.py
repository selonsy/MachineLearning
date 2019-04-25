import numpy as np
import time
import cv2
import glob
import os
import sys

sys.path.append(os.getcwd())
from net.tracker import SiamFPNTracker
from net.config import config
from tqdm import tqdm
from IPython import embed


def run_SiamFPN(seq_path, model_path, init_box):
    '''
        seq_path:str,视频序列地址  eg:'D:\\dataset\\otb\\Basketball' 

        model_path:str,跟踪模型地址 eg:[197, 213, 34, 81]
        init_box:array,初始帧box
    '''
    x, y, w, h = init_box # 这里的x,y是左上角的坐标
    tracker = SiamFPNTracker(model_path)
    res = []
    frames = [seq_path + '/img/' + x for x in np.sort(os.listdir(seq_path + '/img'))]
    frames = [x for x in frames if '.jpg' in x]
    title = seq_path.split('/')[-1]
    # ToDo:这里貌似不需要进行切片，在我下载的数据集里面，图片和gt的数量是对应的
    # 但是记得之前下载的数据存在图片数量多于gt的情况，所以需要进行切片操作，故这里暂时屏蔽
    # 后面可以具体查看以下几个视频的跟踪情况，确定是否有影响
    # if title == 'David':        # ? frames 471 gt 471
    #     frames = frames[299:]
    # elif title == 'Football1':  # ? frames 74 gt 74
    #     frames = frames[0:74]
    # elif title == 'Freeman3':   # ? frames 460 gt 460
    #     frames = frames[0:460]
    # elif title == 'Freeman4':   # ? frames 283 gt 283
    #     frames = frames[0:283]
    # elif title == 'Diving':     # ? frames 215 gt 215
    #     frames = frames[:215]
    # elif title == 'Tiger1':     # ? frames 354 gt 354
    #     frames = frames[5:]
    # starting tracking
    tic = time.clock()
    for idx, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.imread(frame)
        # frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        if idx == 0:
            tracker.init(frame, init_box) # array([197, 213, 34, 81])
            bbox = (x + w / 2 - 1 / 2, y + h / 2 - 1 / 2, w, h) # (213.5, 253.0, 34, 81)
            bbox = np.array(bbox).astype(np.float64)
        else:
            bbox, score = tracker.update(frame)  # x,y,w,h
            bbox = np.array(bbox)
        res.append(list((bbox[0] - bbox[2] / 2 + 1 / 2, bbox[1] - bbox[3] / 2 + 1 / 2, bbox[2], bbox[3])))
    duration = time.clock() - tic
    result = {}
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(len(frames) / duration, 3)
    return result

if __name__ == "__main__":
    seq_path = 'D:\\dataset\\otb\\Basketball'
    model_path = r"D:\workspace\MachineLearning\HelloWorld\59version\data\models\siamfpn_50_trainloss_1.1085_validloss_0.9913.pth"
    init_box = np.array([197, 213, 34, 81])
    run_SiamFPN(seq_path,model_path,init_box)