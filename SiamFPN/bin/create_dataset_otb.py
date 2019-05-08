import functools
import multiprocessing as mp
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from glob import glob
from multiprocessing import Pool
sys.path.append(os.getcwd())
import cv2
import numpy as np
from fire import Fire
from IPython import embed
from tqdm import tqdm

from lib.utils import add_box_img, get_instance_image
from net.config import *



def worker(output_dir, video_dir):    
    image_names = glob(os.path.join(video_dir, 'img/*.jpg'))
    if config.MACHINE_TYPE == Machine_type.Windows:
        image_names = sorted(image_names, key=lambda x: int(x.replace('/',"\\").split("\\")[-1].split('.')[0]))            
        video_name = video_dir.replace('/',"\\").split("\\")[-1]
    else:
        image_names = sorted(image_names, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        video_name = video_dir.split('/')[-1]
    save_folder = os.path.join(output_dir, video_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    trajs = {0:[]}
    trkid = 0
    anno_str = "groundtruth_rect.txt"
    # if video_name == "Jogging": # ToDo:这个视频有两个人的跟踪框,暂时用一个的训练
    #     anno_str = "groundtruth_rect.1.txt"
    vid_anno_path = os.path.join(video_dir, anno_str)
    with open(vid_anno_path, 'r') as f:
        bboxs = f.readlines()
        # 有些是,号分隔;有些是空格或者制表符分隔
        if ',' in bboxs[0]:
            bboxs = [list(map(int, box.split(','))) for box in bboxs]
        else:
            bboxs = [list(map(int, box.split())) for box in bboxs]
        # gt的cx,cy需要减1
        bboxs = [np.array(box) - [1, 1, 0, 0] for box in bboxs]
    assert len(bboxs)==len(image_names),'bboxs的数量必须要和image_names的一致'  
    for i, image_name in enumerate(image_names):
        img = cv2.imread(image_name)
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        bbox = bboxs[i] # 这里的bbox是 x,y,w,h (x,y为左上角的坐标)
        if (bbox == [-1,-1,0,0]).all():
            continue # ToDo:有两个视频的最后一句是四个0,先跳过,如:Board
        filename = os.path.basename(image_name).split('.')[0]
        trajs[0].append(filename)
        instance_crop_size = int(np.ceil((config.instance_size + config.max_translate * 2) * (1 + config.scale_resize)))
        # 转换成cx,cy,w,h格式
        bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]]) # (213.5, 253.0, 34, 81)        
        instance_img, w, h, _ = get_instance_image(img, bbox,
                                                    config.exemplar_size, instance_crop_size,
                                                    config.context_amount,
                                                    img_mean)
        instance_img_name = os.path.join(save_folder,filename + ".{:02d}.x_{:.2f}_{:.2f}.jpg".format(trkid, w, h))
        cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs


def processing(vid_dir, output_dir, num_threads=mp.cpu_count()):       
    # # 直接遍历vid_dir下面的所有的文件夹
    # _all_videos = glob(vid_dir + '*')
    # all_videos = []
    # # 过滤掉tb_50.txt\tb_100.txt等几个文件
    # for video in _all_videos:
    #     if os.path.isdir(video):
    #         all_videos.append(video)

    # 仅处理OTB指定的数据集,避免出现某些奇怪的错误(gt的名称不正确等等)
    videos = ['cvpr13.txt','tb_50.txt','tb_100.txt']
    all_videos = []
    for video in videos:
        direct_file = vid_dir + video
        with open(direct_file, 'r') as f:
            direct_lines = f.readlines()
        video_names = np.sort([x.split('\t')[0] for x in direct_lines])
        all_videos.extend(video_names)
    all_videos = set(all_videos)
    all_videos = [vid_dir+video for video in all_videos]
    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    # 下面的代码无法调试(可以在后面跑的时候打开，显示进度条)
    # if config.MACHINE_TYPE == Machine_type.Linux:
    #     num_threads = 6 # Linux服务器开5个进程     
    with Pool(processes=num_threads) as pool: # 多进程并发操作进程池
        for ret in tqdm(pool.imap_unordered(
                functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            meta_data.append(ret)

    # # 改为下面的直接进行(可以调试,方便处理跑到中途挂了的情况)
    # for i,video_dir in enumerate(all_videos):
    #     # if i <= 96:
    #     #     continue 
    #     ret = worker(output_dir,video_dir)
    #     # ret = (video_dir.replace('/',"\\").split("\\")[-1],{})
    #     meta_data.append(ret)

    # save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    # Fire(processing)
    if config.MACHINE_TYPE == Machine_type.Linux:
        # linux
        vid_dir = r'/home/sjl/dataset/otb/'    
        output_dir = r'/home/sjl/dataset/otb_Crops'
    else:        
        # windows
        vid_dir = 'D:\\dataset\\OTB\\' # r"D:\workspace\MachineLearning\HelloWorld\59version\dataset\ILSVRC"    
        output_dir = 'D:\\dataset\\OTB_Crops' # r"D:\workspace\MachineLearning\HelloWorld\59version\dataset\ILSVRC_Crops"
    processing(vid_dir, output_dir)
