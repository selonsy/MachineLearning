import lmdb
import cv2
import numpy as np
import os
import sys
import hashlib
import functools
sys.path.append(os.getcwd())
from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool
from IPython import embed
import multiprocessing as mp
from net.config import *

def worker(video_name):
    image_names = glob(video_name + '/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv


def create_lmdb(data_dir, output_dir, num_threads=mp.cpu_count()):
    video_names = glob(data_dir + '/*')
    video_names = [x for x in video_names if 'meta_data.pkl' not in x]
    # video_names = [x for x in video_names if os.path.isdir(x)]
    db = lmdb.open(output_dir, map_size=int(2e9)) # 200e9表示200G,1024*1024*1024表示1G,单位是Byte

    # 同理
    if config.MACHINE_TYPE == Machine_type.Linux:
        num_threads = 6 # Linux服务器开5个进程 
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)
    # for x in video_names:
    #     ret = worker(x)
    #     with db.begin(write=True) as txn:
    #         for k, v in ret.items():
    #             txn.put(k, v) 

if __name__ == '__main__':
    # Fire(create_lmdb)    
    if config.MACHINE_TYPE == Machine_type.Linux:
        # linux
        # data_dir = r'/home/sjl/dataset/ILSVRC2015_Crops'    
        # output_dir = r'/home/sjl/dataset/ILSVRC2015_Crops_Lmdb'
        data_dir = r'/home/sjl/dataset/otb_Crops'    
        output_dir = r'/home/sjl/dataset/otb_Crops_Lmdb'
    else:
        # windows
        data_dir = r'D:\dataset\OTB_Crops' # r"D:\workspace\MachineLearning\HelloWorld\59version\dataset\ILSVRC_Crops"    
        output_dir = r'D:\dataset\OTB_Crops_Lmdb' # r"D:\workspace\MachineLearning\HelloWorld\59version\dataset\ILSVRC_Crops_Lmdb"        
    create_lmdb(data_dir, output_dir)