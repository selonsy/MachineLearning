import lmdb
import cv2
import numpy as np
import os
import hashlib
import functools

from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool
from IPython import embed
import multiprocessing as mp


def worker(video_name):
    image_names = glob(video_name + '/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        try:
            _, img_encode = cv2.imencode('.jpg', img)
        except:
            embed()
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv


def create_lmdb(data_dir, output_dir, num_threads=mp.cpu_count()):
    video_names = glob(data_dir + '/*')
    video_names = [x for x in video_names if os.path.isdir(x)]
    db = lmdb.open(output_dir, map_size=int(1024*1024*1024))  # 50e9 46.5G  单位是Byte
    # with Pool(processes=num_threads) as pool:
    #     for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
    #         with db.begin(write=True) as txn:
    #             for k, v in ret.items():
    #                 txn.put(k, v)    
    for x in video_names:
        ret = worker(x)
        with db.begin(write=True) as txn:
            for k, v in ret.items():
                txn.put(k, v) 

if __name__ == '__main__':
    # Fire(create_lmdb)
    data_dir = "D:\\workspace\\vot\\asimo\\HelloRicky-SiameseRPN\\Siamese-RPN\\ILSVRC_Crops"
    output_dir = "D:\\workspace\\vot\\asimo\\HelloRicky-SiameseRPN\\Siamese-RPN\\ILSVRC_Crops.lmdb" 
    create_lmdb(data_dir,output_dir)
    print("Done")
