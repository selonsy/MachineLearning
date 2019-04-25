from IPython import embed

import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
import multiprocessing as mp

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

sys.path.append(os.getcwd())
from net.config import *
from lib.utils import get_instance_image, add_box_img

def worker(output_dir, video_dir):
    if 'YT-BB' in video_dir:
        image_names = glob(os.path.join(video_dir, '*.jpg'))
        image_names = sorted(image_names, key=lambda x: int(x.split('/')[-1].split('_')[1]))
        video_name = video_dir.split('/')[-1]
        save_folder = os.path.join(output_dir, video_name)
        anno_path = '/mnt/diska1/YT-BB/xml/youtube_dection_frame_xml_temp'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        trajs = {}
        for image_name in image_names:
            img = cv2.imread(image_name)
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
            anno_name = os.path.join(anno_path, video_name, image_name.split('/')[-1]).replace('.jpg', '.xml')
            tree = ET.parse(anno_name)
            root = tree.getroot()
            bboxes = []
            filename = root.find('filename').text
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                bbox = list(map(int, [bbox.find('xmin').text,
                                      bbox.find('ymin').text,
                                      bbox.find('xmax').text,
                                      bbox.find('ymax').text]))

                trkid = int(obj.find('trackid').text)
                if trkid in trajs:
                    trajs[trkid].append(filename)
                else:
                    trajs[trkid] = [filename]
                instance_crop_size = int(np.ceil((config.instance_size + config.max_translate * 2) * (1 + config.scale_resize)))
                bbox = np.array(
                    [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, bbox[2] - bbox[0] + 1,
                     bbox[3] - bbox[1] + 1])

                instance_img, w, h, _ = get_instance_image(img, bbox,
                                                           config.exemplar_size, instance_crop_size,
                                                           config.context_amount,
                                                           img_mean)
                instance_img_name = os.path.join(save_folder,
                                                 filename + ".{:02d}.x_{:.2f}_{:.2f}.jpg".format(trkid, w, h)) # 在文件名中记录宽高
                cv2.imwrite(instance_img_name, instance_img)
    else:
        # video_dir  = ...\ILSVRC\Data\VID\train\ILSVRC2015_VID_train_0000\ILSVRC2015_train_00001000
        # output_dir = ...\ILSVRC_Crops
        image_names = glob(os.path.join(video_dir, '*.JPEG'))
        if config.MACHINE_TYPE == Machine_type.Windows:
            image_names = sorted(image_names, key=lambda x: int(x.replace('/',"\\").split("\\")[-1].split('.')[0]))            
            video_name = video_dir.replace('/',"\\").split("\\")[-1]
        else:
            image_names = sorted(image_names, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            video_name = video_dir.split('/')[-1]
        save_folder = os.path.join(output_dir, video_name) # ...\ILSVRC_Crops\ILSVRC2015_train_00001000
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        trajs = {}
        for image_name in image_names:
            # image_name = ...\ILSVRC\Data\VID\train\ILSVRC2015_VID_train_0000\ILSVRC2015_train_00001000\000000.JPEG
            img = cv2.imread(image_name)
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))            
            anno_name = image_name.replace('Data', 'Annotations')
            anno_name = anno_name.replace('JPEG', 'xml')
            # anno_name = ...\ILSVRC\Annotations\VID\train\ILSVRC2015_VID_train_0000\ILSVRC2015_train_00001000\000000.xml
            #<editor-fold desc="000000.xml示例">
               '''
               <annotation>
                    <folder>ILSVRC2015_VID_train_0000/ILSVRC2015_train_00001000</folder>
                    <filename>000000</filename>
                    <source>
                        <database>ILSVRC_2015</database>
                    </source>
                    <size>
                        <width>1280</width>
                        <height>720</height>
                    </size>
                    <object>
                        <trackid>0</trackid>
                        <name>n01662784</name>
                        <bndbox>
                            <xmax>888</xmax>
                            <xmin>0</xmin>
                            <ymax>718</ymax>
                            <ymin>81</ymin>
                        </bndbox>
                        <occluded>1</occluded>
                        <generated>0</generated>
                    </object>
                </annotation>
               '''               
            #</editor-fold>
            tree = ET.parse(anno_name)
            root = tree.getroot()
            bboxes = []
            filename = root.find('filename').text # filename = 000000
            for obj in root.iter('object'): # ToDo：有的xml下面有多个object，即在一张图里面有多个bbox,按照trkid的不同多次crop
                                            # 即会出现说在video_dir里面只有20张图片,但是最后剪切到output_dir的可能有40张图片,如果每个对应的xml里面有两个object的话
                bbox = obj.find('bndbox')   
                bbox = list(map(int, [bbox.find('xmin').text,    # 0
                                      bbox.find('ymin').text,    # 81
                                      bbox.find('xmax').text,    # 888
                                      bbox.find('ymax').text]))  # 718
                trkid = int(obj.find('trackid').text) # trkid = 0
                if trkid in trajs:
                    trajs[trkid].append(filename)
                else:
                    trajs[trkid] = [filename]
                instance_crop_size = int(
                    np.ceil((config.instance_size + config.max_translate * 2) * (1 + config.scale_resize)))
                # 改成cx,cy,w,h的格式
                bbox = np.array(
                    [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, bbox[2] - bbox[0] + 1,
                     bbox[3] - bbox[1] + 1])
                # ToDo:这里补足空白的地方还是不大清楚
                instance_img, w, h, _ = get_instance_image(img, bbox,
                                                           config.exemplar_size, instance_crop_size,
                                                           config.context_amount,
                                                           img_mean)
                instance_img_name = os.path.join(save_folder,filename + ".{:02d}.x_{:.2f}_{:.2f}.jpg".format(trkid, w, h))
                # img =(1280,720)  w/h = 1.394
                # instance_img_name = ...\ILSVRC_Crops\ILSVRC2015_train_00001000\000000.00.x_74.19_53.24.jpg"                
                # 340 * 340 大小的, 340 = ( 271 + 12 * 2 ) * ( 1 + 0.15 )
                # w = 74.19 , h = 53.24 是处理后的图片大小, w/h=1.394
                # 不足的空白地方,使用img_mean进行均值填充
                cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs # trajs好像只有一个key,且key的值为0,即trajs[0] = [000000,000001,...]


def processing(vid_dir, ytb_dir, output_dir, num_threads=mp.cpu_count()):
    # get all 4417 videos in vid and all video in ytbb
    vid_video_dir = os.path.join(vid_dir, 'Data/VID')
    ytb_video_dir = ytb_dir
    # all_videos 包括ILSVRC2015和YoutubeBB的数据
    all_videos = glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(vid_video_dir, 'val/*')) + \
                 glob(os.path.join(ytb_video_dir, 'v*/youtube_dection_frame_temp/*'))
                # 怎么会莫名其妙多了一行
                #  glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \

    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 下面的代码无法调试(可以在后面跑的时候打开，显示进度条)
    if config.MACHINE_TYPE == Machine_type.Linux:
        num_threads = 6 # Linux服务器开5个进程 
    with Pool(processes=num_threads) as pool: # 多进程并发操作进程池
        for ret in tqdm(pool.imap_unordered(
                functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            meta_data.append(ret)

    # # 改为下面的直接进行(可以调试)
    # for video_dir in all_videos:
    #     ret = worker(output_dir,video_dir)
    #     meta_data.append(ret)

    # save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    # Fire(processing)
    if config.MACHINE_TYPE == Machine_type.Linux:
        # linux
        vid_dir = r'/home/zzx/vot/VGG/ILSVRCBAK'
        ytb_dir = ""
        output_dir = r'/home/sjl/dataset/ILSVRC2015_Crops'
    else:
        # windows
        vid_dir = r'E:\dataset\ILSVRC2015' # r"D:\workspace\MachineLearning\HelloWorld\59version\dataset\ILSVRC"
        ytb_dir = ""
        output_dir = r'E:\dataset\ILSVRC2015_Crops' # r"D:\workspace\MachineLearning\HelloWorld\59version\dataset\ILSVRC_Crops"   
    processing(vid_dir, ytb_dir, output_dir)

