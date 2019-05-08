import torch
import cv2
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pickle
import lmdb
import hashlib
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from lib.generate_anchors import generate_pyramid_anchors,generate_anchors_rpn,generate_anchors_fpn
from net.config import config
from lib.utils import box_transform, compute_iou, add_box_img, crop_and_pad

from IPython import embed


class ImagnetVIDDataset(Dataset):
    def __init__(self, db_path, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]: x[1] for x in self.meta_data}
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        # self.txn = db.begin(write=False)
        self.txn = None
        self.db_path = db_path
        # print(self.db_path)
        # num表示dataloader的总数,除以batch_size即为循环的轮次
        self.num = len(self.video_names) if config.pairs_per_video_per_epoch is None or not training \
            else config.pairs_per_video_per_epoch * len(self.video_names)

        # data augmentation
        self.max_stretch = config.scale_resize
        self.max_translate = config.max_translate
        self.random_crop_size = config.instance_size
        self.center_crop_size = config.exemplar_size

        self.training = training

        # # 原计算单层锚标签
        # valid_scope = 2 * config.valid_scope + 1
        # self.anchors_ori = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
        #                                 config.anchor_ratios, valid_scope)
        
        # 分层计算锚标签
        # backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE) 
        # array([[256,256],[128,128],[64,64],[32,32],[16,16]])
        # 上面的特征图太大了,不符合我的模型的输出,采用下面自定义的
        # self.anchors = generate_pyramid_anchors(config.FPN_ANCHOR_SCALES,
        #                                         config.FPN_ANCHOR_RATIOS,
        #                                         config.FEATURE_MAP_SIZES,
        #                                         config.BACKBONE_STRIDES,
        #                                         config.FPN_ANCHOR_STRIDE) 
        # 采用原作者里面的anchor计算方式，不适用FPN里面的计算方式
        self.anchors = generate_anchors_fpn(config.BACKBONE_STRIDES,
                                            config.FPN_ANCHOR_SCALES,
                                            config.FPN_ANCHOR_RATIOS,
                                            config.FEATURE_MAP_SIZE)

    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        # print(key)
        if not self.txn:            
            # print("init lmdb in imread")
            db = lmdb.open(self.db_path, readonly=True, map_size=int(1024*1024*1024))
            self.txn = db.begin(write=False)

        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        scale_w = int(w * scale_w) / w
        scale_h = int(h * scale_h) / h
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, shape, cv2.INTER_LINEAR), gt_w, gt_h

    def compute_target(self, anchors, box):
        regression_target = box_transform(anchors, box)

        iou = compute_iou(anchors, box).flatten()
        # print(np.max(iou))
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label

        # pos_index = np.random.choice(pos_index, config.num_pos)
        # neg_index = np.random.choice(neg_index, config.neg_pos)
        # max_index = np.argsort(iou.flatten())[-20:]
        # boxes = anchors[max_index]

    def __getitem__(self, idx):
        while True:
            idx = idx % len(self.video_names)
            video = self.video_names[idx]
            trajs = self.meta_data[video]
            # sample one trajs
            if len(trajs.keys()) > 0:
                break
            else:
                idx = np.random.randint(self.num)

        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]
        assert len(traj) > 1, "video_name: {}".format(video)
        # sample exemplar
        exemplar_idx = np.random.choice(list(range(len(traj))))
        # exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid))

        if 'ILSVRC2015' in video:
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
        else:
            # exemplar_name = \
            #     glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{}.x*.jpg".format(trkid)))[0]
            # print(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))
            # print(glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid))))
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
        exemplar_img = self.imread(exemplar_name)
        # exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        # sample instance
        if 'ILSVRC2015' in exemplar_name:
            frame_range = config.frame_range_vid
        else:
            frame_range = config.frame_range_ytb
        low_idx = max(0, exemplar_idx - frame_range)
        up_idx = min(len(traj), exemplar_idx + frame_range + 1)

        # 样本权重,离中心越远被选中的概率越高(即后面选的实例图片要距模板图片足够远)
        # create sample weight, if the sample are far away from center
        # the probability being choosen are high
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)

        if 'ILSVRC2015' in video:
            instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
        else:
            # instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{}.x*.jpg".format(trkid)))[0]
            instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]

        instance_img = self.imread(instance_name)
        # instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        # 数据集预处理的时候,文件名称有保存gt框的宽高
        gt_w, gt_h = float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])

        # 暂时屏蔽掉 # 25%的灰度图处理 
        # if np.random.rand(1) < config.gray_ratio:
        #     exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
        #     # exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB) # 这两句应该要屏蔽掉
        #     instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
        #     # instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
        
        if config.exem_stretch:
            exemplar_img, _, _ = self.RandomStretch(exemplar_img, 0, 0)
        exemplar_img_mean = np.mean(exemplar_img, axis=(0, 1))
        exemplar_img, _ = crop_and_pad(exemplar_img, (exemplar_img.shape[1] - 1) / 2,
                                       (exemplar_img.shape[0] - 1) / 2, self.center_crop_size,
                                       self.center_crop_size,exemplar_img_mean)
        # exemplar_img_np = exemplar_img.copy()
        exemplar_img = self.z_transforms(exemplar_img)

        instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
        im_h, im_w, _ = instance_img.shape
        cy_o = (im_h - 1) / 2
        cx_o = (im_w - 1) / 2
        cy = cy_o + np.random.randint(- self.max_translate, self.max_translate + 1)
        cx = cx_o + np.random.randint(- self.max_translate, self.max_translate + 1)
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy
        instance_img_mean = np.mean(instance_img, axis=(0, 1))
        instance_img, scale = crop_and_pad(instance_img, cx, cy, self.random_crop_size, self.random_crop_size,instance_img_mean)

        # frame = add_box_img(instance_img, np.array([[gt_cx, gt_cy, gt_w, gt_h]]), color=(0, 255, 255))
        # empty_img = np.zeros_like(frame)
        # empty_img[:127, :127, :] = exemplar_img_np
        # show_img = np.hstack([empty_img, frame])
        # big_img = cv2.resize(show_img, None, fx=2, fy=2)
        # from lib.visual import visual
        # vis = visual(port=6008)
        # vis.plot_img(big_img.transpose(2, 0, 1), win=7, name='a')
        # embed()

        # frame = add_box_img(frame, np.array([[0, 0, gt_w, gt_h]]), color=(0, 255, 0))
        instance_img = self.x_transforms(instance_img)

        # 这里用的box,不是annotation里面的xml数据,我现在暂时假设作者计算的是正确的
        box = np.array(list(map(round, [gt_cx, gt_cy, gt_w, gt_h])))
        # 这里暂时考虑anchor是分层的,所以返回的是多个层的数组
        # regression_target, conf_target = self.compute_target(self.anchors, box)
        regression_targets = []
        conf_targets = []
        for i in range(len(self.anchors)):
            regression_target, conf_target = self.compute_target(self.anchors[i], box)
            regression_targets.append(regression_target)
            conf_targets.append(conf_target.astype(np.int64))

        # 下面有代码可以看到锚标签在原图上面的效果
        # img = instance_img.numpy().transpose(1, 2, 0)
        # pos_index = np.where(conf_target == 1)[0]
        # pos_anchor = self.anchors[pos_index]
        # frame = add_box_img(img, pos_anchor)
        # frame = add_box_img(frame, np.array([[gt_cx, gt_cy, gt_w, gt_h]]), color=(0, 255, 255))
        #
        # # debug the gt_box with original box
        # title = instance_name.split('/')[-1]
        # img = instance_img.numpy().transpose(1, 2, 0)
        # box = np.array([gt_cx, gt_cy, gt_w, gt_h])[None, :]
        # frame = add_box_img(img, box)
        # if 'train' in instance_name:
        #     img_name = '.'.join([instance_name.split('/')[-1].split('.')[0], 'JPEG'])
        #     img_path = glob.glob('/dataset_ssd_quick/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_*/'
        #                          + video + '/' + img_name)[0]
        #     xml_path = glob.glob('/dataset_ssd_quick/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_*/'
        #                          + video + '/' + img_name[:6] + '*')[0]
        #     tree = ET.parse(xml_path)
        #     root = tree.getroot()
        #     bboxes = []
        #     image = cv2.imread(img_path)
        #     for obj in root.iter('object'):
        #         bbox = obj.find('bndbox')
        #         bbox = list(map(int, [bbox.find('xmin').text,
        #                               bbox.find('ymin').text,
        #                               bbox.find('xmax').text,
        #                               bbox.find('ymax').text]))
        #         x_ctr = (bbox[0] + bbox[2]) / 2 - image.shape[1] / 2
        #         y_ctr = (bbox[1] + bbox[3]) / 2 - image.shape[0] / 2
        #         w = bbox[2] - bbox[0]
        #         h = bbox[3] - bbox[1]
        #         bbox = [x_ctr, y_ctr, w, h]
        #         bboxes.append(bbox)
        #     frame2 = add_box_img(image, np.array(bboxes))
        #     frame = frame[:, :, ::-1]
        #     show_img = np.hstack(
        #         [cv2.resize(frame, None, fx=frame2.shape[0] / frame.shape[0], fy=frame2.shape[0] / frame.shape[0]),
        #          frame2])
        # else:
        #     img_name = '.'.join([instance_name.split('/')[-1].split('.')[0], 'JPEG'])
        #     img_path = glob.glob('/dataset_ssd_quick/ILSVRC2015/Data/VID/val/'
        #                          + video + '/' + img_name)[0]
        #     xml_path = glob.glob('/dataset_ssd_quick/ILSVRC2015/Annotations/VID/val/'
        #                          + video + '/' + img_name[:6] + '*')[0]
        #     tree = ET.parse(xml_path)
        #     root = tree.getroot()
        #     bboxes = []
        #     image = cv2.imread(img_path)
        #     for obj in root.iter('object'):
        #         bbox = obj.find('bndbox')
        #         bbox = list(map(int, [bbox.find('xmin').text,
        #                               bbox.find('ymin').text,
        #                               bbox.find('xmax').text,
        #                               bbox.find('ymax').text]))
        #         x_ctr = (bbox[0] + bbox[2]) / 2 - image.shape[1] / 2
        #         y_ctr = (bbox[1] + bbox[3]) / 2 - image.shape[0] / 2
        #         w = bbox[2] - bbox[0]
        #         h = bbox[3] - bbox[1]
        #         box = [x_ctr, y_ctr, w, h]
        #         bboxes.append(box)
        #     frame2 = add_box_img(image, np.array(bboxes))
        #     frame = frame[:, :, ::-1]
        #     show_img = np.hstack(
        #         [cv2.resize(frame, None, fx=frame2.shape[0] / frame.shape[0], fy=frame2.shape[0] / frame.shape[0]),
        #          frame2])
        # cv2.imwrite('gt_box.jpg', show_img)
        # embed()
        # cv2.waitKey(30)

        return exemplar_img, instance_img, regression_targets, conf_targets

    def draw_img(self, img, boxes, name='1.jpg', color=(0, 255, 0)):
        # boxes (x,y,w,h)
        img = img.copy()
        img_ctx = (img.shape[1] - 1) / 2
        img_cty = (img.shape[0] - 1) / 2
        for box in boxes:
            point_1 = img_ctx - box[2] / 2 + box[0], img_cty - box[3] / 2 + box[1]
            point_2 = img_ctx + box[2] / 2 + box[0], img_cty + box[3] / 2 + box[1]
            img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                                color, 2)
        cv2.imwrite(name, img)

    def __len__(self):
        return self.num

if __name__ == "__main__":            
    pass