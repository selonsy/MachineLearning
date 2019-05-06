import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from lib.generate_anchors import generate_anchors_fpn, generate_track_windows
from net.fpn import SiamFPN50, SiamFPN101, SiamFPN152
from IPython import embed
from lib.utils import get_exemplar_image, get_instance_image, box_transform_inv
from lib.custom_transforms import ToTensor
from net.config import config
from net.network import SiameseAlexNet

torch.set_num_threads(1)  # otherwise pytorch will take all cpus

class SiamRPNTracker_bak:
    def __init__(self, model_path):
        self.model = SiameseAlexNet()
        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(torch.load(model_path)['model'])
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

        valid_scope = 2 * config.valid_scope + 1
        # self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
        #                                 config.anchor_ratios,
        #                                 valid_scope)
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(
            np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.pos = np.array(
            [bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based
        # self.pos = np.array(
        #     [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2])  # same to original code
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2,
                              bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])
        # self.bbox = np.array(
        #     [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]])  # same to original code
        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))

        exemplar_img, _, _ = get_exemplar_image(frame, self.bbox,
                                                config.exemplar_size, config.context_amount, self.img_mean)
        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        self.model.track_init(exemplar_img.cuda())

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        instance_img, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                         config.instance_size,
                                                         config.context_amount, self.img_mean)
        instance_img = self.transforms(instance_img)[None, :, :, :]
        pred_score, pred_regression = self.model.track(instance_img.cuda())

        pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                 2,
                                                                                                                 1)
        pred_offset = pred_regression.reshape(-1, 4,
                                              config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                 2,
                                                                                                                 1)
        delta = pred_offset[0].cpu().detach().numpy()
        box_pred = box_transform_inv(self.anchors, delta)
        score_pred = F.softmax(pred_conf, dim=2)[
            0, :, 1].cpu().detach().numpy()

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) /
                     (sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = change((self.target_sz[0] / self.target_sz[1]) /
                     (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * score_pred
        pscore = pscore * (1 - config.window_influence) + \
            self.window * config.window_influence
        best_pscore_id = np.argmax(pscore)
        target = box_pred[best_pscore_id, :] / scale_x

        lr = penalty[best_pscore_id] * \
            score_pred[best_pscore_id] * config.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))
        return self.bbox, score_pred[best_pscore_id]


class SiamFPNTracker:
    def __init__(self, model_path):
        self.model = SiamFPN50()
        if not config.CUDA:
            checkpoint = torch.load(model_path, map_location='cpu') 
        else:   
            checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        if config.CUDA:
            self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

        # valid_scope = 2 * config.valid_scope + 1
        # self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
        #                                 config.anchor_ratios,
        #                                 valid_scope)
        # backbone_shapes = config.FEATURE_MAP_SIZES
        # self.anchors = generate_pyramid_anchors(config.FPN_ANCHOR_SCALES,
        #                                         config.FPN_ANCHOR_RATIOS,
        #                                         backbone_shapes,
        #                                         config.BACKBONE_STRIDES,
        #                                         config.FPN_ANCHOR_STRIDE)
        self.anchors = generate_anchors_fpn(config.BACKBONE_STRIDES,
                                            config.FPN_ANCHOR_SCALES,
                                            config.FPN_ANCHOR_RATIOS,
                                            config.FEATURE_MAP_SIZE)                                            
        # self.window = np.tile(
        #     np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],[config.anchor_num, 1, 1]).flatten()
        self.windows = generate_track_windows()
  
    def init(self, frame, bbox):
        """ initialize siamfpn tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.pos = np.array(
            [bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based
        # self.pos = np.array(
        #     [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2])  # same to original code
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2,
                              bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])
        # self.bbox = np.array(
        #     [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]])  # same to original code
        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))

        exemplar_img, _, _ = get_exemplar_image(frame, self.bbox,
                                                config.exemplar_size, config.context_amount, self.img_mean)
        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        if config.CUDA:
            exemplar_img = exemplar_img.cuda()
        # 调用跟踪初始化代码,先计算模板帧的特征卷积结果,单次学习,后面暂时不更新
        # 考虑高分样本反馈,模板帧的特征卷积结果可能发生变化,可以加权处理
        # 权重就考虑样本的得分,默认初始帧的得分为1.其余样本的得分0.9<x<1 0.9暂定
        self.model.track_init(exemplar_img) # torch.Size([1, 3, 127, 127])

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        # ToDo:看看这几个返回的值都是些什么东西
        instance_img, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size, 
                config.instance_size, config.context_amount, self.img_mean)
        # cv2.imshow("update", instance_img)
        instance_img = self.transforms(instance_img)[None, :, :, :]
        if config.CUDA:
            instance_img = instance_img.cuda()
        pred_scores, pred_regressions = self.model.track(instance_img)
        
        def change(r):
            # np.maximum：(X, Y, out=None)；X 与 Y 逐位比较取其大者
            return np.maximum(r, 1. / r)

        def sz(w, h):
            # 在bounding_box
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # 这里比较复杂，我们先分层预测，每层选最佳的匹配，并记录下score
        # 后面对score进行排序，返回最高得分的预测结果
        # PS：记录下不是 19*19 组获得最高评分的次数，分析FPN的效果        
        results_bboxs  = []
        results_scores = []
        for i in range(len(pred_scores)):
            if i!=1:
                continue
            pred_score = pred_scores[i] # torch.Size([1, 6, 37, 37])
            pred_regression = pred_regressions[i] # torch.Size([1, 12, 37, 37])
            score_size = config.FEATURE_MAP_SIZE[i] # 37
            anchor_num = 3 # 暂时定为3 即[0.5,1,2]

            pred_conf = pred_score.reshape(-1, 2, anchor_num * score_size * score_size).permute(0,2,1) # torch.Size([1, 4107, 2])
            pred_offset = pred_regression.reshape(-1, 4, anchor_num * score_size * score_size).permute(0,2,1) # # torch.Size([1, 4107, 4])
            
            delta = pred_offset[0].cpu().detach().numpy() # (4107, 4)
            box_pred = box_transform_inv(self.anchors[i], delta) # (4107, 4)
            score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy() # (4107,)

            # # 不进行后面的尺度惩罚等等,直接选最大得分的试试
            # best_pscore_id = np.argmax(score_pred)
            # target = box_pred[best_pscore_id, :] / scale_x
            # res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[0])
            # res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[1])
            # res_w = np.clip(target[2], 
            #         config.min_scale * self.origin_target_sz[0], 
            #         config.max_scale * self.origin_target_sz[0])
            # res_h = np.clip(target[3], 
            #         config.min_scale * self.origin_target_sz[1], 
            #         config.max_scale * self.origin_target_sz[1])
            # bbox = np.array([res_x, res_y, res_w, res_h])          
            # results_bboxs.append(bbox)
            # results_scores.append(score_pred[best_pscore_id])
            # continue

            # 进行尺度惩罚等措施,但是相关的超参数不知道怎么确定
            s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) /
                        (sz_wh(self.target_sz * scale_x)))  # scale penalty (4107,)
            r_c = change((self.target_sz[0] / self.target_sz[1]) /
                        (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty (4107,)
            penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k) # (4107,) penalty_k=0.22
            pscore = penalty * score_pred # (4107,)
            # window_influence = 0.4
            pscore = pscore * (1 - config.window_influence) + self.windows[i] * config.window_influence # (4107,)
            best_pscore_id = np.argmax(pscore)
            target = box_pred[best_pscore_id, :] / scale_x

            lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box # lr_box = 0.3

            res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[0])
            res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[1])

            # min_scale = 0.1 max_scale = 10
            # numpy.clip(a, a_min, a_max, out=None) 
            # 将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
            res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, 
                    config.min_scale * self.origin_target_sz[0], 
                    config.max_scale * self.origin_target_sz[0])
            res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, 
                    config.min_scale * self.origin_target_sz[1], 
                    config.max_scale * self.origin_target_sz[1])

            bbox = np.array([res_x, res_y, res_w, res_h])          
            results_bboxs.append(bbox)
            results_scores.append(pscore[best_pscore_id])
        max_score_id = np.argmax(results_scores)   
        _box = results_bboxs[max_score_id]  
        _socre = results_scores[max_score_id]
        # results = sorted(results.items,key=lambda x:x[1], reverse=True) # 按照得分进行排序
        # _box = results.keys[0]
        # _socre = results[0]
        x, y, w, h = _box
        self.pos = np.array([x, y])
        self.target_sz = np.array([w, h])
        self.bbox = (
                np.clip(_box[0], 0, frame.shape[1]).astype(np.float64),
                np.clip(_box[1], 0, frame.shape[0]).astype(np.float64),
                np.clip(_box[2], 10, frame.shape[1]).astype(np.float64),
                np.clip(_box[3], 10, frame.shape[0]).astype(np.float64))

        return self.bbox, _socre


if __name__ == "__main__":
    model_path = r"D:\workspace\MachineLearning\HelloWorld\59version\data\models\siamfpn_50_trainloss_1.1085_validloss_0.9913.pth"

    model = SiamFPNTracker(model_path)
    pass
