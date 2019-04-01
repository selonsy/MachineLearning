# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math

from .utils import get_subwindow_tracking

class TrackerConfig4FPN(object):
    '''
    TrackerConfig4FPN 类定义了跟踪器参数
    '''
    # 默认的超参数
    # These are the default hyper-params for DaSiamRPN 0.3827

    # 余弦窗，惩罚大位移
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]

    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)

    # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    total_strides = [4, 8, 16, 32]
    score_sizes = [37,19,10,6] # [math.ceil((instance_size-exemplar_size)/x+1) for x in total_strides]
    # ANCHOR_SCALES = (32, 64, 128, 256)
    anchor_scales = (32, 64, 128, 256)
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.5, 1, 2] 
    
    # scales = [8, ] # 尺度只有1种
    anchor_num =  3 #len(ratios) * len(scales)
    anchors = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    # 参数更新
    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1

def SiamFPN_init(im, target_pos, target_sz, net):
    """
    SiamFPN_init：SiamFPN网络初始化
        :param im: 跟踪的图片
        :param target_pos: 目标的中心点
        :param target_sz:  目标区域的宽高
        :param net: 跟踪网络
    """
    state = dict()
    p = TrackerConfig4FPN()
    # p.update(net.cfg)
    state['im_h'] = im.shape[0] # 图片的高度
    state['im_w'] = im.shape[1] # 图片的宽度

    # if p.adaptive:
    #     # 根据目标和输入图像的大小调整搜索区域，比例小于0.4%，需要调大搜索区域
    #     if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
    #         p.instance_size = 287  # small object big search region
    #     else:
    #         p.instance_size = 271
    #     # 根据网络总步长计算出得分图大小19*19*(2k or 4k)，记得siamRPN图结构上面写的是17*17，是因为上面的instance_size大小为255，这里改成了271.
    #     p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1
    # generate_anchor:构造出以图像中心为原点，格式为[cx, cy, w, h]的锚点矩阵

    p.anchors = generate_anchors4fpn(p.total_strides, p.anchor_scales, p.ratios, p.score_sizes)

    # 求图片RGB三像素的行列均值,len(avg_chans)=3，后面用来进行填充操作
    avg_chans = np.mean(im, axis=(0, 1))

    # wc_z和hc_z表示纹理填充后的宽高，s_z为等效边长
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    
    # initialize the exemplar
    # get_subwindow_tracking：填充并截取出目标
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0)) # z.size=([1, 3, 127, 127])
    net.template(z.cuda()) # 运行 temple 函数计算模板结果

    ## 两种窗
    windows = []
    for score_size in p.score_sizes:
        if p.windowing == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
        elif p.windowing == 'uniform':
            window = np.ones((score_size, score_size))
        window = np.tile(window.flatten(), p.anchor_num)
        if len(windows)==0:
            windows = window
        else:
            windows = np.concatenate((windows,window),axis=0)
        # windows.append(window)
        
    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['windows'] = windows
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state

def SiamFPN_track(state, im):
    """
    docstring here
        :param state: 
        :param im: 
    """
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    windows = state['windows']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    # 计算扩展后尺寸,context_amount=0.5,即扩展0.5倍的（宽+高），即 w = w+0.5*(w+h),h同理
    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    # 在前一个目标位置为搜索区域x提取缩放的截图
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # tracker_eval4fpn 预测出新的位置和得分
    target_pos, target_sz, score = tracker_eval4fpn(net, x_crop.cuda(), target_pos, target_sz * scale_z, windows, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state

def tracker_eval4fpn(net, x_crop, target_pos, target_sz, windows, scale_z, p):
    """
    预测出新的位置和得分
        :param net: 
        :param x_crop: 
        :param target_pos: 
        :param target_sz: 
        :param window: 
        :param scale_z: 
        :param p: 
    """

    # 运行网络的检测分支，得到坐标回归量和得分
    deltas, scores = net(x_crop)      # delta:1,20,19,19     score:1,10,19,19

    # torch.Tensor.permute 置换此张量的尺寸。比如三维就有0，1，2这些dimension。
    # torch.Tensor.contiguous 返回包含与自张量相同的数据的连续张量。如果自张量是连续的，则此函数返回自张量。
    # torch.Tensor.numpy 将自张量作为 NumPy ndarray 返回。此张量和返回的 ndarray 共享相同的底层存储。自张量的变化将反映在 ndarray 中，反之亦然。
    assert len(deltas)==len(scores)
        
    concat_delta=[]
    concat_score=[]

    for i in range(len(deltas)):
        # k=5
        # 置换delta，其形状由 N x 4k x H x W 变为4x(kx17x17)。score形状为2x(kx17x17)，并取其后一半结果
        delta = deltas[i].permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(scores[i].permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()
        
        # 论文中的偏移公式，得到最后的bbox，此利用回归实现，前提是四个参数与gt的值相差不大。
        delta[0, :] = delta[0, :] * p.anchors[i][:, 2] + p.anchors[i][:, 0]
        delta[1, :] = delta[1, :] * p.anchors[i][:, 3] + p.anchors[i][:, 1]
        delta[2, :] = np.exp(delta[2, :]) * p.anchors[i][:, 2]
        delta[3, :] = np.exp(delta[3, :]) * p.anchors[i][:, 3]
        # 4,4107
        print(delta.shape) # (4, 4107)

        if len(concat_delta)==0:
            concat_delta = delta
            concat_score = score
        else:
            concat_delta = np.concatenate((concat_delta,delta),axis=1)
            concat_score = np.concatenate((concat_score,score),axis=0)

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty，尺度惩罚
    s_c = change(sz(concat_delta[2, :], concat_delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (concat_delta[2, :] / concat_delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    # can't multiply sequence by non-int of type 'float'
    # 原因是数组中有不是数字的字符串存在
    pscore = penalty * concat_score 

    # window float
    # pscore按一定权值叠加一个窗分布值。找出最优得分的索引
    
    pscore = pscore * (1 - p.window_influence) + windows * p.window_influence
    best_pscore_id = np.argmax(pscore)

    # 获得目标的坐标及尺寸。delta除以scale_z映射到原图
    target = concat_delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * concat_score[best_pscore_id] * p.lr

    # 由预测坐标偏移得到目标中心，宽高进行滑动平均
    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, concat_score[best_pscore_id]


# 关于数据增强
# augmentation = imgaug.augmenters.Sometimes(0.5, [
#                     imgaug.augmenters.Fliplr(0.5),
#                     imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
#                 ])