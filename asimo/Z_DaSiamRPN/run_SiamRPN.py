# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math

from utils import get_subwindow_tracking


def generate_anchor(total_stride, scales, ratios, score_size):
    '''
    构造出以图像中心为原点，格式为[cx, cy, w, h]的锚点矩阵
    '''
    # 构造锚点数组。
    # size似乎改成 Receptive Field（感受野）更好理解。scale为8，需要根据输入小心设计

    score_size = int(score_size)

    anchor_num = len(ratios) * len(scales)
    # anchor为5*4矩阵，是因为anchor_num=5，而每个锚需要返回cx，cy，w，h共4个值。
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    
    # Todo:这里暂时不清楚为什么要设置一个size的值，并用来去除下面的比率
    # 貌似是感受野，但看不懂其计算公式，也不知道拿来有什么用。
    size = total_stride * total_stride # size=64
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales: # scale=8 是因为上面的size的值为64，基准。若是FPN的话，基准的值有32,64,128,256,512五种。
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1
    # anchor
    # [[  0.   0. 104.  32.]
    #  [  0.   0.  88.  40.]
    #  [  0.   0.  64.  64.]
    #  [  0.   0.  40.  80.]
    #  [  0.   0.  32.  96.]]

    # 对锚点组进行广播，并设置其坐标。
    # 加上ori偏移后，xx和yy以图像中心为原点
    # numpy.tile(A,B)函数：重复A，B次
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    # np.meshgrid将输入的数组进行扩展，xx为竖向扩展，yy为横向扩展。扩展的大小两个互相关。
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    # 此处的xx，yy即为生成的anchor的中心点             
    # flatten：压缩成一个一维的数组
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor # (anchor11==anchor).all() 可以用来判断两个矩阵是否所有元素都相等，any()则只要有一个相等就为真。

class TrackerConfig(object):
    '''
    TrackerConfig 类定义了跟踪器参数
    '''
    # 默认的超参数
    # These are the default hyper-params for DaSiamRPN 0.3827
    # 余弦窗，惩罚大位移
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3] # 宽高比有5种
    scales = [8, ] # 尺度只有1种
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1

def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
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
    delta, score = net(x_crop) # delta:1,20,19,19     score:1,10,19,19

    # torch.Tensor.permute 置换此张量的尺寸。比如三维就有0，1，2这些dimension。
    # torch.Tensor.contiguous 返回包含与自张量相同的数据的连续张量。如果自张量是连续的，则此函数返回自张量。
    # torch.Tensor.numpy 将自张量作为 NumPy ndarray 返回。此张量和返回的 ndarray 共享相同的底层存储。自张量的变化将反映在 ndarray 中，反之亦然。

    # k=5
    # 置换delta，其形状由 N x 4k x H x W 变为4x(kx17x17)。score形状为2x(kx17x17)，并取其后一半结果
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    # 论文中的偏移公式，得到最后的bbox，此利用回归实现，前提是四个参数与gt的值相差不大。
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

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
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score 

    # window float
    # pscore按一定权值叠加一个窗分布值。找出最优得分的索引
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    # 获得目标的坐标及尺寸。delta除以scale_z映射到原图
    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    # 由预测坐标偏移得到目标中心，宽高进行滑动平均
    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net):
    """
    SiamRPN_init：SiamRPN网络初始化
        :param im: 跟踪的图片
        :param target_pos: 目标的中心点
        :param target_sz: 目标区域的宽高
        :param net: 跟踪网络
    """
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    state['im_h'] = im.shape[0] # 图片的高度
    state['im_w'] = im.shape[1] # 图片的宽度

    if p.adaptive:
        # 根据目标和输入图像的大小调整搜索区域，比例小于0.4%，需要调大搜索区域
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271
        # 根据网络总步长计算出得分图大小19*19*(2k or 4k)，记得siamRPN图结构上面写的是17*17，是因为上面的instance_size大小为255，这里改成了271.
        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1
    # generate_anchor:构造出以图像中心为原点，格式为[cx, cy, w, h]的锚点矩阵
    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

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
    net.temple(z.cuda()) # 运行 temple 函数计算模板结果

    # 两种窗
    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def SiamRPN_track(state, im):
    """
    docstring here
        :param state: 
        :param im: 
    """
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
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

    # tracker_eval 预测出新的位置和得分
    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
