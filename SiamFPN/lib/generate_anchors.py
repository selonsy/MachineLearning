import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from net.config import config
from lib.utils import show_anchors
import math
from IPython import embed
############################################################
#  generate_anchors
############################################################


def generate_anchors_rpn(total_stride, base_size, scales, ratios, score_size):
    '''SiamRPN的锚标签生成

        @total_stride, int: 8 # total stride of backbone

        @base_size, int: 8 

        @scales, array: [8]

        @ratios, array: [0.33, 0.5, 1, 2, 3] 
        注：按论文的理解应该是宽高比，即w/h，但是下面的计算是按照高宽比进行的

        @score_size, int: 19

        return anchors -> (1805,4)   
        1805 = len(ratios) * len(scales) * score_size * score_size
             = 5 * 1 * 19 * 19     
    '''
    anchor_num = len(ratios) * len(scales) # 锚种类数
    anchor = np.zeros((anchor_num, 4), dtype=np.float32) # (5,4)
    size = base_size * base_size # 锚的大小（映射回原图的基准大小，比例只是在此基础上进行变化，宽高比变化不影响锚的大小）
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4)) # 重复anchor共score_size**2次并变形为(1805,4)    
    ori = - (score_size // 2) * total_stride # -72
    # the left displacement 
    # [ori + total_stride * dx for dx in range(score_size)] = [-72,-64,...,64,72] , len = 19
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])

    # xx (1805,1) yy (1805,1)
    xx = np.tile(xx.flatten(), (anchor_num, 1)).flatten()
    yy = np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    
    # 给锚的(x,y)赋值
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)    

    return anchor


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    # 保证面积不变，只是宽高的比例有变化
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    # 压缩成(N,4)的形式
    # return np.concatenate(anchors, axis=0)
    return anchors


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])


def generate_track_windows():
    windows = []
    # self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
    #                           [config.anchor_num, 1, 1]).flatten()

    # a = np.hanning(config.score_size) # shape:(19,)
    # b = np.outer(a, a) # (19, 19)
    # '''outer
    #     ①对于多维向量，全部展开变为一维向量
    #     ②第一个参数表示倍数，使得第二个向量每次变为几倍。
    #     ③第一个参数确定结果的行，第二个参数确定结果的列。
    #     import numpy as np
    #     x1 = [1,2,3]
    #     x2 = [4,5,6]
    #     outer = np.outer(x1,x2)
    #     print outer

    #     输出:
    #     [[ 4  5  6]       #1倍
    #     [ 8 10 12]        #2倍
    #     [12 15 18]]       #3倍
    # '''
    # c = b[None, :] # (1, 19, 19)
    # print(c.ndim) # 3
    # print(np.array([config.anchor_num, 1, 1]).ndim) # 1
    # d = np.tile(c,[config.anchor_num, 1, 1]) # (5, 19, 19)
    # e = d.flatten() # (1805,)

    for size in config.FEATURE_MAP_SIZE:
        a = np.hanning(size)
        b = np.outer(a, a)
        c = b[None, :]
        d = np.tile(b, [3, 1, 1])  # 3 means anchors_num
        e = d.flatten()
        windows.append(e)
    return windows


def generate_anchors_fpn(total_stride, base_size, scales, ratios, score_size):
    '''SiamRPN的锚标签生成

        @total_stride, int: 8 # total stride of backbone

        @base_size, int: 8 

        @scales, array: [8]

        @ratios, array: [0.33, 0.5, 1, 2, 3] 
        注：按论文的理解应该是宽高比，即w/h，但是下面的计算是按照高宽比进行的

        @score_size, int: 19

        return anchors -> (1805,4)   
        1805 = len(ratios) * len(scales) * score_size * score_size
             = 5 * 1 * 19 * 19     
    '''
    anchor_num = len(ratios) * len(scales) # 锚种类数
    anchor = np.zeros((anchor_num, 4), dtype=np.float32) # (5,4)
    size = base_size * base_size # 锚的大小（映射回原图的基准大小，比例只是在此基础上进行变化，宽高比变化不影响锚的大小）
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4)) # 重复anchor共score_size**2次并变形为(1805,4)    
    ori = - (score_size // 2) * total_stride # -72
    # the left displacement 
    # [ori + total_stride * dx for dx in range(score_size)] = [-72,-64,...,64,72] , len = 19
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])

    # xx (1805,1) yy (1805,1)
    xx = np.tile(xx.flatten(), (anchor_num, 1)).flatten()
    yy = np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    
    # 给锚的(x,y)赋值
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)    

    return anchor


if __name__ == '__main__':
    # Anchors
    # 原计算单层锚标签

    # ToDo:anchor在原图上面的显示聚集于左上角的一坨，没有在整个图上面依次展开
    valid_scope = 2 * config.valid_scope + 1
    anchors_ori = generate_anchors_rpn(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                    # config.anchor_ratios,
                                    config.FPN_ANCHOR_RATIOS,
                                    valid_scope)
    anchors = generate_pyramid_anchors(config.FPN_ANCHOR_SCALES,
                                            config.FPN_ANCHOR_RATIOS,
                                            config.FEATURE_MAP_SIZES,
                                            config.BACKBONE_STRIDES,
                                            config.FPN_ANCHOR_STRIDE) 
    def compare_anchor(old,new):
        for i in range(old.shape[0]):
            if not (old[i]==new[i]).all():
                return False
        return True
    is_compare = compare_anchor(anchors_ori,anchors[1])
    # show_anchors(anchors_ori[:10,:])
    # show_anchors(anchors[1])
    pass
