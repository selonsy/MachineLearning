import numpy as np
from net.config import config
import math

############################################################
#  generate_anchors
############################################################

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

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,anchor_stride):
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
        d = np.tile(b,[3, 1, 1]) # 3 means anchors_num
        e = d.flatten()
        windows.append(e) 
    return windows

if __name__ == '__main__':
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE) # array([[256,256],[128,128],[64,64],[32,32],[16,16]])
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                backbone_shapes,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)

    