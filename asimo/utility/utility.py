'''
这个文件用来放所有的自建函数，特别复杂的可以在当前文件夹内新建并整理集合到一起
'''

import numpy as np
def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    '''
        SiamRPN的生成锚框的函数,返回(1445,4)  1445 = 17 * 17 * 5
        即累计1445个框，每个框的中心点坐标以及宽度和高度(xx,yy,w,h)
        Arguments:
                total_stride {[type]} -- [description] 8
                base_size {[type]} -- [description] 8
                scales {[type]} -- [description] [8]
                ratios {[type]} -- [description] [0.33, 0.5, 1.0, 2.0, 3.0]
                score_size {[type]} -- [description] 17
    '''
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
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

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # (5,4x225) to (225x5,4)
    ori = - (score_size // 2) * total_stride
    # the left displacement
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    # (15,15)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    # (15,15) to (225,1) to (5,225) to (225x5,1)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def generate_anchors4fpn(total_strides, anchor_scales, ratios, score_sizes):
    '''
    构造出以图像中心为原点，格式为[cx, cy, w, h]的锚点矩阵

    ratios:标签比例，暂时为3种,
    '''
    anchors=[]
    # 构造锚点数组。
    for i in range(len(score_sizes)):
        score_size = int(score_sizes[i])
        scale = total_strides[i]
        anchor_num = len(ratios) 
        
        # anchor为5*4矩阵，是因为anchor_num=5，而每个锚需要返回cx，cy，w，h共4个值。
        anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
        size = anchor_scales[i]
        count = 0
        for ratio in ratios:            
            ws = int(np.sqrt(size / ratio))
            hs = int(ws * ratio)
            # for scale in scales:    # scale=8 是因为上面的size的值为64，基准。若是FPN的话，基准的值有32,64,128,256,512五种。
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1    
        # 对锚点组进行广播，并设置其坐标。
        # 加上ori偏移后，xx和yy以图像中心为原点
        # numpy.tile(A,B)函数：重复A，B次
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size / 2) * scale
        # np.meshgrid将输入的数组进行扩展，xx为竖向扩展，yy为横向扩展。扩展的大小两个互相关。
        xx, yy = np.meshgrid([ori + scale * dx for dx in range(score_size)],
                            [ori + scale * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        # 此处的xx，yy即为生成的anchor的中心点             
        # flatten：压缩成一个一维的数组
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        anchors.append(anchor)
    return anchors # (anchor11==anchor).all() 可以用来判断两个矩阵是否所有元素都相等，any()则只要有一个相等就为真。

def compute_iou(anchors, box):
    '''
        计算锚框和基准框的交并比
    '''
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou

def box_transform(anchors, gt_box):
    '''
        回归分支，将锚框的坐标及宽高和基准框的坐标及宽高进行归一化，方便计算回归分支的Smooth L1 loss.
        返回归一化之后的(x,y,w,h)
    '''
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / anchor_w
    target_y = (gt_cy - anchor_yctr) / anchor_h
    target_w = np.log(gt_w / anchor_w)
    target_h = np.log(gt_h / anchor_h)
    regression_target = np.hstack((target_x, target_y, target_w, target_h))
    return regression_target

def compute_target(self, anchors, box):
    '''
        损失计算，返回归一化后的锚框，以及按交并比算的分类信息
    '''
    regression_target = box_transform(anchors, box)

    iou = compute_iou(anchors, box).flatten()
    # print(np.max(iou))
    pos_index = np.where(iou > config.pos_threshold)[0]
    neg_index = np.where(iou < config.neg_threshold)[0]
    label = np.ones_like(iou) * -1
    label[pos_index] = 1
    label[neg_index] = 0
    return regression_target, label