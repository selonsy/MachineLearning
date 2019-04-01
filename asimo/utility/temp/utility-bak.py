import numpy as np
import re
import sys
import copy
import collections
import glob

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

def compute_distance(boxA, boxB):
    """
    计算两个边界框中心点的距离
        :param boxA: 边界框A（格式：X,Y,W,H）
        :param boxB: 边界框B（格式：X,Y,W,H）
    """
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist

def compute_iou(boxA, boxB):
    """
    计算两个边界框的重叠率（x>=0 && x<1）
        :param boxA: 边界框A（格式：X,Y,W,H）
        :param boxB: 边界框B（格式：X,Y,W,H）
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

def parse_region(string):
    """
    作用：格式化区域字符串

    Split Strings with Multiple Delimiters
    使用多个分隔符来分割字符串
        :param string: 
    """
    # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
    tokens = map(float, re.split(',| |\t', string))
    tokens = list(tokens)
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    elif len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([Point(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)])
    return None

def encode_region(region):
    if isinstance(region, Polygon):
        return ','.join(['{:.2f},{:.2f}'.format(p[0], p[1]) for p in region.points])
    elif isinstance(region, Rectangle):
        return '{},{},{},{}'.format(region.x, region.y, region.width, region.height)
    else:
        return ""

def convert_region(region, to):
    """
    polygon和rectangle两种region互相转化
        :param region: 
        :param to: 
    """
    if to == 'rectangle':

        if isinstance(region, Rectangle):
            return copy.copy(region)
        elif isinstance(region, Polygon):
            top = sys.float_info.max
            bottom = sys.float_info.min
            left = sys.float_info.max
            right = sys.float_info.min

            for point in region.points:
                top = min(top, point.y)
                bottom = max(bottom, point.y)
                left = min(left, point.x)
                right = max(right, point.x)

            return Rectangle(left, top, right - left, bottom - top)

        else:
            return None
    if to == 'polygon':

        if isinstance(region, Rectangle):
            points = []
            points.append((region.x, region.y))
            points.append((region.x + region.width, region.y))
            points.append((region.x + region.width, region.y + region.height))
            points.append((region.x, region.y + region.height))
            return Polygon(points)

        elif isinstance(region, Polygon):
            return copy.copy(region)
        else:
            return None

    return None

   