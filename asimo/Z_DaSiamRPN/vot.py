"""
\file vot.py

@brief Python utility functions for VOT integration

@author Luka Cehovin, Alessio Dore

@date 2016

"""

import sys
import copy
import collections
import glob
import re
from os.path import realpath, dirname, join
from config import *

try:
    import trax
    import trax.server
    TRAX = True
except ImportError:
    TRAX = False

# collections模块的namedtuple子类不仅可以使用item的index访问item，还可以通过item的name进行访问
# 可以理解为C语言中的struct
Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])


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

# 仅供python版本的benchmarks使用


def vot_encode_region(region):
    return '{} {} {} {}'.format(region.x, region.y, region.width, region.height)


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


class VOT(object):
    """ Base class for Python VOT integration """

    def __init__(self, region_format, bench_mark_type='', seq_list_path='', seq_startFrame='', seq_endFrame='', seq_imgFormat='', ground_truth_txt='', seq_name=''):
        """ Constructor

        Args:
            region_format: Region format options
        """
        # assert(condition):用来让程序测试这个condition，如果condition为false，那么raise一个AssertionError出来
        assert(region_format in ['rectangle', 'polygon'])
        if TRAX:
            options = trax.server.ServerOptions(region_format, trax.image.PATH)
            self._trax = trax.server.Server(options)

            request = self._trax.wait()
            assert(request.type == 'initialize')
            if request.region.type == 'polygon':
                self._region = Polygon([Point(x[0], x[1])
                                        for x in request.region.points])
            else:
                self._region = Rectangle(
                    request.region.x, request.region.y, request.region.width, request.region.height)
            self._image = str(request.image)
            self._trax.status(request.region)
        elif bench_mark_type == 'python':

            # Todo: 需要适配Linux，暂时先不管

            # seq_list_path='',seq_startFrame='',seq_endFrame='',seq_imgFormat='',ground_truth_txt='',seq_name=''
            # ['David', '300', '770', '[129, 80, 64, 78]', '{0:04d}.jpg', 'David_0']
            # selonsy:为tracker_benchmarks_python版本所增设的接口
            _files = sorted(glob.glob(seq_list_path+'\*.jpg'))
            startIndex = _files.index(
                seq_list_path+'\\'+seq_imgFormat.format(int(seq_startFrame)))
            endIndex = _files.index(
                seq_list_path+'\\'+seq_imgFormat.format(int(seq_endFrame)))
            _files = _files[startIndex:endIndex+1]
            # print(_files)
            self._files = _files
            self._frame = 0
            self._seq_name = seq_name
            ground_truth_txt = ground_truth_txt.strip(
                '[').strip(']').replace(' ', '')
            self._region = convert_region(
                parse_region(ground_truth_txt), region_format)
            print(self._region)
            self._result = []
        elif bench_mark_type == 'matlab':
            # matlab 版本
            # python3 D:\workspace\vot\asimo\other\DaSiamRPN\code\my_Tracker.py
            # matlab
            # D:\workspace\vot\tracker_benchmark_python\data\Basketball\img\
            # 1
            # 725
            # [197,213,34,81]
            # 4
            # basketball_1
            # selonsy:为tracker_benchmarks_matlab版本所增设的接口
            # seq_imgFormat = '{0:0{0}d}.jpg'.format(int(seq_imgFormat))
            seq_imgFormat = '{0:0%(num)dd}.jpg' % {'num': int(seq_imgFormat)}
            if system_type==1:
                _files = sorted(glob.glob(seq_list_path+'/*.jpg'))
            else:
                _files = sorted(glob.glob(seq_list_path+'\*.jpg'))            
            startIndex = _files.index(
                seq_list_path+seq_imgFormat.format(int(seq_startFrame)))
            endIndex = _files.index(
                seq_list_path+seq_imgFormat.format(int(seq_endFrame)))
            _files = _files[startIndex:endIndex+1]
            # print(_files)
            self._files = _files
            self._frame = 0
            self._seq_name = seq_name
            ground_truth_txt = ground_truth_txt.strip(
                '[').strip(']').replace(' ', '')
            self._region = convert_region(
                parse_region(ground_truth_txt), region_format)
            # print(self._region)
            self._result = []
        else:
            # 如果没有 trax，从’images.txt’中逐行读取图像路径
            # strip：用于移除字符串头尾指定的字符(默认为空格或换行符)或字符序列
            self._files = [x.strip('\n')
                           for x in open('images.txt', 'r').readlines()]
            self._frame = 0
            # 从’region.txt’中读取第一行区域坐标,parse_region:根据输入字符串解析出坐标
            self._region = convert_region(parse_region(
                open('region.txt', 'r').readline()), region_format)
            self._result = []

    def region(self):  # 返回初始化区域，即跟踪第一帧的Ground_Truth
        """
        Send configuration message to the client and receive the initialization
        region and the path of the first image

        Returns:
            initialization region
        """

        return self._region

    def report(self, region, confidence=0):  # 记录结果
        """
        Report the tracking results to the client

        Arguments:
            region: region for the frame
        """
        assert(isinstance(region, Rectangle) or isinstance(region, Polygon))
        if TRAX:
            if isinstance(region, Polygon):
                tregion = trax.region.Polygon(
                    [(x.x, x.y) for x in region.points])
            else:
                tregion = trax.region.Rectangle(
                    region.x, region.y, region.width, region.height)
            self._trax.status(tregion, {"confidence": confidence})
        else:
            self._result.append(region)
            self._frame += 1

    def frame(self):  # 如果帧索引超出文件列表返回None，否则返回对应帧文件
        """
        Get a frame (image path) from client

        Returns:
            absolute path of the image
        """
        if TRAX:
            if hasattr(self, "_image"):
                image = str(self._image)
                del self._image
                return image

            request = self._trax.wait()

            if request.type == 'frame':
                return str(request.image)
            else:
                return None

        else:
            if self._frame >= len(self._files):
                return None
            return self._files[self._frame]

    def quit(self):  # 退出
        if TRAX:
            self._trax.quit()
        elif hasattr(self, '_result'):
            # with open('output.txt', 'w',encoding='utf-8') as f:
            with open(join(result_path, 'DaSiamRPN_{0}.txt'.format(self._seq_name)), 'w', encoding='utf-8') as f:
                for r in self._result:
                    f.write(vot_encode_region(convert_region(r, 'rectangle')))
                    f.write('\n')

    def result(self):
        return self._result

    def frames(self):
        return self._frame

    def __del__(self):  # 删除一个对象时，python解释器默认调用__del__,也可以手动的删除对象，即: del object
        self.quit()    # 自动调用的话，断点进不来
