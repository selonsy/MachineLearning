import numpy as np
from enum import Enum

class Machine_type(Enum):
    Windows = 1
    Linux   = 2

class Config:
    # selonsy
    MACHINE_TYPE = Machine_type.Windows         # 机器类型：WINDOWS 1, LINUX 2
    LAYER_Z_SOCRE_SIZE = np.array([32,16,8,4])  # 模板金字塔特征图大小,依次为p2,p3,p4,p5(P2为第2层,p5为顶层)
    LAYER_X_SOCRE_SIZE = np.array([68,34,17,9]) # 实例金字塔特征图大小,依次为p2,p3,p4,p5
    FEATURE_MAP_SIZE = np.array([37,19,10,6])   # 分层输出特征图大小,依次为p2,p3,p4,p5
    FEATURE_MAP_SIZES = np.array([[37,37],[19,19],[10,10],[6,6]]) 
    FPN_ANCHOR_NUM = 3                          # 由于分层预测,所以每层的尺度唯一,即anchor的数量为3
    FPN_ANCHOR_RATIOS = np.array([0.5, 1, 2])   # FPN的anchor的比例有三种
    FPN_ANCHOR_SCALES = (32, 64, 128, 256) # (32, 64, 128, 256, 512) # Length of square anchor side in pixels
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    FPN_ANCHOR_STRIDE = 1
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32] # [4, 8, 16, 32, 64]

    # dataset related
    exemplar_size = 127                    # exemplar size
    instance_size = 271                    # instance size
    context_amount = 0.5                   # context amount
    sample_type = 'uniform'

    # training related
    exem_stretch = False
    ohem_pos = False
    ohem_neg = False
    ohem_reg = False
    fix_former_3_layers = True
    pairs_per_video_per_epoch = 1          # pairs per video
    train_ratio = 0.99                     # training ratio of VID dataset
    frame_range_vid = 100                  # frame range of choosing the instance
    frame_range_ytb = 1
    train_batch_size = 8 #32                  # training batch size
    valid_batch_size = 8                   # validation batch size
    train_num_workers = 4                  # number of workers of train dataloader
    valid_num_workers = 4                  # number of workers of validation dataloader
    clip = 10                              # grad clip

    start_lr = 3e-2
    end_lr = 1e-5
    epoch = 50
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
                                           # decay rate of LR_Schedular
    step_size = 1                          # step size of LR_Schedular
    momentum = 0.9                         # momentum of SGD
    weight_decay = 0.0005                  # weight decay of optimizator

    seed = 6666                            # seed to sample training videos
    log_dir = './data/logs'                # log dirs
    max_translate = 12                     # max translation of random shift
    scale_resize = 0.15                    # scale step of instance image
    total_stride = 8                       # total stride of backbone
    valid_scope = int((instance_size - exemplar_size) / total_stride / 2)
    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    anchor_base_size = 8
    pos_threshold = 0.6
    neg_threshold = 0.3
    num_pos = 16
    num_neg = 48
    lamb = 5
    save_interval = 1
    show_interval = 5 # 100
    show_topK = 3
    pretrained_model = "" # '/mnt/usershare/zrq/pytorch/lab/model/zhangruiqi/finaltry/sharedata/alexnet.pth'

    # tracking related
    gray_ratio = 0.25
    blur_ratio = 0.15
    score_size = int((instance_size - exemplar_size) / 8 + 1)
    penalty_k = 0.22
    window_influence = 0.40
    lr_box = 0.30
    min_scale = 0.1
    max_scale = 10


config = Config()
