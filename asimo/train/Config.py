"""
Configuration for training SiamFC and tracking evaluation
Written by Heng Fan
"""
import numpy as np

class Config:
    def __init__(self):


        self.show_interval = 100 # 用于多久显示一次训练的信息
        self.anchor_scales = np.array([32, 64, 128, 256])  #  np.array([8, ]) siameseRPN; siamFPN: 32, 64, 128, 256 ,512
        self.anchor_ratios = np.array([0.5, 1, 2]) # np.array([0.33, 0.5, 1, 2, 3])
        self.anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)
        self.num_pos = 16  # 正样本的数量
        self.num_neg = 48  # 负样本的数量
        self.lamb = 100  # cls和reg的调节比率    
        self.log_dir = 'models//logs'
        # anchor_scales = (32, 64, 128, 256 ,512)
        # context_amount = 0.5  # context amount for the exemplar
        # ratios = [0.5, 1, 2] 

        # parameters for training
        self.pos_pair_range = 100
        self.num_pairs = 53200 #5.32e4  # z&x 的图片对数
        self.val_ratio = 0.1
        self.num_epoch = 1 # 训练轮次，暂定为1，原始为50（一轮是指将所有的训练数据跑一遍）
        self.batch_size = 8
        self.examplar_size = 127
        # self.instance_size = 255
        self.instance_size = 271
        self.sub_mean = 0
        self.train_num_workers = 4 # 12  # number of threads to load data when training
        self.val_num_workers = 4 # 8
        self.stride = 8
        self.rPos = 16
        self.rNeg = 0
        self.label_weight_method = "balanced"

        self.lr = 1e-2               # learning rate of SGD
        self.momentum = 0.9          # momentum of SGD
        self.weight_decay = 5e-4     # weight decay of optimizator
        self.step_size = 1           # step size of LR_Schedular
        self.gamma = 0.8685          # decay rate of LR_Schedular

        # parameters for tracking (SiamFC-3s by default)
        self.num_scale = 3
        self.scale_step = 1.0375
        self.scale_penalty = 0.9745
        self.scale_LR = 0.59
        self.response_UP = 16
        self.windowing = "cosine"
        self.w_influence = 0.176

        self.video = "Lemming"
        self.visualization = 1
        self.bbox_output = True
        self.bbox_output_path = "./tracking_result/"

        self.context_amount = 0.5
        self.scale_min = 0.2
        self.scale_max = 5
        self.score_size = int((self.instance_size - self.examplar_size) / 8 + 1) # 255/127=17,271/127=19

        # path to your trained model
        # self.net_base_path = "/home/hfan/Desktop/PyTorch-SiamFC/Train/model/"
        self.net_base_path = r"D:\workspace\vot\asimo\SiamFPN\SiamFC-PyTorch-master\Train\model"
        # path to your sequences (sequence should be in OTB format)
        # self.seq_base_path = "/home/hfan/Desktop/demo-sequences/"
        self.seq_base_path = r"D:\workspace\vot\tracker_benchmark\OTB"
        # which model to use
        self.net = "SiamFC_50_model.pth"