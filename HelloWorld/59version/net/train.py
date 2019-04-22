import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import cv2
import pickle
import lmdb
import torch.nn as nn
import time

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from collections import OrderedDict

from net.config import config
from net.network import SiameseAlexNet
from net.dataset import ImagnetVIDDataset
from lib.custom_transforms import Normalize, ToTensor, RandomStretch, \
    RandomCrop, CenterCrop, RandomBlur, ColorAug
from lib.loss import rpn_smoothL1, rpn_cross_entropy_balance
from lib.visual import visual
from lib.utils import get_topk_box, add_box_img, compute_iou, box_transform_inv, adjust_learning_rate

from IPython import embed
from net.fpn import SiamFPN50,SiamFPN101,SiamFPN152

torch.manual_seed(config.seed)


def train(data_dir, model_path=None, vis_port=None, init=None):
    # loading meta data
    # -----------------------------------------------------------------------------------------------------
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    # split train/valid dataset
    # -----------------------------------------------------------------------------------------------------
    train_videos, valid_videos = train_test_split(all_videos,
                                                  test_size=1 - config.train_ratio, random_state=config.seed)

    # define transforms
    train_z_transforms = transforms.Compose([
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # open lmdb
    # db = lmdb.open(data_dir + '_lmdb', readonly=True, map_size=int(1024*1024*1024)) # 200e9,单位Byte
    db_path = data_dir + '_lmdb'
    # create dataset
    # -----------------------------------------------------------------------------------------------------
    train_dataset = ImagnetVIDDataset(db_path, train_videos, data_dir, train_z_transforms, train_x_transforms)
    # test __getitem__
    # train_dataset.__getitem__(1)
    # exit(0)

    anchors = train_dataset.anchors  # (1805,4) = (19*19*5,4)
    # dic_num = {}
    # ind_random = list(range(len(train_dataset)))
    # import random
    # random.shuffle(ind_random)
    # for i in tqdm(ind_random):
    #     exemplar_img, instance_img, regression_target, conf_target = train_dataset[i+1000]

    valid_dataset = ImagnetVIDDataset(db_path, valid_videos, data_dir, valid_z_transforms, valid_x_transforms, training=False)
    # create dataloader   
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, 
                                pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False, 
                                pin_memory=True, num_workers=config.valid_num_workers, drop_last=True)
   
    # create summary writer
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)
    if vis_port:
        vis = visual(port=vis_port)

    # start training
    # -----------------------------------------------------------------------------------------------------
    # model = SiameseAlexNet()
    model = SiamFPN50()
    model.init_weights() # 权重初始化
    if config.CUDA:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    # load model weight
    # -----------------------------------------------------------------------------------------------------
    start_epoch = 1
    if model_path and init:
        print("init training with checkpoint %s" % model_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model_dict = model.state_dict()
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print("inited checkpoint")
    elif model_path and not init:
        print("loading checkpoint %s" % model_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
        print("loaded checkpoint")
    elif not model_path and config.pretrained_model:
        print("init with pretrained checkpoint %s" %
              config.pretrained_model + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(config.pretrained_model)
        # change name and load parameters
        checkpoint = {k.replace(
            'features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

    #  layers
    def freeze_layers(model):
        print('------------------------------------------------------------------------------------------------')
        for layer in model.featureExtract[:10]:
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.Conv2d):
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.MaxPool2d):
                continue
            elif isinstance(layer, nn.ReLU):
                continue
            else:
                raise KeyError('error in fixing former 3 layers')
        # print("fixed layers:")
        # print(model.featureExtract[:10])
        '''
        fixed layers:
        Sequential(
        (0): Conv2d(3, 96, kernel_size=(11, 11), stride=(2, 2))
        (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): ReLU(inplace)
        (4): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1))
        (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (7): ReLU(inplace)
        (8): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1))
        (9): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        '''

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(start_epoch, config.EPOCH + 1):
        train_loss = []
        model.train()
        if config.fix_former_3_layers:  # 暂时去掉 # 固定前三层卷积的v.requires_grad = False
            if torch.cuda.device_count() > 1:
                freeze_layers(model.module)
            else:
                freeze_layers(model)
        loss_temp_cls = 0
        loss_temp_reg = 0
        # for i, data in enumerate(tqdm(trainloader)): # can't pickle Transaction objects
        for k, data in enumerate(trainloader): # 这里有问题,loader没有遍历完就跳走了
            # print("done")
            # return
            # (8,3,127,127)\(8,3,271,271)\(8,1805,4)\(8,1805)
            # 8为batch_size,1445 = 19 * 19 * 5,5 = anchors_num
            # exemplar_imgs, instance_imgs, regression_target, conf_target = data
            exemplar_imgs, instance_imgs, regression_targets, conf_targets = data

            # conf_target (8,1125) (8,225x5)
            if config.CUDA:
                # 这里有问题,regression_targets是list,不能直接使用.cuda(),后面考虑将其压缩成(N,4)的形式
                # regression_targets, conf_targets = torch.tensor(regression_targets).cuda(), torch.tensor(conf_targets).cuda()
                exemplar_imgs, instance_imgs = exemplar_imgs.cuda(), instance_imgs.cuda()
            
            # # 基于一层的损失计算
            # # (8,10,19,19)\(8,20,19,19)
            # pred_score, pred_regression = model(exemplar_imgs, instance_imgs)
            # # (8,1805,2)
            # pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
            # # (8,1805,4)
            # pred_offset = pred_regression.reshape(-1, 4,config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
            
            # cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
            #                                      ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            # reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
            # loss = cls_loss + config.lamb * reg_loss
            # 基于金字塔模型的损失计算
            pred_scores, pred_regressions = model.mytrain(exemplar_imgs, instance_imgs)
            # FEATURE_MAP_SIZE、FPN_ANCHOR_NUM
            '''
            when batch_size = 2, anchor_num = 3
            torch.Size([N, 6, 37, 37])
            torch.Size([N, 6, 19, 19])
            torch.Size([N, 6, 10, 10])
            torch.Size([N, 6, 6, 6])

            torch.Size([N, 12, 37, 37])
            torch.Size([N, 12, 19, 19])
            torch.Size([N, 12, 10, 10])
            torch.Size([N, 12, 6, 6])
            '''
            loss = 0
            cls_loss_sum = 0
            reg_loss_sum = 0
            for i in range(len(pred_scores)):
                pred_score = pred_scores[i]
                pred_regression = pred_regressions[i]
                anchors_num = config.FPN_ANCHOR_NUM * config.FEATURE_MAP_SIZE[i] * config.FEATURE_MAP_SIZE[i]
                pred_conf = pred_score.reshape(-1, 2, anchors_num).permute(0,2,1)                
                pred_offset = pred_regression.reshape(-1, 4, anchors_num).permute(0,2,1)  

                conf_target = conf_targets[i]
                regression_target = regression_targets[i].type(torch.FloatTensor) # pred_offset是float类型
                if config.CUDA:
                    conf_target = conf_target.cuda()
                    regression_target = regression_target.cuda()
                # 二分类损失计算(交叉熵)
                cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors[i],
                                                    ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
                # 回归损失计算(Smooth L1) # 这里应该有问题,回归损失的值为0                
                reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)

                _loss = cls_loss + config.lamb * reg_loss
                loss += _loss # 这里四层的loss先直接加起来,后面考虑加权处理

                # 用于tensorboard展示
                cls_loss_sum = cls_loss_sum + cls_loss
                reg_loss_sum = reg_loss_sum + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()

            step = (epoch - 1) * len(trainloader) + k
            summary_writer.add_scalar('train/cls_loss', cls_loss_sum.data, step)
            summary_writer.add_scalar('train/reg_loss', reg_loss_sum.data, step)
            loss = loss.detach().cpu()
            train_loss.append(loss)
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()
            # if vis_port:
            #     vis.plot_error({'rpn_cls_loss': cls_loss.detach().cpu().numpy().ravel()[0],
            #                     'rpn_regress_loss': reg_loss.detach().cpu().numpy().ravel()[0]}, win=0)

            print("Epoch %d  batch %d  training loss: %f " % (epoch,k+1, loss))

            if (k + 1) % config.show_interval == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, k, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
                # 视觉展示
                if vis_port:
                    anchors_show = train_dataset.anchors
                    exem_img = exemplar_imgs[0].cpu(
                    ).numpy().transpose(1, 2, 0)
                    inst_img = instance_imgs[0].cpu(
                    ).numpy().transpose(1, 2, 0)

                    # show detected box with max score
                    topk = config.show_topK
                    vis.plot_img(exem_img.transpose(
                        2, 0, 1), win=1, name='exemple')
                    cls_pred = conf_target[0]
                    gt_box = get_topk_box(
                        cls_pred, regression_target[0], anchors_show)[0]

                    # show gt_box
                    img_box = add_box_img(inst_img, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1),
                                 win=2, name='instance')

                    # show anchor with max score
                    cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                    scores, index = torch.topk(cls_pred, k=topk)
                    img_box = add_box_img(inst_img, anchors_show[index.cpu()])
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1),
                                 win=3, name='anchor_max_score')

                    cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                    topk_box = get_topk_box(
                        cls_pred, pred_offset[0], anchors_show, topk=topk)
                    img_box = add_box_img(inst_img, topk_box)
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1),
                                 win=4, name='box_max_score')

                    # show anchor and detected box with max iou
                    iou = compute_iou(anchors_show, gt_box).flatten()
                    index = np.argsort(iou)[-topk:]
                    img_box = add_box_img(inst_img, anchors_show[index])
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1),
                                 win=5, name='anchor_max_iou')

                    # detected box
                    regress_offset = pred_offset[0].cpu().detach().numpy()
                    topk_offset = regress_offset[index, :]
                    anchors_det = anchors_show[index, :]
                    pred_box = box_transform_inv(anchors_det, topk_offset)
                    img_box = add_box_img(inst_img, pred_box)
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1),
                                 win=6, name='box_max_iou')
        
        train_loss = np.mean(train_loss)
        # print("done")
        # exit(0)
        # 验证
        valid_loss = []
        model.eval()
        # for i, data in enumerate(tqdm(validloader)):
        for i, data in enumerate(validloader):
            exemplar_imgs, instance_imgs, regression_targets, conf_targets = data            
            if config.CUDA:                
                exemplar_imgs, instance_imgs = exemplar_imgs.cuda(), instance_imgs.cuda()
            
            pred_scores, pred_regressions = model.mytrain(exemplar_imgs, instance_imgs)
            loss = 0
            for i in range(len(pred_scores)):
                pred_score = pred_scores[i]
                pred_regression = pred_regressions[i]
                anchors_num = config.FPN_ANCHOR_NUM * config.FEATURE_MAP_SIZE[i] * config.FEATURE_MAP_SIZE[i]
                pred_conf = pred_score.reshape(-1, 2, anchors_num).permute(0,2,1)                
                pred_offset = pred_regression.reshape(-1, 4, anchors_num).permute(0,2,1)  

                conf_target = conf_targets[i]
                regression_target = regression_targets[i].type(torch.FloatTensor) # pred_offset是float类型
                if config.CUDA:
                    conf_target = conf_target.cuda()
                    regression_target = regression_target.cuda()
                # 二分类损失计算(交叉熵)
                cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors[i],
                                                    ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
                # 回归损失计算(Smooth L1) # 这里应该有问题,回归损失的值为0                
                reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
            
                _loss = cls_loss + config.lamb * reg_loss
                loss += _loss # 这里四层的loss先直接加起来,后面考虑加权处理            
            valid_loss.append(loss.detach().cpu())
        valid_loss = np.mean(valid_loss)
        
        print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" %(epoch, valid_loss, train_loss))
        summary_writer.add_scalar('valid/loss',valid_loss, (epoch + 1) * len(trainloader))
        # 调整学习率
        adjust_learning_rate(optimizer,config.gamma)  # adjust before save, and it will be epoch+1's lr when next load
        # 保存训练好的模型
        if epoch % config.save_interval == 0:
            if not os.path.exists('./data/models/'):
                os.makedirs("./data/models/")
            
            save_name = "./data/models/siamrpn_{}_trainloss_{:.4f}_validloss_{:.4f}.pth".format(epoch,train_loss,valid_loss)
            new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1:
                new_state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    namekey = k[7:]  # remove `module.`
                    new_state_dict[namekey] = v
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model: {}'.format(save_name))


if __name__ == "__main__":
    data_dir = r"D:\workspace\MachineLearning\HelloWorld\59version\dataset\ILSVRC_Crops"
    model_path = ""
    vis_port = None
    init = None
    train(data_dir, model_path, vis_port, init)
