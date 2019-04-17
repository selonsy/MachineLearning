"""
PyTorch implementation of SiamFC (Luca Bertinetto, et al., ECCVW, 2016)
Written by Heng Fan
"""

from utility.config import config
#from asimo.tracking import fpn
from tracking.fpn import SiamFPN50
from train.Utils import *
from train.VIDDataset import VIDDataset
from train.DataAugmentation import *
from train.Utils import *
#from train.SiamNet import *
#from tracking.fpn import SiamFPN50
from train.loss import *

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import gc
import sys
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([transforms.ToTensor()])  
unloader = transforms.ToPILImage()

# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)


def train(data_dir, train_imdb, val_imdb, model_save_path="./model/", use_gpu=True):

    ## initialize training configuration
    #config = Config()

    # do data augmentation in PyTorch;
    # you can also do complex data augmentation as in the original paper
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride
    
    # # 这里会改变x的大小为255，我们暂时屏蔽掉
    # center_crop_size = config.instance_size
    # random_crop_size = config.instance_size

    # torchvision.transforms是pytorch中的图像预处理包，一般用Compose把多个步骤整合到一起
    train_z_transforms = transforms.Compose([
        # 此处应用了数据增强，暂时屏蔽，后续考虑放开
        RandomStretch(), # RandomStretch：随机拉伸
        CenterCrop((config.examplar_size, config.examplar_size)), # CenterCrop：在图片的中间区域进行裁剪
        ToTensor() # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
                   # ToPILImage: convert a tensor to PIL image
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        #CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),  # 暂时屏蔽掉
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # load data (see details in VIDDataset.py)
    train_dataset = VIDDataset(train_imdb, data_dir, config, train_z_transforms, train_x_transforms)
    val_dataset = VIDDataset(val_imdb, data_dir, config, valid_z_transforms, valid_x_transforms, "Validation")

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.train_num_workers, drop_last=True)
    #val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
    #                       shuffle=True, num_workers=config.val_num_workers, drop_last=True)

    # create SiamFC network architecture (see details in SiamNet.py)
    # net = SiamNet()


    # create summary writer
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)

    net = SiamFPN50()
    net.init_weights()
    # move network to GPU if using GPU
    # use_gpu=False
    if use_gpu:
        net.cuda()
    if torch.cuda.device_count() > 1: # 多GPU使用
        net = nn.DataParallel(net)
    # define training strategy;
    # the learning rate of adjust layer (i.e., a conv layer)
    # is set to 0 as in the original paper
    # optimizer = torch.optim.SGD([
    #     {'params': net.feat_extraction.parameters()},
    #     {'params': net.adjust.bias},
    #     {'params': net.adjust.weight, 'lr': 0},
    # ], config.lr, config.momentum, config.weight_decay)
    
    # 这里需要优化net的所有的参数么？还是可以选择性的选取某个层？
    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # adjusting learning in each epoch：在每次迭代中调整学习率
    scheduler = StepLR(optimizer, config.step_size, config.gamma)

    # used to control generating label for training;
    # once generated, they are fixed since the labels for each
    # pair of images (examplar z and search region x) are the same
    train_response_flag = False
    valid_response_flag = False

    # ------------------------ training & validation process ------------------------
    for epoch in range(config.NUM_EPOCH):

        # adjusting learning rate
        scheduler.step()

        # ------------------------------ training ------------------------------
        # indicating training (very important for batch normalization)
        net.train()

        # used to collect loss
        train_loss = []

        # for j, data in enumerate(tqdm(train_loader)):  # 可以展示进度条，但是本机系统内存不足，容易崩
        for i, data in enumerate(train_loader):
            
            # 遍历训练集，得到：模板图片、检测图片、回归分支目标偏移量、分类分支目标分类（0/1）
            exemplar_imgs, instance_imgs, regression_target, label_target = data
            # label_target (8,1125) (8,225x5)                          
            regression_target, label_target = regression_target.cuda(), label_target.cuda()

            # 得到预测分类和预测偏移量
            pred_score, pred_regression = net.mytrain(exemplar_imgs.cuda(), instance_imgs.cuda())

            # selonsy：注意这里进行了数组的变形
            pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
            pred_offset = pred_regression.reshape(-1, 4,config.anchor_num * config.score_size * config.score_size).permute(0,2,1)

            cls_loss = rpn_cross_entropy_balance(pred_conf, label_target, config.num_pos, config.num_neg)
            reg_loss = rpn_smoothL1(pred_offset, regression_target, label_target)
            loss = cls_loss + config.lamb * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/loss', loss.data, step)
            train_loss.append(loss.detach().cpu())

            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()

            if (i + 1) % config.show_interval == 0:
                # tqdm 模块本身提供的输出信息的方法
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0

            #continue
           
            print("Epoch %d  batch %d  training loss: %f " % (epoch+1,i+1, np.mean(train_loss)))
               
        train_loss = np.mean(train_loss)

        print ("Epoch %d  training loss: %f " % (epoch+1, train_loss))        
        
        adjust_learning_rate(optimizer, 1 / config.warm_scale)

        save_name = "./models/siamrpn_warm.pth"
        new_state_dict = net.state_dict()
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

        # ------------------------------ saving model ------------------------------
        #if not os.path.exists(model_save_path):
        #    os.makedirs(model_save_path)
        #torch.save(net, model_save_path + "SiamFPN_" + str(i + 1) + "_model.pth")

        exit()
            ## fetch data, i.e., B x C x W x H (batchsize x channel x wdith x heigh)
            ## 8*3*127*127   &  8*3*271*271 
            #exemplar_imgs, instance_imgs, bbox , path_z,path_x = data  # 这个地方，需要将bbox的数据加载进来，不然后面没法计算L—reg的损失·
                            
            #batch_size = exemplar_imgs.size(0)
            #gts = bbox[0]
            #for k in range(batch_size):
            #    gt=gts[k]
            #    gt =[float(i) for i in gt.split(',')]
            #    if use_gpu:
            #        exemplar_img_single = exemplar_imgs[k].unsqueeze(0).cuda() # 1*3*127*127
            #        instance_img_single = instance_imgs[k].unsqueeze(0).cuda() # 1*3*271*271
            #    else:
            #        exemplar_img_single = exemplar_imgs[k].unsqueeze(0) # 1*3*127*127
            #        instance_img_single = instance_imgs[k].unsqueeze(0) # 1*3*271*271
                
            #    # image_z = unloader(exemplar_img_single.squeeze(0))
            #    # image_x = unloader(instance_img_single.squeeze(0))                
            #    # imz = plt.imshow(image_z)
            #    # imx = plt.imshow(image_x)
            #    # rect = plt.Rectangle((gt[0], gt[1]), gt[0] + gt[2],gt[1] + gt[3], linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)                     
            #    # plt.gca().add_patch(rect)                                    
            #    # plt.show()
                
            #    # output = net.train(Variable(exemplar_imgs), Variable(instance_imgs))
            #    deltas,scores = net.mytrain(Variable(exemplar_img_single), Variable(instance_img_single))
            #    del exemplar_img_single,instance_img_single
            
            #    # clear the gradient
            #    optimizer.zero_grad()

            #    # loss
            #    loss = net.weight_loss(deltas,scores,gt,use_gpu)
            #    del deltas, scores 
               
            #    # backward
            #    loss.backward()     # error: element 0 of tensors does not require grad and does not have a grad_fn

            #    # update parameter
            #    optimizer.step()

            #    # collect training loss
            #    train_loss.append(float(loss.data))
                
            #    # print("Epoch %d  batch %d  image %d   training loss: %f " % (i+1,j+1,k+1, loss.data))

            ## del data, exemplar_imgs, instance_imgs, bbox, path_z,path_x 
            ## gc.collect()
         
        # ------------------------------ validation ------------------------------
        # indicate validation
        net.eval()

        # used to collect validation loss
        val_loss = []

        # for j, data in enumerate(tqdm(val_loader)):
        for j, data in enumerate(val_loader):

            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            # create label for validation (only do it one time)
            if not valid_response_flag:
                valid_response_flag = True
                response_size = output.shape[2:4]
                valid_eltwise_label, valid_instance_weight = create_label(response_size, config, use_gpu)

            # loss
            loss = net.weight_loss(output, valid_eltwise_label, valid_instance_weight)

            # collect validation loss
            val_loss.append(loss.data)

        print ("Epoch %d   training loss: %f, validation loss: %f" % (i+1, np.mean(train_loss), np.mean(val_loss)))
