'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
# from .run_SiamFPN import generate_anchors4fpn, TrackerConfig4FPN
import gc
from config import config

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        # cuz upsample is deprecated.
        # return F.upsample(x, size=(H,W), mode='bilinear') + y
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up x(1,3,127,127)
        c1 = F.relu(self.bn1(self.conv1(x))) # c1(1,64,64,64)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) # c1(1,64,32,32)
        c2 = self.layer1(c1) # c2(1,256,32,32)
        c3 = self.layer2(c2) # c3([1, 256, 32, 32])
        c4 = self.layer3(c3) # c4([1, 1024, 8, 8])
        c5 = self.layer4(c4) # c5([1, 2048, 4, 4])
        # Top-down
        p5 = self.toplayer(c5) # p5([1, 256, 4, 4])
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # p4([1, 256, 8, 8])
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # p3([1, 256, 16, 16])
        p2 = self._upsample_add(p3, self.latlayer3(c2)) # p2([1, 256, 32, 32])
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        # ToDo：是否缺少P6层
        return p2, p3, p4, p5

class SiamFPN(FPN):
    def __init__(self, block, feature_out=256, anchor_num=3):
        super(SiamFPN, self).__init__(Bottleneck, block)

        self.anchor_num = anchor_num
        self.feature_out = feature_out

        self.conv_reg1 = nn.Conv2d(256, feature_out*4*anchor_num, 3)      # 模板分支的回归子分支
        self.conv_reg2 = nn.Conv2d(256, feature_out, 3)                   # 检测分支的回归子分支
        self.conv_cls1 = nn.Conv2d(256, feature_out*2*anchor_num, 3)      # 模板分支的分类子分支
        self.conv_cls2 = nn.Conv2d(256, feature_out, 3)                   # 检测分支的分类子分支
        self.regress_adjust = nn.Conv2d(4*anchor_num, 4*anchor_num, 1)    # 1x1卷积

        self.reg_kernel = []
        self.cls_kernel = []        

    def forward_bak(self, x):  # x:1,3,271,271
        # px2, px3, px4, px5 = super().forward(x)
        px = super().forward(x)
        deltas = []
        scores = []
        bnn12 = nn.BatchNorm2d(12)  # .cuda()
        bnn6 = nn.BatchNorm2d(6)  # .cuda()
        for p_index in range(len(px)):
            # print(px[p_index].size())
            # torch.Size([1, 256, 68, 68])
            # torch.Size([1, 256, 34, 34])
            # torch.Size([1, 256, 17, 17])
            # torch.Size([1, 256, 9, 9])
            delta = self.regress_adjust(F.conv2d(self.conv_reg2(px[p_index]), self.reg_kernel[p_index]))
            delta = bnn12(delta)
            # print(delta.size())
            # torch.Size([1, 12, 37, 37])
            # torch.Size([1, 12, 19, 19])
            # torch.Size([1, 12, 10, 10])
            # torch.Size([1, 12, 6, 6])
            socre = F.conv2d(self.conv_cls2(px[p_index]), self.cls_kernel[p_index])
            socre = bnn6(socre)
            # print(socre.size())
            # torch.Size([1, 6, 37, 37])
            # torch.Size([1, 6, 19, 19])
            # torch.Size([1, 6, 10, 10])
            # torch.Size([1, 6, 6, 6])
            deltas.append(delta)
            scores.append(socre)
            del delta, socre
            gc.collect()
        return deltas, scores

        # x_f = self.featureExtract(x)
        # return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
        #        F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def template(self, z):  # z:1,3,127,127
        # pz2, pz3, pz4, pz5 = super().forward(z)
        pz = super().forward(z)
        for p in pz:
            # print(p.size())
            # torch.Size([1, 256, 32, 32])
            # torch.Size([1, 256, 16, 16])
            # torch.Size([1, 256, 8, 8])
            # torch.Size([1, 256, 4, 4])
            bn = nn.BatchNorm2d(256)
            reg_kernel_raw = self.conv_reg1(p)
            # print(reg_kernel_raw.size())
            # torch.Size([1, 6144, 30, 30])
            # torch.Size([1, 6144, 14, 14])
            # torch.Size([1, 6144, 6, 6])
            # torch.Size([1, 6144, 2, 2])
            cls_kernel_raw = self.conv_cls1(p)
            # print(cls_kernel_raw.size())
            # torch.Size([1, 3072, 30, 30])
            # torch.Size([1, 3072, 14, 14])
            # torch.Size([1, 3072, 6, 6])
            # torch.Size([1, 3072, 2, 2])
            kernel_size = reg_kernel_raw.data.size()[-1]   # 模板宽度
            # print(kernel_size)
            # 30
            # 14
            # 6
            # 2
            reg_kernel_view = reg_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
            # print(reg_kernel_view.size())
            # torch.Size([12, 512, 30, 30])
            # torch.Size([12, 512, 14, 14])
            # torch.Size([12, 512, 6, 6])
            # torch.Size([12, 512, 2, 2])
            cls_kernel_view = cls_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)
            # print(cls_kernel_view.size())
            # torch.Size([6, 512, 30, 30])
            # torch.Size([6, 512, 14, 14])
            # torch.Size([6, 512, 6, 6])
            # torch.Size([6, 512, 2, 2])
            bnn = nn.BatchNorm2d(256)  # .cuda()
            reg_kernel_view = bnn(reg_kernel_view)
            cls_kernel_view = bnn(cls_kernel_view)
            self.reg_kernel.append(reg_kernel_view)
            self.cls_kernel.append(cls_kernel_view)

            del p, reg_kernel_view, cls_kernel_view, reg_kernel_raw, cls_kernel_raw
        del pz
        gc.collect()
        # z_f = self.featureExtract(z)            # 调用featureExtract提取模板z的特征,1,256,6,6
        # r1_kernel_raw = self.conv_r1(z_f)       # 由z_f得到坐标回归量r1_kernel_raw,1,5120,4,4
        # cls1_kernel_raw = self.conv_cls1(z_f)   # 由z_f得到分类得分cls1_kernel_raw,1,2560,4,4
        # kernel_size = r1_kernel_raw.data.size()[-1] # 模板宽度
        # # torch.Tensor.view 返回一个新的张量，其数据与自身张量相同，但大小不同
        # # r1_kernel_raw原始的形状：[feature_out∗4∗anchor,kernel_size,kernel_size]
        # self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)   # 20,256,4,4
        # self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size) # 10,256,4,4

    def weight_loss(self, deltas, scores, gt, use_gpu):
        """
        weighted cross entropy loss


        卧槽   才发现，貌似训练的时候不需要进行偏移量的运算？？尼玛~
        """
        # return F.binary_cross_entropy_with_logits(prediction,label,weight,size_average=False) / self.config.batch_size

        assert len(deltas) == len(scores)
        concat_delta = []
        concat_score = []

        p = TrackerConfig4FPN()
        p.anchors = generate_anchors4fpn(
            p.total_strides, p.anchor_scales, p.ratios, p.score_sizes)

        for i in range(len(deltas)):
            # 置换delta，其形状由 N x 4k x H x W 变为4x(kx17x17)。score形状为2x(kx17x17)，并取其后一半结果
            delta = deltas[i].permute(1, 2, 3, 0).contiguous().view(
                4, -1).data.cpu().numpy()
            score = F.softmax(scores[i].permute(1, 2, 3, 0).contiguous().view(
                2, -1), dim=0).data[1, :].cpu().numpy()

            # 论文中的偏移公式，得到最后的bbox，此利用回归实现，前提是四个参数与gt的值相差不大。
            delta[0, :] = delta[0, :] * p.anchors[i][:, 2] + p.anchors[i][:, 0]
            delta[1, :] = delta[1, :] * p.anchors[i][:, 3] + p.anchors[i][:, 1]
            delta[2, :] = np.exp(delta[2, :]) * p.anchors[i][:, 2]
            delta[3, :] = np.exp(delta[3, :]) * p.anchors[i][:, 3]

            if len(concat_delta) == 0:
                concat_delta = delta
                concat_score = score
            else:
                concat_delta = np.concatenate((concat_delta, delta), axis=1)
                concat_score = np.concatenate((concat_score, score), axis=0)

        rpn_loss_cls = 0
        rpn_loss_box = 0
        lambda4balance = 10

        # 接下来进行NMS，iou比较0.7 & 0.3

        def get_labels(scores):
            labels = []
            new_socres = []
            filter_indexs = []
            m = 0
            for score in scores:
                if score > 0.7:
                    labels.append(1)
                    _s = [score, 1]
                    new_socres.append(_s)
                    filter_indexs.append(m)
                elif score < 0.3:
                    labels.append(0)
                    _s = [score, 0]
                    new_socres.append(_s)
                    filter_indexs.append(m)
                m = m+1
            labels = torch.LongTensor(labels)
            filter_indexs = torch.LongTensor(filter_indexs)
            new_socres = torch.Tensor(new_socres)
            if use_gpu:
                return new_socres.cuda(), labels.cuda(), filter_indexs.cuda()
            else:
                return new_socres, labels, filter_indexs

        new_socres, new_labels, filter_indexs = get_labels(concat_score)
        new_socres = Variable(new_socres, requires_grad=True)
        # new_labels = Variable(new_labels,requires_grad=True)
        rpn_loss_cls = F.cross_entropy(new_socres, new_labels)

        # 考虑将deltas映射回原图再求loss
        bbox_size = filter_indexs.shape[0]  # concat_delta.shape[1]
        new_deltas = np.transpose(concat_delta)
        new_deltas = torch.from_numpy(new_deltas)
        new_deltas = torch.index_select(new_deltas, 0, filter_indexs)

        # gt_value = [1,2,100,60]
        targets = np.tile(gt, bbox_size).reshape((-1, 4))
        targets = torch.from_numpy(targets).float()

        if use_gpu:
            new_deltas = new_deltas.cuda()
            targets = targets.cuda()

        new_deltas = Variable(new_deltas, requires_grad=True)
        # targets = Variable(targets,requires_grad=True)
        rpn_loss_box = F.smooth_l1_loss(
            new_deltas, targets)/new_deltas.shape[0]
        del deltas, scores, new_deltas, targets, new_socres
        gc.collect()
        return rpn_loss_cls + lambda4balance * rpn_loss_box

        # L_cls分类损失，p为anchor预测为目标的概率（得分），p_star为（0，negtive label，1，positive label）
        def L_cls(p, p_star):
            return -math.log(p_star*p+(1-p_star)*(1-p))

        def L_reg(t_x, t_y, t_w, t_h):
            a_x, a_y, a_w, a_h = 1, 1, 100, 60  # 假设，因为暂时没有值传递过来，其实取的是gt的值，即标准bbox的值

    def featureExtract(self,x):
        return super().forward(x)
        
    def forward(self, template, detection):
        N = template.size(0) # ([N, 3, 127, 127]) \ ([N, 3, 271, 271])
        template_features = self.featureExtract(template)   # p2,p3,p4,p5
        detection_features = self.featureExtract(detection) # p2,p3,p4,p5

        pred_scores = []
        pred_regressions = []

        # 分层进行,模板帧和实例帧相关滤波,得到 score_size * score_size * [2,4] * anchor_num
        for i in range(len(template_features)):        
            kernel_score_t = self.conv_cls1(template_features[i])
            t_shape = kernel_score_t.shape[-1]
            kernel_score = kernel_score_t.view(N, 2 * self.anchor_num, 256, t_shape, t_shape)
            kernel_regression = self.conv_reg1(template_features[i]).view(N, 4 * self.anchor_num, 256, t_shape, t_shape)

            conv_score = self.conv_cls2(detection_features[i])    
            conv_regression = self.conv_reg2(detection_features[i]) 
            d_shape = conv_score.shape[-1]
            
            feature_map_size = d_shape - t_shape + 1

            conv_scores = conv_score.reshape(1, -1, d_shape, d_shape) 
            score_filters = kernel_score.reshape(-1, 256, t_shape, t_shape) 
            pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 2 * self.anchor_num, feature_map_size, feature_map_size)

            conv_reg = conv_regression.reshape(1, -1, d_shape, d_shape) 
            reg_filters = kernel_regression.reshape(-1, 256, t_shape, t_shape) 
            pred_regression = self.regress_adjust(
                F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 4 * self.anchor_num, feature_map_size,feature_map_size))
            
            pred_scores.append(pred_score)
            pred_regressions.append(pred_regression)

        '''
        when batch_size = 2, anchor_num = 3
        torch.Size([2, 6, 37, 37])
        torch.Size([2, 6, 19, 19])
        torch.Size([2, 6, 10, 10])
        torch.Size([2, 6, 6, 6])
        torch.Size([2, 12, 37, 37])
        torch.Size([2, 12, 19, 19])
        torch.Size([2, 12, 10, 10])
        torch.Size([2, 12, 6, 6])
        '''
        return pred_scores, pred_regressions 

    # 权重初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out',nonlinearity='relu')
                nn.init.normal_(m.weight.data, std=0.0005)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data, std=0.0005)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SiamFPN50(SiamFPN):
    def __init__(self):
        block = [3, 4, 6, 3]
        super(SiamFPN50, self).__init__(block)

class SiamFPN101(SiamFPN):
    def __init__(self):
        block = [3, 4, 23, 3]
        super(SiamFPN101, self).__init__(block)

class SiamFPN152(SiamFPN):
    def __init__(self):
        block = [3, 8, 36, 3]
        super(SiamFPN152, self).__init__(block)

def FPN101():
    return FPN(Bottleneck, [3, 4, 23, 3])

def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])

def FPN152():
    return FPN(Bottleneck, [3, 8, 36, 3])

if __name__ == "__main__":
    net = SiamFPN50()
    # fms = net(Variable(torch.randn(1, 3, 127, 127)))
    # fms = net(Variable(torch.randn(1, 3, 271, 271)))
    pred_scores, pred_regressions = net.train(Variable(torch.randn(2, 3, 127, 127)),Variable(torch.randn(2, 3, 271, 271)))
    for fm in pred_scores:
        print(fm.size())
    for fm in pred_regressions:
        print(fm.size())


# fpn101
# 1,3,127,127
# torch.Size([1, 256, 32, 32]) p2
# torch.Size([1, 256, 16, 16]) p3
# torch.Size([1, 256, 8, 8])   p4
# torch.Size([1, 256, 4, 4])   p5
# 1,3,271,271
# torch.Size([1, 256, 68, 68]) p2
# torch.Size([1, 256, 34, 34]) p3
# torch.Size([1, 256, 17, 17]) p4
# torch.Size([1, 256, 9, 9])   p5
