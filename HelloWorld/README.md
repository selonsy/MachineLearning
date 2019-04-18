# Siamese-RPN

This is a PyTorch implementation of SiameseRPN. This project is mainly based on [SiamFC-PyTorch](https://github.com/StrangerZhang/SiamFC-PyTorch) and [DaSiamRPN](https://github.com/foolwood/DaSiamRPN).

For more details about siameseRPN please refer to the paper : [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) by Bo Li, Junjie Yan,Wei Wu, Zheng Zhu, Xiaolin Hu.

This repository includes training and tracking codes. 

## News

After use the Youtube-BB dataset, the code can get 0.59 auc on OTB100 and about 0.315 eao on VOT2015. I upload it to the 59version folder. It's trained without warm up.

## Data preparation:

python bin/create_dataset.py --data-dir /dataset_ssd/ILSVRC2015 --output-dir /dataset_ssd/vid15rpn_finetune

python bin/create_lmdb.py --data-dir /dataset_ssd/vid15rpn_finetune --output-dir /dataset_ssd/vid15rpn_finetune.lmdb

## Traing phase:

CUDA_VISIBLE_DEVICES=2 python bin/train_siamfc.py --data_dir /dataset_ssd/vid15rpn_large

## Test phase:

CUDA_VISIBLE_DEVICES=2 python bin/test_OTB.py -ms ./models/siamrpn_* -v cvpr2013

python version == 3.6.5

pytorch version == 1.0.0

Without using imagenet pretrain or youtube-bb dataset, this code can get 0.545 auc on OTB50, and 0.22 EAO on VOT2015. The paper's model can get 0.33 EAO on VOT2017 without these.

We are still trying to reimplement the results in paper.

If you found any bug or have any suggestion about this code, hope you can tell us. 

My email address is zhangruiqi429@gmail.com. 

## Existing bugs

There are still some bugs in this code getting 0.54 AUC on OTB50, but changing these bugs doesn't get improvement by now. 

1，Trained with cv2.BGR2RGB,test without it.

2，Only load former 3 layers and didn't freeze the BN layers' parameters.

3，Use warm epochs to avoid grad explosion, can use grad clip instead.

fixed in 59versions.

## pretrained models
https://drive.google.com/drive/folders/1HJOvl_irX3KFbtfj88_FVLtukMI1GTCR
This is the pretrained modificated alexnet provided by Li Bo.


## update

can get 0.626 auc on otb100 and 0.31 eao on vot2015 with Youtube-BB dataset

## Reference

[1] Li B , Yan J , Wu W , et al. High Performance Visual Tracking with Siamese Region Proposal Network[C]// 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2018.
