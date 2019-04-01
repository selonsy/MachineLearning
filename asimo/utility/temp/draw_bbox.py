import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from utility import *
import re
# from scripts import *

'''
功能：
1、输入一张图片的地址，以及一个给定的BBox，在图片上画出来。
2、输入一张图片的地址，以及两个给定的BBox，在图片上画出来（不同的颜色），并在控制台中计算Precsion & Accurancy（即中心点误差和重叠率）。
3、输入一系列图片的地址，以及一系列给定的BBox，在视频中画出BBox，一直到视频结束。
4、输入一系列图片的地址，以及两套给定的BBox，在视频中画出BBox（不同的颜色），并在控制台中计算Precsion & Accurancy。
'''

seq_list_path = 'D:\\workspace\\vot\\asimo\\SiamFPN\\bag\\'
ground_truth_txt = "D:\\workspace\\vot\\asimo\\SiamFPN\\bag\\groundtruth.txt"
result_txt = "D:\\workspace\\vot\\asimo\\SiamFPN\\results\\DaSiamRPN_bag_1.txt"

# 功能序号
function_num = 2

pics=[]
bboxs=[]
bboxs1=[]
if function_num==1:
    pics = [r"D:\workspace\vot\asimo\SiamFPN\bag\00000001.jpg"]
    bboxs = ['334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41']
elif function_num==2:
    pics = [r"D:\workspace\vot\asimo\SiamFPN\bag\00000001.jpg"]
    bboxs = [
    '334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41',
    '305.83252152099124 137.6370619698798 109.45960413666742 95.85583463262809',
    '312.1420511293659,146.38778561323034,106.13089774126823,96.41442877353934']
    # 0:312.1420511293659
    # 1:146.38778561323034
    # 2:106.13089774126823
    # 3:96.41442877353934
elif function_num==3:
    pics = sorted(glob.glob(seq_list_path+'*.jpg'))
    bboxs = [encode_region(convert_region(parse_region(x.strip('\n')),'rectangle')) for x in open(ground_truth_txt, 'r').readlines()]
elif function_num==4:
    pics = sorted(glob.glob(seq_list_path+'*.jpg'))
    bboxs = [encode_region(convert_region(parse_region(x.strip('\n')),'rectangle')) for x in open(ground_truth_txt, 'r').readlines()]    
    bboxs1 = [encode_region(convert_region(parse_region(x.strip('\n')),'rectangle')) for x in open(result_txt, 'r').readlines()]    

def view_result():
       
    if function_num==1:
        fig = plt.figure()
        image = Image.open(pics[0]).convert('RGB')
        im = plt.imshow(image, zorder=0)

        res = convert_region(parse_region(bboxs[0]),'rectangle')
        x, y, w, h = res.x,res.y,res.width,res.height

        rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)       
        plt.gca().add_patch(rect)                
        # fig.show()
        plt.show()
        input()
    if function_num==2:
        fig = plt.figure()
        image = Image.open(pics[0]).convert('RGB')
        im = plt.imshow(image, zorder=0)

        res = convert_region(parse_region(bboxs[0]),'rectangle')
        gres = convert_region(parse_region(bboxs[1]),'rectangle')        
        x, y, w, h = res.x,res.y,res.width,res.height
        gx, gy, gw, gh = gres.x,gres.y,gres.width,gres.height
            
        rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)    
        gtRect = plt.Rectangle((gx, gy), gw, gh, linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)   
        plt.gca().add_patch(rect)    
        plt.gca().add_patch(gtRect)   

        if len(bboxs)==3:
            bres = convert_region(parse_region(bboxs[2]),'rectangle')
            bx, by, bw, bh = bres.x,bres.y,bres.width,bres.height 
            btRect = plt.Rectangle((bx, by), bw, bh, linewidth=3, edgecolor="#0000ff", zorder=1, fill=False)    
            plt.gca().add_patch(btRect)  

        bbox_a = [x, y, w, h]
        bbox_b = [gx, gy, gw, gh]

        error = round(compute_distance(bbox_a,bbox_b),2)
        overlap = round(compute_iou(bbox_a,bbox_b),2)

        plt.text(0, -10, "error={0}".format(error), size=15)
        plt.text(150, -10, "overlap={0}".format(overlap), size=15)
        # fig.show()
        plt.show()
        input()
    if function_num==3:
        fig = plt.figure()
        image = Image.open(pics[0]).convert('RGB')
        im = plt.imshow(image, zorder=0)

        res = convert_region(parse_region(bboxs[0]),'rectangle')
        x, y, w, h = res.x,res.y,res.width,res.height
        
        rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)       
        plt.gca().add_patch(rect)  
            
        def update_fig(num):    
            image = Image.open(pics[num]).convert('RGB')
            im.set_data(image)
            res = convert_region(parse_region(bboxs[num]),'rectangle')
            x, y, w, h = res.x,res.y,res.width,res.height                        
            rect.set_xy((x,y))
            rect.set_width(w)
            rect.set_height(h)           
            return im, rect
        ani = animation.FuncAnimation(fig, update_fig, frames=len(pics), interval=10, blit=True)
        plt.axis("off")
        plt.show()
        input()
    if function_num==4:
        # fig = plt.figure()
        fig,ax = plt.subplots()
        ax.grid()
        error_template = 'error = %.2f pixel'
        error_text = ax.text(0.05, 0.9, '', transform=ax.transAxes,color = "y",weight="bold")
        overlap_template = 'overlap = %.2f %%'
        overlap_text = ax.text(0.40, 0.9, '', transform=ax.transAxes,color = "y",weight="bold")

        image = Image.open(pics[0]).convert('RGB')
        im = plt.imshow(image, zorder=0)

        res = convert_region(parse_region(bboxs[0]),'rectangle')
        gres = convert_region(parse_region(bboxs1[0]),'rectangle')
        x, y, w, h = res.x,res.y,res.width,res.height
        gx, gy, gw, gh = gres.x,gres.y,gres.width,gres.height

        rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)    
        gtRect = plt.Rectangle((gx, gy), gw, gh, linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)   
        plt.gca().add_patch(rect)    
        plt.gca().add_patch(gtRect)          
        
        def update_fig(num):    
            image = Image.open(pics[num]).convert('RGB')
            im.set_data(image)
            res = convert_region(parse_region(bboxs[num]),'rectangle')
            gres = convert_region(parse_region(bboxs1[num]),'rectangle')
            x, y, w, h = res.x,res.y,res.width,res.height    
            gx, gy, gw, gh = gres.x,gres.y,gres.width,gres.height        
            im.set_data(image)
            rect.set_xy((x,y))
            rect.set_width(w)
            rect.set_height(h) 
            gtRect.set_xy((gx,gy))
            gtRect.set_width(gw)
            gtRect.set_height(gh) 

            bbox_a = [x, y, w, h]
            bbox_b = [gx, gy, gw, gh]

            error = round(compute_distance(bbox_a,bbox_b),2)
            overlap = round(compute_iou(bbox_a,bbox_b),4)    
            
            error_text.set_text(error_template %(error))
            overlap_text.set_text(overlap_template %(overlap*100))                      

            return im, rect,gtRect,error_text,overlap_text
        ani = animation.FuncAnimation(fig, update_fig, frames=len(pics),interval=50, blit=True) # init_func=init, 
        plt.axis("off")
        plt.show()
        input()

if __name__ == '__main__':
    view_result()
