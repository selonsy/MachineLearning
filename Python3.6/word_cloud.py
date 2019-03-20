#!/usr/bin/env python
"""
Masked wordcloud
================

Using a mask you can generate wordclouds in arbitrary shapes.
"""

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba

from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator

d = path.dirname(__file__)

#读取要生成词云的文件
print(d)
o_text = open(path.join(d, 'assets/qq_word_zw_pure.txt'),"r",encoding="utf-8").read()
#通过jieba分词进行分词并通过空格分隔
seg_list = jieba.cut(o_text, cut_all=True)
# #print(seg_list)
# print(next(seg_list))
#seg_list1 = np.unique(seg_list)
# print(seg_list)
# exit()
text = " ".join(seg_list)
#print(text)
#exit()
#alice_mask = np.array(Image.open(path.join(d, "sjl.jpg")))

stopwords = set(STOPWORDS)
#stopkey_selonsy = {'今天','没有','哈哈','这个','不是','可以','还是','什么','表情','图片','视频','实时','网页消息'}
stopkey_hit=[line.strip() for line in open('assets/hit-stopwords.txt',"r",encoding="utf-8").readlines()]
#stopwords.update(stopkey_selonsy)
stopwords.update(stopkey_hit)

font = r'C:\Windows\Fonts\simfang.ttf'
wc = WordCloud(
                # 设置背景颜色
                background_color="white",
                # 设置字体
                font_path=font, 
                # 设置最大显示的词云数
                max_words=2000, 
                #mask=alice_mask,
                stopwords=stopwords,
                max_font_size=100,
                #random_state=42,
                width=1920,
                height=1080,
                # 设置有多少种随机生成状态，即有多少种配色方案
                random_state=30,
                #scale=0.5
               )    
# generate word cloud
wc.generate(text)

# create coloring from image
#image_colors = ImageColorGenerator(alice_mask)

# store to file
wc.to_file(path.join(d, "output/qq_word_zw.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()

# 我们还可以直接在构造函数中直接给颜色
# 通过这种方式词云将会按照给定的图片颜色布局生成字体颜色策略
#plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
#plt.axis("off")
#plt.figure()

# plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
# plt.axis("off")
# plt.show()
