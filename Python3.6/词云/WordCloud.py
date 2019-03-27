
import collections
import numpy as np
import re
import jieba
from PIL import Image
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
#list0 = jieba.cut('我原来是一名java工程师，后来喜欢上了python语言', cut_all=True)
#print("全模式:",list(list0))

#list1 = jieba.cut('我原来是一名java工程师，后来喜欢上了python语言', cut_all=False)
#print("精准模式:",list(list1))

##输出结果：
#全模式: ['我', '原来', '是', '一名', 'java', '工程', '工程师', '', '', '后来', '喜欢', '上', '了', 'python', '语言']
#精准模式: ['我', '原来', '是', '一名', 'java', '工程师', '，', '后来', '喜欢', '上', '了', 'python', '语言']

data_txt = open(r"D:\workspace\MachineLearning\Python3.6\词云\wechat_log.txt",'r',encoding='utf-8').read()

#文本预处理
pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"')
data_txt = re.sub(pattern, '', data_txt)

#文本分词
cut_txt = jieba.cut(data_txt)
object_list=[]
remove_words = [u"的","习近平",u'对',u'等',u'能',u'都',u'。',u' ',u'、',u'中',u'在',u'了',u'，',u'“',u'”',u'一个',u'是',u'人民日报']

#词频统计
for word in cut_txt:
    if word not in remove_words:
        object_list.append(word)

word_counts = collections.Counter(object_list)
#print(word_counts)

#定义词频背景
path_image = r'D:\workspace\MachineLearning\Python3.6\词云\word_cloud_pic.jpg'
background_image = np.array(Image.open(path_image))
font_path = r'C:\Windows\Fonts\simfang.ttf'
wd = WordCloud(
    font_path=font_path,     #设置字体格式，不然会乱码
    background_color="white",#设置背景颜色
    mask=background_image,   #设置背景图
    max_words=4000,          #设置最大显示的词云数
    max_font_size=100                
).generate_from_frequencies(word_counts)

#保存词云图
wd.to_file('zhu.png')
#显示词云图
plt.imshow(wd,interpolation="bilinear")
plt.axis("off")
plt.show()