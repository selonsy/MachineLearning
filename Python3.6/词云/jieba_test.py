import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))        # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))     # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")        # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

'''output
Full Mode: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
Default Mode: 我/ 来到/ 北京/ 清华大学
他, 来到, 了, 网易, 杭研, 大厦
小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, ，, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
'''

'''添加自定义词典
载入词典
开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词。虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率
用法： 
jieba.load_userdict(file_name) # file_name 为文件类对象或自定义词典的路径

词典格式和 dict.txt 一样，一个词占一行；
每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。
file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。
词频省略时使用自动计算的能保证分出该词的词频。

例如：
创新办 3 i
云计算 5
凱特琳 nz
台中

范例：
之前： 李小福 / 是 / 创新 / 办 / 主任 / 也 / 是 / 云 / 计算 / 方面 / 的 / 专家 /
加载自定义词库后：　李小福 / 是 / 创新办 / 主任 / 也 / 是 / 云计算 / 方面 / 的 / 专家 /
'''