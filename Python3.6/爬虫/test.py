

#/usr/bin/env python
# -*- coding:utf-8 -*-
 #设置文件编码为utf-8，为了让系统识别，如果系统文件编码为gbk，该文件将不会被识别。
#Author:W-D
import sys
print (sys.getdefaultencoding())#打印当前系统默认编码（不一定是头中声明的编码）
test="你好"#该变量的编码是unicode（utf-8），文件编码如果是gbk，该变量编码依然不会改变
gbk_test=test.encode("gbk") #转码为gbk
print(gbk_test)#打印
print(gbk_test.decode("gbk").encode("utf-8"))#gbk转utf-8
print(gbk_test.decode("gbk"))#将gbk解码为unicode，变成字符串打印

# print(b'\xae'.decode('utf-8'))

# 结果：
# utf-8
# b'\xc4\xe3\xba\xc3'
# b'\xe4\xbd\xa0\xe5\xa5\xbd'
# 你好
print('®'.encode('utf-8'))
print(b'\xa0'.decode('utf-8'))