'''
主流数据文件类型（.dat/.txt/.json/.csv）导入导出到python
'''

# csv 1
import pandas as pd
a = [1,2,3]
b = [4,5,6]    
dataframe = pd.DataFrame({'a_name':a,'b_name':b})                 # 将数据转成dataframe类型
dataframe.to_csv("test.csv",index=False,sep=',')                  # 保存数据：将DataFrame存储为csv格式，index表示是否显示行名，default=True，sep表示分隔符
data = pd.read_csv('test.csv',encoding = 'GBK', engine="python")  # 读取数据：data的格式：DataFrame

# csv 2
import csv
with open("test.csv","w") as csvfile:                             # 保存数据
    data = csv.writer(csvfile)    
    data.writerow(["index","a_name","b_name"])                    # 先写入columns_name    
    data.writerows([[0,1,3],[1,2,3],[2,3,4]])                     # 写入多行用writerows
with open("test.csv","r") as csvfile:                             # 读取数据
    data = csv.reader(csvfile)                                    # 这里不需要readlines
    for line in data:
        pass
    
# dat

