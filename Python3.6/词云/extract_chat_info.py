# -*- coding: utf-8 -*-
# 聊天时间段统计
import csv
import time
import codecs
import seaborn as sns
import matplotlib.pyplot as plt
import xml.dom.minidom as xmldom
# 彦承(2017/6/8 19:35:01):  Cancelled
# <voipinvitemsg><roomid>0</roomid><key>0</key><status>4</status><invitetype>1</invitetype></voipinvitemsg>
# <voipextinfo><recvtime>1496921701</recvtime></voipextinfo>
# <voiplocalinfo><wordingtype>1</wordingtype><duration>0</duration></voiplocalinfo>

# 1、我们累计打了多少次电话，其中成功多少次（指时长大于0的）。累计通话分钟数，换算成小时，换算成天，换算成秒。
# 2、统计我和她分别的情况。
# 3、统计打电话的时间段。

process_file_name = 'sjl_qsy'
chat_time_list = []
chat_time_sum = {'me':0,'other':0}
chat_time_count = {'me':{'success':0,'fail':0},'other':{'success':0,'fail':0}}
with codecs.open("ori_data/Q-S-Y-S-with-record.txt", 'r', encoding='utf-8') as f:
    for row in f:    
        # if len(row) > 0:
        #     try:
        #         chat_time = time.localtime(int(row[0]))
        #         chat_time_list.append(chat_time.tm_hour * 60 + chat_time.tm_min)
        #     except Exception as e:
        #         pass
        if 'voipinvitemsg' not in row: # 仅处理语音通话的部分
            continue        
        if '彦承' in row:
            id = 'me'
        elif '智障星驻地球傻逼办副主任' in row:
            id = 'other'
        else:
            raise Exception("用户名不在其中!")        
        chat = row.split(":")[-1].replace("\r\n", "")
        chat = '<root>' + chat + '</root>'
        dom = xmldom.parseString(chat)
        recvtime = dom.getElementsByTagName('recvtime')[0].childNodes[0].data
        # 单位 秒, 不是每个都有 duration 标签。
        duration_dom = dom.getElementsByTagName('duration')
        duration = 0
        if len(duration_dom)>0:
            duration = duration_dom[0].childNodes[0].data
        chat_time = time.localtime(int(recvtime))
        chat_time_list.append(chat_time.tm_hour * 60 + chat_time.tm_min)
        
        chat_time_sum[id] += int(duration)
        if int(duration) > 0:
            chat_time_count[id]['success'] += 1
        else:
            chat_time_count[id]['fail'] += 1
print(chat_time_sum)
print(chat_time_count)

sns.set()
sns.kdeplot(chat_time_list, cut=0, shade=True)
plt.yticks([])

plt.xticks((0, 240, 480, 720, 960, 1200, 1439),
           ('0:00', '04:00', '08:00', '12:00', '14:00', "20:00", "23:59"))
plt.savefig("output/{}_chat_time.png".format(process_file_name))
