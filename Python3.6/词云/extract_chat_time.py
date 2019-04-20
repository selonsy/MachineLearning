# -*- coding: utf-8 -*-
# 聊天时间段统计
import csv
import time
import codecs
import seaborn as sns
import matplotlib.pyplot as plt

process_file_name = '2019'
chat_time_list = []
with codecs.open("data/{}.csv".format(process_file_name), 'r', encoding='utf-8') as f:
    csv_file = csv.reader(f)
    for row in csv_file:
        if len(row) > 0:
            try:
                chat_time = time.localtime(int(row[0]))
                chat_time_list.append(chat_time.tm_hour * 60 + chat_time.tm_min)
            except Exception as e:
                pass

sns.set()
sns.kdeplot(chat_time_list, cut=0, shade=True)
plt.yticks([])

plt.xticks((0, 240, 480, 720, 960, 1200, 1439),
           ('0:00', '04:00', '08:00', '12:00', '14:00', "20:00", "23:59"))
plt.savefig("data/{}_chat_time.png".format(process_file_name))
