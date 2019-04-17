
'''
网易云下载有版权的歌曲（包括需要会员付费下载的歌曲）
（无版权的歌曲url会返回空）

今天我们来下载无版权的歌曲
'''

import csv
import threading
from spider4music.utils import glovar
from spider4music.core.extractors.netease import *
from main import music_search,music_download
from spider4music.core.common import music_list_merge

# 初始化
glovar.init_option()
glovar.set_option('outdir','./spider4music/songs')



def download_by_name(keyword):
    music_list = []
    thread_pool = []
    errors = []

    glovar.set_option('keyword', keyword)

    # 多线程搜索
    for source in glovar.get_option('source').split():
        t = threading.Thread(target=music_search, args=(source, music_list, errors))
        thread_pool.append(t)
        t.start()
    for t in thread_pool:
        t.join()


    # 对搜索结果排序和去重
    music_list = music_list_merge(music_list)

    idx = 0
    if len(music_list)==0:
        print("{0} 搜索结果为空!".format(keyword))
        return
    else:

        # 暂时选择size最大的下载
        # 有问题，可能是别人翻唱的，还有一些合成的电音size也比较大。
        best_size = 0
        for i,d in enumerate(music_list):
            if int(d["size"])>best_size:
                idx = i 

    #[{'album': '筱筱琴影', 'duration': '0:02:56', 'id': 495575651, 'singer': '皮特潘', 'size': 6.73, 'source': 'netease', 'title': '走过咖啡屋\xa0十孔口琴（Cover 千百惠）'}, {'album': '六弦赋', 'duration': '0:02:56', 'id': 476550971, 'singer': '武汉吉他联盟', 'size': 6.73, 'source': 'netease', 'title': '《走过咖啡屋》-十孔口琴独奏（Cover 千百惠）'}, {'album': '流淌的歌声 第十期', 'duration': '0:02:59', 'id': 1350803651, 'singer': '千百惠', 'size': 6.83, 'source': 'netease', 'title': '走过咖啡屋 (Live)'}]
    
    music_download(idx, music_list)



with open('./spider4music/docs/no_authorized_list.csv', 'r', encoding="utf-8") as f:
   data = list(csv.reader(f,delimiter='$'))
   for d in data:
       # d [284245,走过咖啡屋,千百惠,想你的时候,0]
       
       keyword = d[1] + ' ' + d[2]

       #keyword = '走过咖啡屋 千百惠'

       # 应该把标题里面的括号里面的东西包括括号本身都删除再搜索
       # 让关键字尽量的精简并有效。
       download_by_name(keyword)

       #break