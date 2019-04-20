# -*- coding: utf-8 -*-
# 采用结巴分词对聊天记录等数据进行过滤清洗

import csv
import codecs
import jieba
import re

# 读取分词表数据
def read_stopwords(filepath):
    return [line.strip() for line in codecs.open(filepath, 'r', encoding='utf-8').readlines()]

stopword_1 = read_stopwords("dict/stop.dat")
stopword_2 = read_stopwords("dict/hit-stopwords.txt")
stopword_3 = read_stopwords("dict/my_stopwords.txt")
# 合并去重
stopwords = list(set(stopword_1+stopword_2+stopword_3))
# print(len(stopword_1))
# print(len(stopword_2))
# print(len(stopword_3))
# print(len(stopwords))

def filter_data(input_file, output_file):
    with codecs.open(output_file, 'w', encoding='utf-8') as w:
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            for row in f:
                cut = jieba.lcut(row)
                stop_cut = []
                for word in cut:
                    if word.strip() != '' and word not in stopwords:
                        stop_cut.append(word)
                if len(stop_cut) > 0:
                    w.write(" ".join(stop_cut))
                    w.write("\r")  # w.write("\n") 会不会起作用？

def filter_data_wechat(input_file, output_file):
    with codecs.open(output_file, 'w', encoding='utf-8') as w:
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            for row in f: 
                # 智障星驻地球傻逼办副主任(2017/5/14 13:50:02): 晚上回去打电话给我[Smirk]\r\n              
                # 彦承(2017/5/15 12:52:30): 嗯嗯[捂脸]\r\n
                
                # [奸笑],[Smirk]等等,表示[]里面的表示表情,需要用正则去掉
                
                # 智障星驻地球傻逼办副主任(2017/5/30 19:35:34): <msg><videomsg length="1987331" playlength="14" offset="1987344" fromusername="" status="4" cameratype="0" source="1"                                          aeskey="04a2dff6467340a585e2577e74904eeb" cdnvideourl="30680201000461305f020100020414545efd02032e20990204a8c402b70204592d58fb043d617570766964656f5f303661383639346539646430343361325f313439363134343130375f3139333530373330303531373462333563336237323336330201000201000400" cdnthumburl="30680201000461305f020100020414545efd02032e20990204a8c402b70204592d58fb043d617570766964656f5f303661383639346539646430343361325f313439363134343130375f3139333530373330303531373462333563336237323336330201000201000400" cdnthumblength="17462" cdnthumbwidth="960" cdnthumbheight="544" cdnthumbaeskey="04a2dff6467340a585e2577e74904eeb" encryver="1" fileparam="" md5 ="1cde767fa678367dbbdd884ff6460214" newmd5 ="82b4dee72b39a0510d081620972dca78"  /><statextstr></statextstr></msg>
                # <img src="SystemMessages_HongbaoIcon.png"/>You've opened the <_wc_custom_link_ color="#FD9931" href="weixin://weixinhongbao/opendetail?sendid=1000039401201705316019595987206&sign=6d0b58754c2d3f8ce18fba87264ac29c93b05766b4c3a00e25ec5039c8d06db09f66a61fb407844b4817902d8397d88ab6611518501f9708de5e257d9dc8f267135fa208cee3ff9e7e2153922b135de4201a6be15c3db6d00c7657113a9c0bfa&ver=6">Red Packet</_wc_custom_link_> of 邱小燕.
                # 智障星驻地球傻逼办副主任(2017/7/30 17:32:15): <?xml version="1.0"?>
                # <msg>
                # 	<videomsg aeskey="beb3d4c4dc7d4e5e96faec3bfeee494c" cdnthumbaeskey="beb3d4c4dc7d4e5e96faec3bfeee494c" cdnvideourl="30680201000461305f0201000204f809e8a202032dcd670204a54de6b70204597c8bd2043d617570766964656f5f623365313530393736323530393937655f313530313333343438305f3231323131383239303731373561316139373138303736380201000201000400" cdnthumburl="30680201000461305f0201000204f809e8a202032dcd670204a54de6b70204597c8bd2043d617570766964656f5f623365313530393736323530393937655f313530313333343438305f3231323131383239303731373561316139373138303736380201000201000400" length="1871475" playlength="9" cdnthumblength="9707" cdnthumbwidth="290" cdnthumbheight="512" fromusername="wxid_pxwab3lnpriu22" md5="0adebc7e02079534913c928ec1c899e2" newmd5="28f4c0b046fb1c6b7947b7123d62996a" isad="0" />
                # </msg>
                #智障星驻地球傻逼办副主任(2017/8/13 14:12:12): wxid_pxwab3lnpriu22:<sysmsg type="NewXmlVoipRedialMsg"><NewXmlVoipRedialMsg><text><![CDATA[Call interrupted by other apps ]]></text><link><scene>voipredial_voice</scene><text><![CDATA[Redial]]></text></link>
                # </NewXmlVoipRedialMsg>
                # </sysmsg>

                # 有一些系统消息,需要去掉,采用正则,匹配所有的<>    
                # 有一些链接消息,如:智障星驻地球傻逼办副主任(2017/3/18 10:16:34): 【情侣装春装2017新款韩版春季连帽休闲情侣卫衣学生上衣大码班服潮】http://c.b1wt.com/h.eaeVkU?cv=2G01N7GEB0&sm=76c0c9 点击链接，再选择浏览器打开；或复制这条信息，打开👉手机淘宝👈￥2G01N7GEB0￥            
                sys_msg_re = re.compile('<[\S\s]*>')
                url_re = re.compile('((ht|f)tps?):\/\/[\w\-]+(\.[\w\-]+)+([\w\-\.,@?^=%&:\/~\+#]*[\w\-\@?^=%&\/~\+#])?')
                if sys_msg_re.search(row) or url_re.search(row):
                    continue  
                if "彦承" not in row:
                    continue    
                p = re.compile('\[\S*]')
                chat = row.split(":")[-1].replace("\r\n", "")                
                cut = jieba.lcut(re.sub(p, "", chat))
                stop_cut = []
                for word in cut:
                    if word.strip() != '' and word not in stopwords:
                        stop_cut.append(word)
                if len(stop_cut) > 0:
                    w.write(" ".join(stop_cut))
                    w.write("\n")

if __name__ == "__main__":
    # 加载用户自定义分词表
    jieba.load_userdict("dict/user_dict.dat")
    # jieba.enable_parallel(8)
    file_name = 'wangyuanyuan__.txt' # Q-S-Y-S.txt  test.txt
    input_file = 'ori_data/{0}'.format(file_name)
    output_file = 'filter_data/s_{0}'.format(file_name)
    
    # row = '智障(2017/7/30 17:32:15): <?xml version="1.0"?>'
    # sys_msg_re = re.compile('<[\S\s]*>')
    # ss = sys_msg_re.search(row)
    # exit(0)
    filter_data_wechat(input_file,output_file)
    
    print("done")
