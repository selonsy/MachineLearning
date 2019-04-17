
import csv
import re
from selenium import webdriver
from bs4 import BeautifulSoup

''' 爬虫之网易云音乐歌单歌曲信息 selonsy 2019_4_8_15_01_33
1.列表信息在<iframe>内联框架中，使用urllib3或者requests无法爬取到内联框架的内容，因此我们使用selenium获取<iframe>框架的内容.
2.

'''

# chromedriver的路径（下载后放在chrome的安装路径即可，版本需要与chrome相匹配）
path = 'C:\Program Files (x86)\Google\Chrome\Application\chromedriver'
# 歌单id
songlist_id = '74274667'  # 彦承喜欢的音乐
# 保存文件名称
songlist_name = 'selonsy' # CSV格式

# 歌单的URL
url = 'https://music.163.com/#/playlist?id={0}'.format(songlist_id)
driver = webdriver.Chrome(path)
driver.get(url)
# 歌单信息嵌入在iframe里面，id为g_iframe
# 如果报错missing value值，则将Chrome与Chromedriver更新到最新
driver.switch_to.frame('g_iframe')
# lxml是个解析器
soup = BeautifulSoup(driver.page_source, 'lxml')


# 获取所有的行标签tr，并进行遍历
trlist = soup.tbody.find_all('tr')
songlist = []
songlist_no_authorized = []
for each in trlist:

    # soup.find_all(name='div',attrs={"class":"footer"}) #按照字典的形式给attrs参数赋值    

    # replace('\xa0',' ')：\xa0 是不间断空白符 &nbsp; 
    # 我们通常所用的空格是 \x20 ，是在标准ASCII可见字符 0x20~0x7e 范围内。而 \xa0 属于 latin1 （ISO/IEC_8859-1）中的扩展字符集字符，代表空白符nbsp(non-breaking space)。 latin1 字符集向下兼容 ASCII （ 0x20~0x7e ）。

    # id
    id = each.find('td').div.span['data-res-id']
    # 歌曲名称
    name = each.find('b')['title'].replace('\xa0', ' ')
    # 作者 & 专辑
    temp = each.find_all('div', 'text')
    author = temp[0].span['title'].replace('\xa0', ' ')
    folder = temp[1].a['title'].replace('\xa0', ' ')
    
    # 是否有版权（即该行显示为灰色），是否包含class：js-dis
    is_authorized = 0 if 'js-dis' in each['class'] else 1
    
    song = []
    song.append(id)
    song.append(name)
    song.append(author)
    song.append(folder)
    song.append(is_authorized)

    songlist.append(song)

    if not is_authorized:
        songlist_no_authorized.append(song)

# 导出为CSV文件
# encoding采用utf-8，否则默认为gbk，很多特殊字符会失败
# 'w' 表示写,'a' 表示追加
with open('./spider4music/docs/%s.csv' % songlist_name, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, delimiter='$') # delimiter分隔符,例如：95871$偷功$胡伟立$太极张三丰电影原声带$1
    for row in songlist:
        writer.writerow(row)          

with open('./spider4music/docs/no_authorized_list.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, delimiter='$')
    for row in songlist_no_authorized:
        writer.writerow(row)   
