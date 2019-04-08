
'''
貌似只能下载有版权的歌曲
'''


import csv
import urllib.request
import importlib
import json
import binascii
from Crypto.Cipher import AES
import requests
import glovar
from core.common import *

def get_mp3(id,name):
    url = 'http://music.163.com/song/media/outer/url?id=' + str(id)+'.mp3'
    try:
        print("正在下载：{0}".format(name))
        urllib.request.urlretrieve(url, 'songs/{0}.mp3'.format(name))
        print("Finish...")
    except Exception as e:
        print("Fail...")

def netease_download(music):
    ''' 从网易云音乐下载 '''
    eparams = {
        'method': 'POST',
        'url': 'http://music.163.com/api/song/enhance/player/url',
        'params': {
            'ids': [music['id']],
            'br': 320000,
        }
    }
    data = {'eparams': encode_netease_data(eparams)}

    s = requests.Session()
    s.headers.update(glovar.FAKE_HEADERS)
    s.headers.update({
        'referer': 'http://music.163.com/',
    })
    if glovar.get_option('proxies'):
        s.proxies.update(glovar.get_option('proxies'))

    r = s.post('http://music.163.com/api/linux/forward', data=data)

    if r.status_code != requests.codes.ok:
        raise RequestError(r.text)
    j = r.json()
    if j['code'] != 200:
        raise ResponseError(j)

    music['url'] = j['data'][0]['url']
    music['rate'] = int(j['data'][0]['br'] / 1000)
    music['size'] = round(j['data'][0]['size'] / 1048576, 2)
    music['name'] = '%s - %s.mp3' % (music['singer'], music['title'])

    music_download(music)


def encode_netease_data(data):
    data = json.dumps(data)
    key = binascii.unhexlify('7246674226682325323F5E6544673A51')
    encryptor = AES.new(key, AES.MODE_ECB)
    # 补足data长度，使其是16的倍数
    pad = 16 - len(data) % 16
    fix = chr(pad) * pad
    byte_data = (data + fix).encode('utf-8')
    return binascii.hexlify(encryptor.encrypt(byte_data)).upper().decode()

glovar.init_option()

with open('selonsy1.csv', 'r', encoding="utf-8") as f:
   data = list(csv.reader(f,delimiter='$'))
   for d in data:
       # id = d[0]
       # name = d[1]
       music = {'id':d[0],'singer':d[2],'title':d[1]}       
       netease_download(music)