# -*- coding: utf-8 -*-

import codecs
from wordcloud import WordCloud

file_name = 's_wangyuanyuan__' # qq_word Q-S-Y-S
with codecs.open("filter_data/{}.txt".format(file_name), 'r', encoding='utf-8') as f:
    text = f.read()
font_path = r'C:\Windows\Fonts\msyh.ttf'
wordcloud = WordCloud(background_color="white", 
                      width=2000, height=1720, 
                      margin=2, max_words=300, 
                      font_step=2,
                      font_path=font_path
                      ).generate(text)

wordcloud.to_file("output/{}.png".format(file_name))
