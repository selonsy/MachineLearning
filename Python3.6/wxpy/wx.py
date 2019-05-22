# https://wxpy.readthedocs.io/zh/latest/itchat.html
# https://github.com/youfou/wxpy


from wxpy import *

from wxpy import *
bot = Bot(cache_path='D:\workspace\MachineLearning\Python3.6\wxpy/wxpy.pkl')
# embed()
# 机器人账号自身
myself = bot.self

my_friend = bot.friends().search('部长')[0]
print(my_friend)
# 向文件传输助手发送消息
# bot.file_helper.send('Hello from wxpy!')

# 在 Web 微信中把自己加为好友
# bot.self.add()
# bot.self.accept()

# 发送消息给自己
# bot.self.send('能收到吗？')


# 堵塞线程，并进入 Python 命令行
embed()