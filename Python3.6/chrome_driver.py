#coding:utf-8
'''
selenium 自动化测试工具

'''
from selenium import webdriver # 导入webdriver包

import time

driver = webdriver.Chrome()# 初始化一个谷歌浏览器实例：driver
# 加启动配置

option = webdriver.ChromeOptions()
option.add_argument('disable-infobars')#  屏蔽"chorme正受到自动测试软件的控制"提示信息

#启动浏览器的时候不想看的浏览器运行，那就加载浏览器的静默模式，让它在后台偷偷运行。用headless
option.add_argument('headless')

driver.maximize_window() # 最大化浏览器 

time.sleep(5) # 暂停5秒钟

driver.get("http://www.taobao.com") # 通过get()方法，打开一个url站点