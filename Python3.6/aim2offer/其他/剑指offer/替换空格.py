'''
题目描述
请实现一个函数，将一个字符串中的每个空格替换成“%20”。
例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

时间限制：1秒 空间限制：32768K
'''

# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here

        # 下面这种不对，替换了第一次的时候，s的长度发生了变化，但是for循环的次数没变，位置错位。
        # l = len(s)
        # for x in range(l):
        #     if s[x] == " " and x < l-1:
        #         s = s[:x] + "%20" + s[x+1:]
        #     elif s[x] == " " and x == l-1:
        #         s = s[:x] + "%20"
        # return s

        l = len(s) 
        ss = ""       
        for x in range(l):
            if s[x] == " ":
                ss = ss + "%20" 
            else:
                ss = ss + s[x]
        return ss

s1 = "We Are Happy"
s2 = "   "
o1 = "We%20Are%20Happy"
o2 = "%20%20%20"
r1 = Solution().replaceSpace(s1)
r2 = Solution().replaceSpace(s2)
print(r1==o1)
print(r2==o2)