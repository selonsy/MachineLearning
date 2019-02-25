# -*- coding:utf-8 -*-
'''
题目描述
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

时间限制：1秒 空间限制：32768K
'''

class Solution:
    # array 二维列表
    # 第一种：很傻逼的全部遍历。PS：竟然也过了，简直了。。。
    def Find(self, target, array):        
        for i in range(len(array)):
            for j in range(len(array[0])):
                if array[i][j] == target:
                   return True
        return False 
    # 第二种：考虑数据的特点，从左下角或者右上角开始。以左下角为例：
    # 目标比当前大，右移。比当前小，上移。
    def Find_best(self,target,array):
        i = len(array)-1
        j = 0
        while i >= 0 and j < len(array[0]):
        # for(i=m-1;i>=0;i=i-1):
        #     for(j=0;j<=n;j=j+1):
        # python中的for循环不支持上面的形式，range可以设定步长，步长可以为负数，但是貌似不能用变量初始化range。
                if array[i][j] == target:
                    return True
                elif array[i][j]<target:
                    j=j+1
                else:
                    i=i-1
        return False



# Test
target=11
array=[
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
result1 = Solution().Find(target,array)
result2 = Solution().Find_best(target,array)
print(result1)
print(result2)