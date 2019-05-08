'''
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
问总共有多少条不同的路径？
例如，上图是一个7 x 3 的网格。有多少可能的路径？

说明：m 和 n 的值均不超过 100。

示例 1:
输入: m = 3, n = 2
输出: 3
解释:
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右

示例 2:
输入: m = 7, n = 3
输出: 28
'''
import time
import copy
class Solution:   
    # def uniquePaths(self, m: int, n: int) -> int: 
    def func1(self, m, n):
        if n == 1: return 1
        if n == 2: return m
        if n == 3: return m*(1+m)/2
        res = 0
        for i in range(n-1,0,-1):
            res += i * Solution.func1(self, i, n-1)
        return res
     
    def getVarLen(self):
        co = self.func1.__code__        
        return co.co_argcount

ori_data = [[3,2],[7,3],[3,7]]
expect_data = [3,28,28]
assert len(ori_data) == len(expect_data),'输入输出数据数量不一致'

s = Solution()
funcs = []
var_len = s.getVarLen() - 1 # exclude self
for key in Solution.__dict__.keys():
    if "func" in key:
        funcs.append(getattr(s, key)) # 反射获取实例化的func
for f in range(len(funcs)):
    func = funcs[f]
    assert func!=None,'func不能为空！'
    begin = time.time() # ToDo：这里的计时有问题，需要改进
    data_length = len(expect_data)
    input_data = copy.deepcopy(ori_data) # 前面的func可能修改测试数据，故深拷贝
    for i in range(data_length):
        if var_len == 1:
            res = func(input_data[i])
            assert res == expect_data[i], "func{0}({3}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i])
        elif var_len == 2:
            res = func(input_data[i][0],input_data[i][1])
            assert res == expect_data[i], "func{0}({3},{4}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i][0],input_data[i][1])
        elif var_len == 3:
            res = func(input_data[i][0],input_data[i][1],input_data[i][2])
            assert res == expect_data[i], "func{0}({3},{4},{5}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i][0],input_data[i][1],input_data[i][2])
        elif var_len == 4:
            res = func(input_data[i][0],input_data[i][1],input_data[i][2],input_data[i][3])
            assert res == expect_data[i], "func{0}({3},{4},{5},{6}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i][0],input_data[i][1],input_data[i][2],input_data[i][3])
    end = time.time()
    print("func{0} : {1:.4f} ms".format(f+1, (end-begin)*1000/data_length))
print("done")
