'''
在由若干 0 和 1 组成的数组 A 中，有多少个和为 S 的非空子数组。

示例：
输入：A = [1,0,1,0,1], S = 2
输出：4
解释：
如下面黑体(加引号表示)所示，有 4 个满足题目要求的子数组：
['1,0,1',0,1]
['1,0,1,0',1]
[1,'0,1,0,1']
[1,0,'1,0,1']

提示：
    A.length <= 30000
    0 <= S <= A.length
    A[i] 为 0 或 1
'''
import time
import copy
class Solution:    
    # def numSubarraysWithSum(self, A: List[int], S: int) -> int:
    def func1(self, A, S): # 时间复杂度o(n^2) 超时了 53 / 59 
        l = len(A)
        res = 0
        _sum = 0          
        for i in range(l):                          
            for j in range(i,l):
                _sum += A[j]
                if _sum == S:
                    res += 1
                if _sum > S:                   
                    break  
            _sum = 0              
        return res
    def func2(self, A, S):
        P = [0]
        for x in A: 
            P.append(P[-1] + x)
        import collections
        count = collections.Counter()

        ans = 0
        for x in P:
            ans += count[x]
            count[x + S] += 1

        return ans
    def func3(self, A, S):
        indexes = [-1] + [ix for ix, v in enumerate(A) if v] + [len(A)]
        ans = 0
        if S == 0:
            for i in range(len(indexes) - 1):
                # w: number of zeros between two consecutive ones
                w = indexes[i+1] - indexes[i] - 1
                ans += w * (w+1) / 2  # n个连续0中选长度大于等于1的连续子串数量
            return ans
        for i in range(1, len(indexes) - S):
            j = i + S - 1
            left = indexes[i] - indexes[i-1]
            right = indexes[j+1] - indexes[j]
            ans += left * right
        return ans
    
    def getVarLen(self):
        co = self.func1.__code__        
        return co.co_argcount

ori_data = [[[1,0,1,0,1],2],[[0,0,0,0,1],2]]
expect_data = [4,0]
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
