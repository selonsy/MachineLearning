'''
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

进阶:

如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。

'''
import time
import copy
class Solution:
    # def maxSubArray(self, nums: List[int]) -> int:
    def func1(self, nums):
        for i in range(1, len(nums)):
            nums[i] = nums[i] + max(nums[i-1], 0)
        return max(nums)
    def func2(self,nums): 
        '''    
        r[0]代表以当前位置为结尾的局部最优解
        r[1]代表全局最优解
        '''
        from functools import reduce
        return reduce(lambda r, x: (max(r[0], r[1]+x), max(r[1]+x,x)), nums, (max(nums), 0))[0]
    def func3(self,nums): 
        '''
        动态递归： 主要是利用逐步求解，以连续数组结束位置为每一步的解，sum其实就是记录了上一步骤的解，在这一步骤进行对比，如果上一步骤的解<0则舍弃。最终得到这一步骤解，与之前步骤解的最大值res进行比较，保存当前的最优解。
        '''
        sum = 0
        res = nums[0]
        for num in nums:
            sum = sum+num if sum > 0 else num
            if res < sum:
                res = sum
        return res      
    def func4(self,nums):
        '''
        动态规划
        1、定义一个函数f(n)，以第n个数为结束点的子数列的最大和，存在一个递推关系f(n) = max(f(n-1) + A[n], A[n]);
        2、将这些最大和保存下来后，取最大的那个就是，最大子数组和。因为最大连续子数组 等价于 最大的以n个数为结束点的子数列和。
        '''
        if len(nums) == 0:
            return None
        dp = [0 for i in range(len(nums))]        
        dp[0] = nums[0]
        res = dp[0] 
        for i in range(1,len(nums)):
            dp[i] = max(nums[i], dp[i-1]+nums[i])
            res = max(dp[i], res)
        return res
    def getVarLen(self):
        co = self.func1.__code__        
        return co.co_argcount

ori_data = [[-2,1,-3,4,-1,2,1,-5,4],[1]]
expect_data = [6,1]
assert len(ori_data) == len(expect_data)

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
