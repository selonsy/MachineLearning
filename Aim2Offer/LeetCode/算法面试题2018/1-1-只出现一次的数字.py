'''
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4
'''

class Solution:
    # 超时了
    def singleNumber1(self, nums):
        nums_set = set(nums)
        for i in nums_set:
            if nums.count(i)==1:
                return i
    # 尼玛通过了
    def singleNumber2(self, nums):
        num_dic = {}
        for i in nums:
            if i not in num_dic:
                num_dic[i]=1
            else:
                num_dic[i]=num_dic[i]+1
        return [k for k in num_dic.keys() if num_dic[k]==1][0]
    # 异或运算
    def singleNumber3(self, nums):
        '''
        有序列表 任何数与自身异或结果为0: a ^ a = 0
        有序列表 0与任何数异或结果不变: 0 ^ a = a
        有序列表 遵循交换律: a ^ b = b ^ a

        理解：将数用二进制表示，相同的数异或会清除自身数据信息。即两个一对就消除了，剩下的就是只出现一次的。
        这个题目可以推广到：数组中只有一个数出现了奇数次，如何找到这个数。
        '''
        a = 0
        for num in nums:
            a = a ^ num
        return a 

s = Solution()
_input = [2,1,2,1,4]
_output = s.singleNumber3(_input)
print(_output)
