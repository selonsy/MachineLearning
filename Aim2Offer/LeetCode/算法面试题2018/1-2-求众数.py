'''
给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在众数。

示例 1:

输入: [3,2,3]
输出: 3
示例 2:

输入: [2,2,1,1,1,2,2]
输出: 2
'''

class Solution:
    def majorityElement1(self, nums):
        nums_dic = {}
        for i in nums:
            if i not in nums_dic:
                nums_dic[i] = 1
            else:
                nums_dic[i] += 1
        list = sorted(nums_dic.items(), key=lambda d: d[1],reverse=True) 
        return list[0][0]
    def majorityElement2(self, nums):
        nums.sort()
        return nums[len(nums)//2]        

s = Solution()
_input = [2,2,1,1,1,2,2]
_output = s.majorityElement2(_input)
print(_output)