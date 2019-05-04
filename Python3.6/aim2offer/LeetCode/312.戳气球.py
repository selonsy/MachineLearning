'''
有 n 个气球，编号为0 到 n-1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。每当你戳破一个气球 i 时，你可以获得 nums[left] * nums[i] * nums[right] 个硬币。 这里的 left 和 right 代表和 i 相邻的两个气球的序号。注意当你戳破了气球 i 后，气球 left 和气球 right 就变成了相邻的气球。

求所能获得硬币的最大数量。

说明:

你可以假设 nums[-1] = nums[n] = 1，但注意它们不是真实存在的所以并不能被戳破。
0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100
示例:

输入: [3,1,5,8]
输出: 167 
解释: nums = [3,1,5,8] --> [3,5,8] -->  [3,8]  -->  [8]  --> []
     coins =  3*1*5  +  3*5*8  +  1*3*8  + 1*8*1  = 167

'''

# https://leetcode.com/problems/burst-balloons/discuss/76243/Python-DP-N3-Solutions

'''
还是挺难的 虽然回顾了矩阵链乘 但是这个dp[i][j]含义是不包含i,j的i-j之间的数组获得的最大乘积，
所以在nums前后添加[1]。没想到，看了别人的分析才推出递推式。贴一个python的版本
'''
import pysnooper
# def maxCoins(self, nums: List[int]) -> int:

@pysnooper.snoop()
def maxCoins(nums):
    if len(nums) == 0:
        return 0
    if len(nums) < 2:
        return nums[0]
    nums = [1] + nums + [1]
    dp = [[0] * len(nums) for _ in range(len(nums))]
    for i in range(len(nums) - 1, -1, -1):
        for j in range(i + 2, len(nums)):
            for k in range(i + 1, j):
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
    # print(dp)
    return dp[0][-1]

print(maxCoins([3,1,5,8]))