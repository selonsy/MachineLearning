'''
给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

示例 1:

输入: 123
输出: 321

 示例 2:

输入: -123
输出: -321

示例 3:

输入: 120
输出: 21

注意:

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−2^31,  2^31 − 1]。
请根据这个假设，如果反转后整数溢出那么就返回 0。
'''
class Solution:
    def reverse1(self, x: int) -> int: 
        if x == 0:
            return 0
        elif x > 0:
            res = str(x)[::-1]
        else:
            res = '-' + str(x)[-1:0:-1]
        res = int(res.lstrip('0'))
        if res > (2**31-1) or res < (-2**31):
            res = 0
        return res
    def reverse2(self, x: int) -> int:
        sum = 0 
        flag = True if x >= 0 else False
        x = abs(x)
        while x != 0:
            i = x % 10 # python的取模对正负数规则不一样，统一转换成正数处理
            sum = sum * 10 + i
            x = x // 10
        if -2**31 < sum < 2**31-1:
            return sum if flag else -sum
        return 0

s = Solution()

func = s.reverse2

input_data = [123,-123,120]
expect_data = [321,-321,21]

assert len(input_data)==len(expect_data)
for i in range(len(expect_data)):
    res = func(input_data[i]) 
    assert res == expect_data[i], "expected is = {0}, but actually is {1}".format(expect_data[i],res)
print("done")