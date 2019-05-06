'''
给定一个二进制字符串 S（一个仅由若干 '0' 和 '1' 构成的字符串）和一个正整数 N，如果对于从 1 到 N 的每个整数 X，
其二进制表示都是 S 的子串，就返回 true，否则返回 false。

示例 1：

输入：S = "0110", N = 3
输出：true

示例 2：

输入：S = "0110", N = 4
输出：false

提示：

    1 <= S.length <= 1000
    1 <= N <= 10^9
'''
import time
import math
class Solution:
    def func1(self, S: str, N: int) -> bool:
        for i in range(1,N+1):
            if str(bin(i)).replace("0b","") not in S:
                return False
        return True

    def func2(self, S: str, N: int) -> bool:
        x = pow(2,int(math.log(N+1)/math.log(2))-1) # k位长二进制数的后端包含了k-1位长的，去掉冗余
        for i in range(x,N+1):            
            if str(bin(i)).replace("0b","") not in S:
                return False
        return True

    def getVarLen(self):
        co = self.func1.__code__        
        return co.co_argcount

input_data = [["0110",3], ["0110",4]]
expect_data = [True, False]
assert len(input_data) == len(expect_data)

s = Solution()
funcs = []
func_len = 0
var_len = s.getVarLen() - 1 # exclude self
for key in Solution.__dict__.keys():
    if "func" in key:
        func_len += 1
if func_len == 1:
    funcs = [s.func1]
elif func_len == 2:
    funcs = [s.func1, s.func2]
for f in range(len(funcs)):
    func = funcs[f]
    # begin = time.time()
    begin = time.clock()
    data_length = len(expect_data)
    for i in range(data_length):
        if var_len == 1:
            res = func(input_data[i])
            assert res == expect_data[i], "func{0}({1}): expected = {2}, but actually = {3}".format(f+1,input_data[i],expect_data[i], res)
        elif var_len == 2:
            res = func(input_data[i][0],input_data[i][1])
            assert res == expect_data[i], "func{0}({1},{2}): expected = {3}, but actually = {4}".format(f+1,input_data[i][0],input_data[i][1],expect_data[i], res)
    # time.sleep(3)
    # end = time.time()
    end = time.clock()
    print(end,begin,end-begin)    
    print("func{0} : {1:.4f} ms".format(f+1, (end-begin)*1000/data_length))
print("done")
