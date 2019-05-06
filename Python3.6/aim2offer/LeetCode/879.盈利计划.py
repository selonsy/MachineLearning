'''
帮派里有 G 名成员，他们可能犯下各种各样的罪行。

第 i 种犯罪会产生 profit[i] 的利润，它要求 group[i] 名成员共同参与。

让我们把这些犯罪的任何子集称为盈利计划，该计划至少产生 P 的利润。

有多少种方案可以选择？因为答案很大，所以返回它模 10^9 + 7 的值。

示例 1：
输入：G = 5, P = 3, group = [2,2], profit = [2,3]
输出：2
解释： 
至少产生 3 的利润，该帮派可以犯下罪 0 和罪 1 ，或仅犯下罪 1 。
总的来说，有两种方案。

示例 2:
输入：G = 10, P = 5, group = [2,3,5], profit = [6,7,8]
输出：7
解释：
至少产生 5 的利润，只要他们犯其中一种罪就行，所以该帮派可以犯下任何罪行 。
有 7 种可能的计划：(0)，(1)，(2)，(0,1)，(0,2)，(1,2)，以及 (0,1,2) 。

提示：
    1 <= G <= 100
    0 <= P <= 100
    1 <= group[i] <= 100
    0 <= profit[i] <= 100
    1 <= group.length = profit.length <= 100

'''
import time

class Solution:
    # profitableSchemes
    # def func1(self, G: int, P: int, group: List[int], profit: List[int]) -> int:
    def func1(self, G, P, group, profit):
        dic = {i:{j: 1 if (i == j and j == 0) else 0  for j in range(P + 1)} for i in range(G + 1)}
        for g, p in zip(group, profit):
            dic = {i:{j:(dic[i][j] + dic[i-g][max(0, j-p)]) if i >= g else dic[i][j] for j in range(P, -1, -1)}for i in range(G, -1, -1)}
        return sum(dic[key][P] for key in dic) % 1000000007
    
    def func2(self, G, P, group, profit):
        '''
        It's a variant of knapsack.
        Suppose dp[p][g] represents the number of schemes when profit is p and group size is g. So for a specific scheme (p, g), dp[i+p][j+g] += dp[i][j]. dp[0][0] is initialized as 1.
        We need to count the number backwards to avoid repeated counting. As for current (p, g), dp[i][j] is always updated later than dp[i+p][j+g] so that what is added to dp[i+p][j+g] is always the dp[i][j] from the previous iteration.
        Besides, final profit can exceed P, so we just count all dp[i>P] to dp[P]. And we can use trick like dp[min(P,i+p)][g+j] += dp[i][j]. While j+g should be strictly bounded within G.
        '''
        dp = [[1]+[0]*G] + [[0]*(G+1) for _ in range(P)]
        for p, g in zip(profit, group):
            for i in range(P,-1,-1):
                for j in range(G-g,-1,-1):
                    dp[min(P,i+p)][g+j] += dp[i][j]
        return sum(dp[P]) % (10**9+7)
    
    def getVarLen(self):
        co = self.func1.__code__        
        return co.co_argcount

input_data = [[5,3,[2,2],[2,3]], [10,5,[2,3,5],[6,7,8]]]
expect_data = [2, 7]
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
    begin = time.time() # ToDo：这里的计时有问题，需要改进
    data_length = len(expect_data)
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
