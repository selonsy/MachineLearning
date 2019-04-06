'''
鸡蛋掉落
你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。
每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。
你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。
每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。
你的目标是确切地知道 F 的值是多少。
无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？

示例 1：
输入：K = 1, N = 2
输出：2
解释：
鸡蛋从 1 楼掉落。如果它碎了，我们肯定知道 F = 0 。
否则，鸡蛋从 2 楼掉落。如果它碎了，我们肯定知道 F = 1 。
如果它没碎，那么我们肯定知道 F = 2 。
因此，在最坏的情况下我们需要移动 2 次以确定 F 是多少。

示例 2：
输入：K = 2, N = 6
输出：3

示例 3：
输入：K = 3, N = 14
输出：4
 
提示：
1 <= K <= 100
1 <= N <= 10000
'''

# 动态规划问题：Dynamic programming，简称DP
#动态规划解决问题的过程分为两步：
#1.寻找状态转移方程式
#2.利用状态转移方程式自底向上求解问题

#如何用最少的次数测出鸡蛋会在哪一层摔碎？ - 刘遥行的回答 - 知乎
#https://www.zhihu.com/question/19690210/answer/514119129

class Solution:   
    def superEggDrop0(self, K, N):
        '''
        dp[k][m] 的含义是k个鸡蛋 移动m次最多能够确定多少楼层
        这个角度思考
        dp[k][m] 最多能够确定的楼层数为L
        那么我选定第一个扔的楼层之后，我要么碎，要么不碎
        这就是把L分成3段
        左边是碎的那段 长度是dp[k][m - 1]
        右边是没碎的那段 长度是dp[k-1][m - 1] 因为已经碎了一个了
        中间是我选定扔的楼层 是1
        所以递推公式是
        dp[k][m] = dp[k - 1][m - 1] + dp[k][m - 1] + 1

        根据递推公式 如果采用k倒着从大到小计算 就可以只存一行的dp[k] 直接原地更新dp[k] 不影响后续计算 
        只需要O(K)空间复杂度 O(KlogN) 鸡蛋完全够用的时候 就是走LogN步 最差情况是1个鸡蛋走N步 O(KN)
        '''
        dp = [0] * (K + 1)
        m = 0
        while dp[K] < N:
            m += 1
            for k in range(K, 0, -1):
                # print(m, k)
                dp[k] = dp[k - 1] + dp[k] + 1
        return m
    # 递归版本：超时啦啦啦啦啦啦 (3,25)
    def superEggDrop1(self, K, N):
        '''
            K: int 鸡蛋, N: int 楼层
            rType:iKt
        分析:
        在这里,我们把N层楼/K个鸡蛋的问题,抽象成一个黑盒子函数F(K,N)
        楼层数N和鸡蛋数K是函数的两个参数,函数的返回值是最优解的最大尝试次数

        假设我们第一个鸡蛋扔出的位置在第X层(1<=X<=N)，会出现两种情况：
        1.第一个鸡蛋没碎
        那么剩余的N-X层楼，剩余K个鸡蛋，可以转变为下面的函数：
        F(K,N-X) + 1 , 1 <= X <= N
        2.第一个鸡蛋碎了
        那么只剩下从1层到X-1层楼需要尝试，剩余的鸡蛋数量是K-1，可以转变为下面的函数：
        F(K-1,X-1) + 1 , 1 <= X <= N
        整体而言，我们要求出的是 N层楼 / K个鸡蛋 条件下，最大尝试次数最小的解，所以这个题目的状态转移方程式如下：
        X可以为1......N,所以有N个Max(F(N-X，K)+ 1,F(X-1,K-1) + 1)的值,最终F(K,N)是这N个值中的最小值,即最优解
        F(K,N) = Min(Max(F(K,N-X) + 1,F(K-1,X-1) + 1)), 1 <= X <= N
        '''
        if N == 0:
            return 0
        if K == 1:
            return N

        ans = min([max([self.superEggDrop(K, N - i),self.superEggDrop(K - 1, i - 1)]) for i in range(1, N + 1)]) + 1
        return ans
    # 循环版本,再次超时 (4,2000)
    def superEggDrop2(self, K, N):
        '''
            K: int 鸡蛋, N: int 楼层
            rType:iKt
        分析:
        '''
        if N < 1 and K < 1:
                return 0
        f = [ [ i for i in range(N+1) ] for j in range(K+1) ]
        for K_i in range( 2, K + 1 ):
            for N_j in range( 1, N + 1 ):
                for k in range( 1, N_j ):
                    f[K_i][N_j]= min ( f[K_i][N_j], 1 + max( f[K_i -1][k - 1], f[K_i][N_j - k]) )
        return f[K][N]

s = Solution()
x = s.superEggDrop0(4,2000)
print(x)
