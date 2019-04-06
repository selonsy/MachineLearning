
'''
题目描述
某大型游乐场引进了一个弹射装置,最多可同时将2名游客弹射到位于高处的着陆区. 
为了确保安全,该项目的工作人员为每一个排队的游客标记了权重,体重越大,权重越高;
同时设定了毎次弹射的权重上限,即如果2名游客的权重之和超过了权重上限,那么只能安排1名游客弹射,另1名游客等候下一次弹射.
现在给出排队参与游戏的所有游客的权重和弹射的权重上限,请您帮工作人员计算出最坏情况下所需的弹射次数?
输入描述:
第一行输入两个整数,N和w,以空格分隔.
其中,N表示排队参与游戏的游客总数,w表示每次弹射的权重上限.(1<= N<= 10^5, 1 <= w <= 10^9)
第二行输入N个整数,以空格分隔,分别代表N个游客中每个人权重值wi.(1 <= wi <= w)

输出描述:
一行输出,为最坏情况下所需的弹射次数。

示例1 
输入输出示例仅供调试,后台判题教据一般不包含示例
输入
6 6 
2 3 3 3 4 5
输出5
例如，其中一种最坏情况下的排队序列为：
3 3 2 5 3 4 

分析：
数据量太大了，复杂度必须保证在O(N)或者O(NlogN)，否则肯定超时

'''

from itertools import permutations

# 自己的算法，复杂度太高，超时了：不应该用python的排列组合
def my_func(n,w,wlist):    
    res = 0
    def fun(arr,n,w):
        arr.reverse()
        count = 0
        while arr:
            x1 = arr.pop()
            if len(arr):
                x2 = arr.pop()
            else:
                x2 = 0
            if x1 + x2 <= w:
                count+=1
            else:
                count+=1
                arr.append(x2)
        return count                
    for i in permutations(wlist, n):
        arr = list(i)    
        x1 = fun(arr,n,w) 
        if x1 > res:
            res = x1
    return res

# 大佬的C++版本转python
def boss_func(n,w,wlist):
    wlist.sort()
    l = 0
    r = n - 1
    ans = 0
    while l < r:
        if(wlist[l] + wlist[r] > w):
            ans+=2
            l+=1
            r-=1
        else:
            l+=1
    ans+= (n - ans) / 2 + (n - ans) % 2
    return int(ans)

#n,w = list(map(int,input().split()))
#wlist = list(map(int,input().split()))
#print(boss_func(n,w,wlist))

n,w = 6,6
wlist = [2,3,3,3,4,5]
print(boss_func(n,w,wlist))

