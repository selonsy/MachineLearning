'''
现在有一组正整数序列a1，a2，，，，an，请计算出其中满足最大公约数为1的三个数ai，aj，ak的组合的数量。
其中，ai，aj，ak的范围为：1<=i<j<k<=n

输入：
8
1 2 3 4 5 6 7 8
输出：
52

注意：N的范围为 1 < N < 9999
'''

from itertools import combinations
def my_func(n,ns):    
    # def fun(x,y): 
    #     if x<y:
    #         x,y=y,x
    #     d=x%y
    #     while d>0:
    #         x,y=y,d
    #         d=x%y
    #     return y
    res = 0
    def fun(x,y):
        while x!=y:
            if x < y:
                x, y = y, x
            x=x-y
        return x
    for i in combinations(ns, 3):
        a,b,c = i       
        x1 = int(fun(fun(a,b),c)) #最大公约数
        if x1==1:
            res+=1
    return res

n = int(input())
ns = list(map(int,input().split()))
print(my_func(n,ns))


# #辗转相除法
# def fun(x,y): #fun函数：用辗转相除法计算最大公约数
#     if x<y:
#         x,y=y,x
#     d=x%y
#     while d>0:
#         x,y=y,d
#         d=x%y
#     return y
# x1=int(fun(fun(a,b),c)) #最大公约数

# print("最大公约数(辗转相除法)：",x1)

