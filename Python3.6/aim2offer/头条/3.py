

'''
给定N(可选作为埋伏点的建筑物数量),D(相距最远的两名特工的距离的最大值)，以及可选建筑物的坐标，计算这次行动中，小队有多少
种埋伏的策略。
注意：
1、两个特工不能埋伏在同一个地点。
2、三个特工是等价的，不能因特工之间互换位置而重复使用，即组合，不算排列。
示例：
输入:
4 3       # 4 = N, 3 = D   (1<=N>=1000000,1<=D<=1000000)
1 2 3 4   # 可选坐标        (可以看成一条数轴，[0,1000000]，从小到大排列)
输出:4    #  策略总数 
'''
def myfunc(n,d,arr):
    res = []
    for i in range(n):
        a = arr[i]
        for j in range(i+1,n):
            b = arr[j]
            if b-a>d:
                continue
            for k in range(j+1,n):
                c = arr[k]
                if c-b<=d and c-a<=d:
                    res.append([a,b,c])
    return len(res) % 99997867 

# n,d = map(int,input().split())
# arr = list(map(int,input().split()))
# n,d = 5,19
# arr = [1,10,20,30,50]
n,d = 4,3
arr = [1,2,3,4]
print(myfunc(n,d,arr))