

'''

网易大厦停车场里可以提供使用的电桩数量是S,1<=S<=100,现在有N台电动车等待充电,1<=N<=999,
第i台电动车所需要的充电时间为t_i,1<=i<=N,
S. N. t_i均为正整数
要求用最优的算法使得所有同事们在最短时间内都充好电,并输出所有人都充完电的用时.

输入描述:
输入为2行,第一行N S,分别为N台电动车和s个电桩,以空格分隔, 第二行为每台电动车充电所需的时长t_i,以空格分隔, 
如:
3 2
2 1 3
输出描述:
输出为1行,内容为所有人都充完电时所消耗的时长,如:
3

'''


def best_for_charge(n,s,arr):
    
    return None

n,s = map(int,input().split())
arr = list(map(int,input().split()))
print(best_for_charge(n,s,arr))