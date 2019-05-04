# 在一个N*N的数组中寻找所有横，竖，左上到右下，右上到左下，四种方向的直线连续D个数字的和里面最大的值

# 输入描述:
# 每个测试输入包含1个测试用例，第一行包括两个整数 N 和 D :
# 3 <= N <= 100
# 1 <= D <= N
# 接下来有N行，每行N个数字d:
# 0 <= d <= 100

# 输出描述:
# 输出一个整数，表示找到的和的最大值

# 输入例子1:
# 4 2
# 87 98 79 61
# 10 27 95 70
# 20 64 73 29
# 71 65 15 0

# 输出例子1:
# 193

def find_max_sum(arr,d):
    return 1


n ,d = map(int,input().split())
arr = [[0]*n for j in range(n)]
for i in range(n):
    arr[i] = list(map(int,input().split()))
print(find_max_sum(arr,d))