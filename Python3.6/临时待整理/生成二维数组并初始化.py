
# 空矩阵
myList1 = [[]]
# (4,3),直接定义
myList2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0],[0,0,0]]
# (4,3),推导式定义
myList3 = [([0] * 3) for i in range(4)]
# (4,3),for循环定义
myList4 = [] 
for i in range(4): 
    myList4.append([0]*3)

# 使用numpy生成矩阵
import numpy as np
myList5 = np.zeros(shape=(4,3))

print("myList1\n",myList1)
print("myList2\n",myList2)
print("myList3\n",myList3)
print("myList4\n",myList4)
print("myList5\n",myList5) 


list = [0,0,0]
myList6 = list*4   # (1,12)
myList7 = [list]*4 # (4,3)

print("myList6\n",myList6)
print("myList7\n",myList7)

myList7[0][0] = 3  # (4,3)的第一列全部变成了3，
# 原因是：[list]*4 是4个指向list的指针，修改4个中间的任
# 意一个，都会导致4个一起改变。可以使用 myList8 的方式规避。
print("update myList7n",myList7)

myList8 = [[0,0,0] for i in range(4)]
print("myList8\n",myList8)

myList8[0][0] = 3
myList8[1][0] = 4
myList8[2][0] = 5

print("update myList8\n",myList8)

