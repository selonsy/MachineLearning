

'''
母牛从3-7岁初每年会生产1头小母牛,10岁后死亡(10岁仍然存活)。 
假设初始有1头刚出生的母牛,请问第n年有多少头母牛?(年从第一年开始计数)
注:
第3年初会出生第一头牛,故第三年有两头母牛
第5年初第3年出身的牛会生产,故第五年有5头母牛。岁数是虚岁
输入描述:
输入一个正整数表示n年
输出描述:
一个数字,表示第n年有多少头牛
'''
global a
a = 1
def myfunc(s,n):
    for i in range(s,n):
        if i>=3 and i<=7:
            global a
            a+=1
            myfunc(1,n-i+1)
        if i>10:            
            a-=1
            return None
    return None
n = int(input())
# n = 12  输出123
myfunc(1,n+1)
print(a)