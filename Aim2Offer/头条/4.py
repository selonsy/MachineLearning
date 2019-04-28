'''
给定一个长度为n的 仅包含0,1的数列。例如1,0,0,1,1,1,0,1
我们可以轻易算出,它的最长全1区间长度是3。 
从数组的第4个数到第6个数.

现在,你获得了k次可以将某个位置上的0变为1的机会,但你可以不用完你的所有机会.
请你给出 ,你使用了你的变化机会后,这个数列的最长全1区间最大是多少.
输入描述:
输入第一行是2个整数n和x,代表数组长度和你的变化机会
输入第二行是n个整数,每个整数是0或1。 代表初始数列
数据保证1<=n <= 500000,0<= k<= 500000
输出描述:
输出包含一个整数,代表你的答案
'''

def myfunc(N,K,num):
    
    def checkInterval(possible,sumvalue):
        startmax = N - possible
        for start in range(0,startmax+1):
            end = start + possible
            if sum(num[start:end]) >= sumvalue:
                return True
        return False
    if(1 not in num):
        print(K)
    elif(K>=N-1):
        print(N)
    else:
        maxleng = 0
        tempmaxleng = 0
        for i in num:
            if(i == 1):
                tempmaxleng += 1
            else:
                if(tempmaxleng>maxleng):
                    maxleng = tempmaxleng
                tempmaxleng = 0
        sumvalue = maxleng
        possible = sumvalue+K
        while(True):
            if(not checkInterval(possible,sumvalue)):
                break 
            sumvalue += 1
            possible += 1
        print(possible -1)    

n,k = map(int,input().split())
arr = list(map(int,input().split()))
myfunc(n,k,arr)