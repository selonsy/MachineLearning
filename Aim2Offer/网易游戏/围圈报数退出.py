'''

n个人围成一个圈，第一个人从0开始报数，报到（m-1）的退出，剩下的人继续从0开始报数。
最后剩下的人为胜者。指定n和m，求胜者编号。

'''

def my_func(n,m):    
    num_list = list(range(1,n+1))
    SayNo = 0
    del_count = 0
    while del_count < len(num_list)-1:
        for i in range(len(num_list)):
            if num_list[i] != 0:
                if SayNo == (m-1):
                    SayNo = 0
                    num_list[i]=0                    
                    del_count+=1
                else:
                    SayNo += 1   
    k=0          
    for i in num_list:
        if i!=0:
            return i
        else:
            k+=1
    if k ==n:
        return n

n,m = map(int,input().split())
print(my_func(n,m))


# 没有考虑10 1的情况