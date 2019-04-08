








def func(n,k):
    count = 0
    split = 0
    while n!=0:
        if n%2==0:
            n = n/2
            split+=1
            count+=1
        else:
            n-=1
            if n==0:
                count+=1
                break
            n/=2
            count+=2
    return count

#n,k= map(int,input().split())
n=15
k=4
print(func(n,k))