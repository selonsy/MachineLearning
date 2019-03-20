

def minus_plus(s):
    num = ''
    t = ''
    nums = []
    ts = []
    for i in s:        
        if ord(i)>=48 and ord(i)<=57:
            num = num+i
            # if t==1:                
            # elif t==0:
        elif i=='+':            
            ts.append(1)
            nums.append(int(num))
            num=''
        elif i=='-':
            ts.append(0)
            nums.append(int(num))
            num=''
    nums.append(int(num))       
    sum = nums[0]
    for i in range(len(nums)-1):
        if(ts[i]==1):
            sum += nums[i+1]
        elif(ts[i]==0):
            sum -= nums[i+1]
    return sum
s = input()
print(minus_plus(s))