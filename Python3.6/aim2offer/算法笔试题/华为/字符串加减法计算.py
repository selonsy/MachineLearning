#题目描述
#要开发一款教育类App,帮助幼儿在识数阶段做一百以内自然数[0,99]的加减法.
#屏幕上会显示1234567890+-=这些按钮,用户在按了若干按钮之后,如果按了=,则会把按=之前的字符串作为一个算式,计算结果.
#中间结果或最后结果可以为负数.
#输入描述:
#输入为一个字符串,形如"23+86-6+37+24-8-13".
#输入字符串中保证:
#1,
#不会包含
#除"1""2""3""4""5""6""7""8""9""0""+""-"之外的字符。
#2.
#长度不为0

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