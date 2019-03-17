# 给定一个字符串，请你将字符串重新编码，将连续的字符替换成“连续出现的个数+字符”。比如字符串AAAABCCDAA会被编码成4A1B2C1D2A。

# 输入描述:
# 每个测试输入包含1个测试用例
# 每个测试用例输入只有一行字符串，字符串只包括大写英文字母，长度不超过10000。

# 输出描述:
# 输出编码后的字符串

# 输入例子1:
# AAAABCCDAA

# 输出例子1:
# 4A1B2C1D2A

def str_encode(_str):
    # temp = {}
    # newstr = ""
    # k=0
    # for s in _str:
    #     if s in temp.keys():
    #         temp[s] = temp[s]+1
    #     else:
    #         temp[s] = 1
    # for key in temp.keys():
    #     newstr = newstr + str(temp[key])+key
    # return newstr
    n=0
    temp = ''
    newstr = ''
    for k,v in enumerate(_str):
        if v == temp or temp=='':
            n+=1            
        else:
            newstr = newstr+str(n)+temp
            n=1            
        temp = v
    return newstr+str(n)+temp


s = input()
print(str_encode(s))

# s = "AAAABCCDAA"
# print(str_encode(s))