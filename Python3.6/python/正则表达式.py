import re



'''
re.match(pattern, string, flags=0)
说明:re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。
pattern	匹配的正则表达式
string	要匹配的字符串。
flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。

修饰符	 描述
re.I	使匹配对大小写不敏感
re.L	做本地化识别（locale-aware）匹配
re.M	多行匹配，影响 ^ 和 $
re.S	使 . 匹配包括换行在内的所有字符
re.U	根据Unicode字符集解析字符。这个标志影响 \w, \W, \b, \B.
re.X	该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解。
'''
print(re.match('www', 'www.runoob.com'))         # 在起始位置匹配
print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配

# <_sre.SRE_Match object; span=(0, 3), match='www'>
# (0, 3)
# None

'''
group(num=0)	匹配的整个表达式的字符串，group() 可以一次输入多个组号，在这种情况下它将返回一个包含那些组所对应值的元组。
groups()	    返回一个包含所有小组字符串的元组，从 1 到 所含的小组号。
'''
line = "Cats are smarter than dogs"
matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)
if matchObj:
   print("matchObj.group() : " ), matchObj.group()
   print("matchObj.group(1) : "), matchObj.group(1)
   print("matchObj.group(2) : "), matchObj.group(2)
else:
   print ("No match!!")
# matchObj.group() :  Cats are smarter than dogs
# matchObj.group(1) :  Cats
# matchObj.group(2) :  smarter

'''
re.search(pattern, string, flags=0)
说明：re.search 扫描整个字符串并返回第一个成功的匹配。
re.match与re.search的区别
re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；
而re.search匹配整个字符串，直到找到一个匹配。
'''

'''
re.sub(pattern, repl, string, count=0, flags=0)
说明:用于替换字符串中的匹配项.
pattern : 正则中的模式字符串。
repl : 替换的字符串，也可为一个函数。
string : 要被查找替换的原始字符串。
count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
'''

'''
re.compile(pattern[, flags])
compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用。
'''
pattern = re.compile(r'\d+')                    # 用于匹配至少一个数字
m = pattern.match('one12twothree34four')        # 查找头部，没有匹配
print (m)
# None
m = pattern.match('one12twothree34four', 2, 10) # 从'e'的位置开始匹配，没有匹配
print (m)
# None
m = pattern.match('one12twothree34four', 3, 10) # 从'1'的位置开始匹配，正好匹配
print (m)                                         # 返回一个 Match 对象
# <_sre.SRE_Match object at 0x10a42aac0>
m.group(0)   # 可省略 0
# '12'
m.start(0)   # 可省略 0,起始位置,子串第一个字符的索引
# 3
m.end(0)     # 可省略 0,结束位置（子串最后一个字符的索引+1）
# 5
m.span(0)    # 可省略 0, (start(group), end(group))
# (3, 5)


'''
findall(string[, pos[, endpos]])
在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。
注意： match 和 search 是匹配一次 findall 匹配所有。
'''

pattern = re.compile(r'\d+')   # 查找数字
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)
 
print(result1)
print(result2)
# ['123', '456']
# ['88', '12']

'''
re.finditer(pattern, string, flags=0)
和 findall 类似，在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回。
'''

'''
re.split(pattern, string[, maxsplit=0, flags=0])
split 方法按照能够匹配的子串将字符串分割后返回列表
'''
import re
re.split('\W+', 'runoob, runoob, runoob.')
['runoob', 'runoob', 'runoob', '']
re.split('(\W+)', ' runoob, runoob, runoob.') 
['', ' ', 'runoob', ', ', 'runoob', ', ', 'runoob', '.', '']
re.split('\W+', ' runoob, runoob, runoob.', 1) 
['', 'runoob, runoob, runoob.']
re.split('a*', 'hello world')   # 对于一个找不到匹配的字符串而言，split 不会对其作出分割
['hello world']