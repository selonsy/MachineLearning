
# python 3.6
# author:selonsy
# 用代码的形式来学习语法，持续更新 
# 目录：list/set/tuple/dic

'''
list
'''
students = ['a','b','c','d','e']
students[1]                     # 'b',下标从0开始
students[-1]                    # 'e',-2表示倒数第二个，以此类推
len(students)                   # 5,数组的长度
students.append('f') 
# ['a', 'b', 'c', 'd', 'e', 'f']
# 默认追加到数组的末尾
students.insert(0,'z')
# ['z', 'a', 'b', 'c', 'd', 'e', 'f']
# 往索引为0的位置插入元素，其余依次后移
students.extend(['x'])          # 将一个列表中每个元素分别添加到另一个列表中，只接受一个参数；
                                # 相当于是将list B 连接到list A上
students_temp = students + ['k','l'] # 两个list相加，效果类似于extend，但是会创建新的list对象
                                     # 需要消耗额外的内存，list较大时，考虑用extend。
x = students.pop()              # f,删除list末尾元素并返还
x = students.pop(1)             # a,删除list指定位置的元素并返还
# ['z', 'b', 'c', 'd', 'e']
del students[1]                 # 删除list指定位置的元素，不返还
# ['z', 'c', 'd', 'e']
students.remove('c')            # 删除具有指定值的元素(仅匹配第一个)，不返还
# ['z', 'd', 'e']
students[0] = 'e'               # 替换某个位置的元素
# ['e', 'd', 'e']
x = students.count('e')         # 2,统计某个值出现的次数
students.sort()                 # 排序
# ['d', 'e', 'e']
# list.sort(cmp=None, key=None, reverse=False)
students.reverse()              # 反序
# ['e', 'e', 'd']


'''
tuple
'''
students = (1,) 
# (1,) 
# 若元素唯一，则需加逗号,来消除歧义。避免被认为是小括号。
students = (1,2,3) 
# 小括号，元素用逗号隔开
students = ('123',1,2,3)
# 元素的数据类型可以不同
temp = (4,['A','B'])
students = students + temp
# ('123', 1, 2, 3, 4, ['A','B'])
# 元组可以连接结合，即相加
del temp 
# 使用del语句来删除整个元组,但是不允许删除元组中的元素。
L = students[-1]
L[0] = 'B'
# ('123', 1, 2, 3, 4, ['B', 'B'])
# tuple指向不变，但指向本身可变


'''
dict
'''
students = {'a':98,'c':96,'b':100} # 定义，键值对(key:value)
# students["d"]                    # KeyError: 'd',key不存在即报错
students["a"] = 97                 # 'a':97,后面赋值会覆盖前面的值
print('a' in students)             # True,判断key是否存在
x = students.get('c')              # 96，get获取指定key的value
x = students.get('d')              # None,key不存在返回None
x = students.get('d',-1)           # -1,不存在返回自己指定的值
x = students.pop('b')              # 删除一个key并返回其值
students.clear()                   # 清空词典所有条目
del students                       # 删除词典


'''
set
'''
students = set([1,2,3,4,5])        # 使用list进行创建
# {1, 2, 3, 4, 5}                  # 花括号
students.add(6)                    # 添加元素
# {1, 2, 3, 4, 5, 6}
students.add(1)                    # 添加重复元素
# {1, 2, 3, 4, 5, 6}               # 可以成功，但无效
students.remove(5)                 # 删除元素
# {1, 2, 3, 4, 6}
temp = set([3,4,5,6])
print(students & temp)
# {3, 4, 6}                        # 数学意义上的交集
print(students.intersection(temp)) # 效果同上
print(students | temp)             # 并集
# {1, 2, 3, 4, 5, 6}
print(students.union(temp))        # 效果同上
print(students - temp)             # 差集
# {1, 2}
print(students.difference(temp))   # 效果同上

print(x)
print(students)