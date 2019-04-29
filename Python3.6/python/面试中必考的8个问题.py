# 1、下面这段代码的输出结果是什么？请解释。

def extendList(val, list=[]):
    list.append(val)
    return list

list1 = extendList(10)
list2 = extendList(123,[])
list3 = extendList('a')

print("list1 = %s" % list1) # list1 = [10, 'a']
print("list2 = %s" % list2) # list2 = [123]
print("list3 = %s" % list3) # list3 = [10, 'a']

# 产生上述情况的原因是：默认列表只在函数被定义的那一刻创建一次。因此list1和list3是在同一个默认列表上进行操作（计算）的。
# 而list2是在一个分离的列表上进行操作（计算）的。（通过传递一个自有的空列表作为列表参数的数值）。

# 如果想要得到预期的输出，即[10]/[123]/['a']的话，可以进行如下的修改：
def extendList(val, list=None):
    if list is None:
        list = []
    list.append(val)
    return list


################################################################################################################
#                                                                                                              #
################################################################################################################

# 2.下面这段代码的输出结果将是什么？请解释。
def multipliers():
  return [lambda x : i * x for i in range(4)]
print(type(multipliers()))              # <class 'list'>
print(type(multipliers()[0]))           # <class 'function'>
print([m(2) for m in multipliers()])    # [6, 6, 6, 6]

# 而预期输出：[0,2,4,6]

'''
产生上述结果的原因是：Python闭包的延迟绑定。这意味着内部函数被调用时，参数的值在闭包内进行查找。

因此，当任何由multipliers()返回的函数被调用时，i的值将在附近的范围进行查找。那时，不管返回的函数是否被调用，for循环已经完成，i被赋予了最终的值3。

因此，每次返回的函数乘以传递过来的值3，因为上段代码传过来的值是2，它们最终返回的都是6(3*2)。

碰巧的是，《The Hitchhiker’s Guide to Python》也指出，在与lambdas函数相关也有一个被广泛被误解的知识点，不过跟这个case不一样。
由lambda表达式创造的函数没有什么特殊的地方，它其实是和def创造的函数式一样的。
'''

# 为了解决上面的问题，可以采用如下的方式：
# 一种解决方法就是用Python生成器。
def multipliers():
    for i in range(4): yield lambda x : i * x
print([m(2) for m in multipliers()])    # [0, 2, 4, 6]
# 另外一个解决方案就是创造一个闭包，利用默认函数立即绑定。
def multipliers():
    return [lambda x, i=i : i * x for i in range(4)]
print([m(2) for m in multipliers()])    # [0, 2, 4, 6]
# 还有种替代的方案是，使用偏函数：
from functools import partial
from operator import mul
def multipliers():
    return [partial(mul, i) for i in range(4)]
print([m(2) for m in multipliers()])    # [0, 2, 4, 6]


################################################################################################################
#                                                                                                              #
################################################################################################################

# 3.下面这段代码的输出结果将是什么？请解释。
class Parent(object):
    x = 1
class Child1(Parent):
    pass
class Child2(Parent):
    pass
print(Parent.x, Child1.x, Child2.x)  # 1 1 1
Child1.x = 2
print(Parent.x, Child1.x, Child2.x)  # 1 2 1
Parent.x = 3
print(Parent.x, Child1.x, Child2.x)  # 3 2 3

# 最后一行输出为什么是3 2 3 而不是 3 2 1.为什么在改变parent.x的同时也改变了child2.x的值？但与此同时没有改变Child1.x的值？

'''
此答案的关键是，在Python中，类变量在内部是以字典的形式进行传递。

如果一个变量名没有在当前类下的字典中发现。则在更高级的类（如它的父类）中尽心搜索直到引用的变量名被找到。
（如果引用变量名在自身类和更高级类中没有找到，将会引发一个属性错误。）

因此,在父类中设定x = 1,让变量x(带有值1)能够在其类和其子类中被引用到。这就是为什么第一个打印语句输出结果是1 1 1

因此，如果它的任何一个子类被覆写了值（例如说，当我们执行语句Child1.x = 2）,这个值只在子类中进行了修改。
这就是为什么第二个打印语句输出结果是1 2 1

最终，如果这个值在父类中进行了修改，（例如说，当我们执行语句Parent.x = 3）,这个改变将会影响那些还没有覆写子类的值
（在这个例子中就是Child2）这就是为什么第三打印语句输出结果是3 2 3
'''


################################################################################################################
#                                                                                                              #
################################################################################################################

# 4.下面这段代码在Python2下输出结果将是什么？请解释。
def div1(x,y):
    print("%s/%s = %s" % (x, y, x/y))

def div2(x,y):
    print("%s//%s = %s" % (x, y, x//y))

div1(5,2)   # 2.5 in python3 2    in python2
div1(5.,2)  # 2.5 in python3 2.5  in python2
div2(5,2)   # 2   in python3 2    in python2
div2(5.,2.) # 2.0 in python3 2.0  in python2

'''
//操作符将总是执行整形除法，不管操作符的类型.
默认情况下，Python 2 自动执行整形计算如果两者都是整数。因此,5/2 结果是2，而5./2结果是2.5
在 Python 3 中，/ 操作符是做浮点除法，而 // 是做整除
（即商没有余数，比如 10 // 3 其结果就为 3，余数会被截除掉，而 (-7) // 3 的结果却是 -3。）
'''


################################################################################################################
#                                                                                                              #
################################################################################################################

# 5、下面代码的输出结果将是什么？
list = ['a','b','c','d','e']
print(list[10:])   # []

'''
上面的代码将输出[],不会产生IndexError错误。
但是，尝试获取list[10]和之后的成员，会导致IndexError。
这成为特别让人恶心的疑难杂症，因为运行的时候没有错误产生，导致bug很难被追踪到。
'''

################################################################################################################
#                                                                                                              #
################################################################################################################

# 6、考虑下列代码片段：
list = [ [ ] ] * 5             # 创建了包含对同一个列表五次引用的列表,即5个列表是引用的同一个列表
print(list)  # output?         我的答案:[[],[],[],[],[]]          实际的答案:[[],[],[],[],[]]
list[0].append(10)      
print(list)  # output?         我的答案:[[10],[],[],[],[]]        实际的答案:[[10], [10], [10], [10], [10]]
list[1].append(20)      
print(list)  # output?         我的答案:[[10],[20],[],[],[]]      实际的答案:[[10, 20], [10, 20], [10, 20], [10, 20], [10, 20]]
list.append(30)         
print(list)  # output?         我的答案:[[10],[20],[],[],[],30]   实际的答案:[[10, 20], [10, 20], [10, 20], [10, 20], [10, 20], 30]

'''
第一行的输出结果直觉上很容易理解，例如 list = [ [ ] ] * 5 就是简单的创造了5个空列表。
然而，理解表达式list=[ [ ] ] * 5的关键一点是它不是创造一个包含五个独立列表的列表，
而是它是一个创建了包含对同一个列表五次引用的列表。##########
只有了解了这一点，我们才能更好的理解接下来的输出结果。

list[0].append(10) 将10附加在第一个列表上。

但由于所有5个列表是引用的同一个列表，所以这个结果将是：
[[10], [10], [10], [10], [10]]
同理，list[1].append(20)将20附加在第二个列表上。但同样由于5个列表是引用的同一个列表，所以输出结果现在是：

[[10, 20], [10, 20], [10, 20], [10, 20], [10, 20]]
作为对比， list.append(30)是将整个新的元素附加在外列表上，因此产生的结果是： [[10, 20], [10, 20], [10, 20], [10, 20], [10, 20], 30].
'''

################################################################################################################
#                                                                                                              #
################################################################################################################

# 7、Given a list of N numbers。
'''
给定一个含有N个数字的列表。

使用单一的列表生成式来产生一个新的列表，该列表只包含满足以下条件的值：

(a)偶数值
(b)元素为原始列表中偶数切片。

例如，如果list[2]包含的值是偶数。那么这个值应该被包含在新的列表当中。因为这个数字同时在原始列表的偶数序列（2为偶数）上。
然而，如果list[3]包含一个偶数，那个数字不应该被包含在新的列表当中，因为它在原始列表的奇数序列上。
'''
# 例如，给定列表如下：
list = [ 1 , 3 , 5 , 8 , 10 , 13 , 18 , 36 , 78 ]
print([x for x in list[::2] if x%2 == 0]) # [10, 18, 78]
# 这个表达式工作的步骤是，第一步取出偶数切片的数字，第二步剔除其中所有奇数。


################################################################################################################
#                                                                                                              #
################################################################################################################

# 8、给定以下字典的子类，下面的代码能够运行么？为什么？
class DefaultDict(dict):
    def __missing__(self, key):
        return []
d = DefaultDict()
d['florp'] = 127

'''
当key缺失时，执行DefaultDict类，字典的实例将自动实例化这个数列。
'''

# 默认值可以很方便

# 众所周知，在Python中如果访问字典中不存在的键，会引发KeyError异常。但是有时候，字典中的每个键都存在默认值是非常方便的。例如下面的例子：

strings = ('puppy', 'kitten', 'puppy', 'puppy',
    'weasel', 'puppy', 'kitten', 'puppy')
counts = {}
for kw in strings:
    counts[kw] += 1
# 该例子统计strings中某个单词出现的次数，并在counts字典中作记录。
# 单词每出现一次，在counts相对应的键所存的值数字加1。但是事实上，运行这段代码会抛出KeyError异常，
# 出现的时机是每个单词第一次统计的时候，因为Python的dict中不存在默认值的说法，可以在Python命令行中验证：

# >>> counts = dict()
# >>> counts
# {}
# >>> counts['puppy'] += 1
# Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
# KeyError: 'puppy'

# 使用判断语句检查
# 既然如此，首先可能想到的方法是在单词第一次统计的时候，在counts中相应的键存下默认值1。这需要在处理的时候添加一个判断语句：

strings = ('puppy', 'kitten', 'puppy', 'puppy',
    'weasel', 'puppy', 'kitten', 'puppy')
counts = {}
for kw in strings:
    if kw not in counts:
        counts[kw] = 1
    else:
        counts[kw] += 1
# counts:
# {'puppy': 5, 'weasel': 1, 'kitten': 2}
# 使用 dict.setdefault() 方法
# 也可以通过 dict.setdefault() 方法来设置默认值：

strings = ('puppy', 'kitten', 'puppy', 'puppy',
    'weasel', 'puppy', 'kitten', 'puppy')
counts = {}
for kw in strings:
    counts.setdefault(kw, 0)
    counts[kw] += 1
# dict.setdefault() 方法接收两个参数，第一个参数是健的名称，第二个参数是默认值。
# 假如字典中不存在给定的键，则返回参数中提供的默认值；反之，则返回字典中保存的值。
# 利用 dict.setdefault() 方法的返回值可以重写for循环中的代码，使其更加简洁：

strings = ('puppy', 'kitten', 'puppy', 'puppy',
    'weasel', 'puppy', 'kitten', 'puppy')
counts = {}
for kw in strings:
    counts[kw] = counts.setdefault(kw, 0) + 1

# 利用 collections.defaultdict 来解决最初的单词统计问题，代码如下：

from collections import defaultdict
strings = ('puppy', 'kitten', 'puppy', 'puppy',
    'weasel', 'puppy', 'kitten', 'puppy')
counts = defaultdict(lambda: 0) # 使用lambda来定义简单的函数
for s in strings:
    counts[s] += 1

# defaultdict类就好像是一个dict，但是它是使用一个类型来初始化的：

# >>> from collections import defaultdict
# >>> dd = defaultdict(list)
# >>> dd
# defaultdict(<type 'list'>, {})
# defaultdict类的初始化函数接受一个类型作为参数，当所访问的键不存在的时候，可以实例化一个值作为默认值：

# >>> dd['foo']
# []
# >>> dd
# defaultdict(<type 'list'>, {'foo': []})
# >>> dd['bar'].append('quux')
# >>> dd
# defaultdict(<type 'list'>, {'foo': [], 'bar': ['quux']})
# 需要注意的是，这种形式的默认值只有在通过 dict[key] 或者 dict.__getitem__(key) 访问的时候才有效