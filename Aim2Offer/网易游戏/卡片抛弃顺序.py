'''
题目描述
假设有一沓卡片,每张卡片上写着一个字符：
我们重复以下步骤：
a.取出最顶上的一张卡牌,抛弃
b.如果剩余卡牌数量是偶数，则翻转整沓卡牌
c.把此时最顶上的一张卡牌，放到这背卡片最后
一直重复此步骤，直到手中没有扑克牌
例如abccd的抛弃顺序为acbdc
现在给出抛弃顺序,求原先的顺序

解题思路：
首先看抛弃的顺序：
order | a    | b    | c    | abandon
abccd | bccd | dccb | ccbd | a
ccbd  | cbd  | cbd  | bdc  | c
bdc   | dc   | cd   | dc   | b
dc    | c    | c    | c    | d
c     | -    | -    | -    | c
总结以上顺序，可以知道恢复顺序为逆序还原。
首先将抛弃顺序 acbdc 逆序为 cdbca(待求序列).设恢复序列为s,设为空.
然后执行以下步骤：
a.从待求中取下一个值并插入到s的前面。如果它是最后一个值，则停止运算并退出。
b.将最后面的牌放到第一个。
c.如果s中的卡牌数量是偶数，则反转整沓卡牌。
具体的执行顺序为：
s     | a    | b    | c    |
c     | c    | c    | c    |
c     | dc   | cd   | dc   |
dc    | bdc  | cbd  | cbd  |
cbd   | ccbd | dccb | bccd |
bccd  | abccd| stop |
'''

class getresultlist(object):
    def __init__(self):
        pass       
    # 把列表的第一个元素放到最后一位
    def start_to_end(self, L=[]):
        if len(L) <= 1:
            pass
        else:
            L.append(list[0])
            L.pop(0)
        return L
    # 把列表的最后一个元素放到第一位
    def end_to_start(self, L=[]):
        if len(L) <= 1:
            pass
        else:
            L.insert(0, L[-1])
            L.pop()
        return L
    def get_ori_list(self,s):
        L = list(s)
        L.reverse()
        ori = []
        for i in range(len(L)):           
            ori.insert(0,L[i])
            if i==len(L)-1:
                break
            ori = getresultlist().end_to_start(ori)
            if len(ori)%2==0:
                ori.reverse()            
        return ori

s = input()
print(getresultlist().get_ori_list(s))
