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

'''

class getresultlist(object):

    def __init__(self):
        pass
    # 删除顶端
    def del_top(self, list=[]):
        list.pop(0)
        return list
    # 逆序
    def reverse(self,list=[]):
        return list.reverse()
    # 把列表的第一个元素放到最后一位
    def start_to_end(self, list=[]):
        if len(list) <= 1:
            pass
        else:
            list.append(list[0])
            list.pop(0)
        return list
    # 把列表的最后一个元素放到第一位
    def end_to_start(self, list=[]):
        if len(list) <= 1:
            pass
        else:
            list.insert(0, list[-1])
            list.pop()
        return list

    def get_ori_list(self,s):
        L = list(s)
        L.reverse()
        news = ''
        for i in range(len(L)):
            
        return None

s = input()
print(getresultlist().get_ori_list(s))
