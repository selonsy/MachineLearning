
import copy
import random
import time


class puzzled:

    # 初始化
    def __init__(self, puzzled):
        self.puzzled = puzzled
        self.__getPuzzledInfo()

    # 获取空白格的位置(zeroX,zeroy)
    def __getPuzzledInfo(self):
        self.puzzledWid = len(self.puzzled[0])
        self.puzzledHei = len(self.puzzled)  # ?
        self.__f1 = False
        for i in range(0, self.puzzledHei):  # 0<=x<self.puzzledHei
            for j in range(0, self.puzzledWid):
                if(self.puzzled[i][j] == 0):
                    self.zeroX = j
                    self.zeroY = i
                    self.__f1 = True
                    break
            if(self.__f1):
                break

    # 打印迷宫
    def printPuzzled(self):
        for i in range(0, len(self.puzzled)):
            print(self.puzzled[i])
        print("")

    # 判断是否达到了最终目标
    # eg:
    '''
    1 2 3
    4 5 6
    7 8 0
    '''

    def isRight(self):
        if(self.puzzled[self.puzzledHei-1][self.puzzledWid-1] != 0):
            return False
        for i in range(0, self.puzzledHei):
            for j in range(0, self.puzzledWid):
                if(i*self.puzzledWid+j+1 != self.puzzled[i][j]):
                    if(i != self.puzzledHei-1 or j != self.puzzledWid-1):
                        return False
        return True

    # 移动数字
    def move(self, dere):  # 0 up,1 down,2 left,3 right
        if(dere == 0 and self.zeroY != 0):
            # 以下交换同时进行，不会错误覆盖
            self.puzzled[self.zeroY-1][self.zeroX], self.puzzled[self.zeroY][self.zeroX] = self.puzzled[self.zeroY][self.zeroX], self.puzzled[self.zeroY-1][self.zeroX]
            self.zeroY -= 1
            return True

        elif(dere == 1 and self.zeroY != self.puzzledHei-1):
            self.puzzled[self.zeroY+1][self.zeroX], self.puzzled[self.zeroY][self.zeroX] = self.puzzled[self.zeroY][self.zeroX], self.puzzled[self.zeroY+1][self.zeroX]
            self.zeroY += 1
            return True

        elif(dere == 2 and self.zeroX != 0):
            self.puzzled[self.zeroY][self.zeroX -
                                     1], self.puzzled[self.zeroY][self.zeroX] = self.puzzled[self.zeroY][self.zeroX], self.puzzled[self.zeroY][self.zeroX-1]
            self.zeroX -= 1
            return True

        elif(dere == 3 and self.zeroX != self.puzzledWid-1):
            self.puzzled[self.zeroY][self.zeroX +
                                     1], self.puzzled[self.zeroY][self.zeroX] = self.puzzled[self.zeroY][self.zeroX], self.puzzled[self.zeroY][self.zeroX+1]
            self.zeroX += 1
            return True
        return False

    # 获取可以移动的方向
    def getAbleMove(self):
        a = []
        if(self.zeroY != 0):
            a.append(0)
        if(self.zeroY != self.puzzledHei-1):
            a.append(1)
        if(self.zeroX != 0):
            a.append(2)
        if(self.zeroX != self.puzzledWid-1):
            a.append(3)
        return a

    # 克隆迷宫
    def clone(self):
        a = copy.deepcopy(self.puzzled)
        return puzzled(a)

    # 打印成一行
    def toString(self):
        a = ""
        for i in range(0, self.puzzledHei):
            for j in range(0, self.puzzledWid):
                a += str(self.puzzled[i][j])
        return a

    # 判断两个迷宫是否相等
    def isEqual(self, p):
        if(self.puzzled == p.puzzled):
            return True
        return False

    # 检查是否可以达到目标状态
    def check(self):
        # 目标状态的逆序数为0，为偶排列(逆序数不包括0)
        y = 0
        for i in range(0, 3):
            for j in range(0, 3):
                for m in range(0, i+1):
                    for n in range(0, j+1):
                        if self.puzzled[m][n] != 0 and self.puzzled[i][j] != 0 and self.puzzled[m][n] > self.puzzled[i][j]:
                            y += 1
                            # print('%d > %d' % (self.puzzled[m][n], self.puzzled[i][j]))
        return (y % 2) == 0

    # 转换成一维数组
    def toOneDimen(self):
        a = []
        for i in range(0, self.puzzledHei):
            for j in range(0, self.puzzledWid):
                a.append(self.puzzled[i][j])
        return a

    # 获取不在最终位置的数字的个数
    def getNotInPosNum(self):
        t = 0
        for i in range(0, self.puzzledHei):
            for j in range(0, self.puzzledWid):
                if(self.puzzled[i][j] != i*self.puzzledWid+j+1):
                    if(i == self.puzzledHei-1 and j == self.puzzledWid-1 and self.puzzled[i][j] == 0):
                        continue
                    t += 1
        return t

    # 获取不在最终位置的数字，要移动到最终位置，所走的步数
    def getNotInPosDis(self):
        t = 0
        it = 0
        jt = 0
        for i in range(0, self.puzzledHei):
            for j in range(0, self.puzzledWid):
                if(self.puzzled[i][j] != 0):
                    it = (self.puzzled[i][j]-1) / self.puzzledWid
                    jt = (self.puzzled[i][j]-1) % self.puzzledWid
                else:
                    it = self.puzzledHei-1
                    jt = self.puzzledWid-1
                t += abs(it-i)+abs(jt-j)
        return t

    # 生成一个随机的m*n的迷宫
    @staticmethod
    def generateRandomPuzzle(m, n, ran):

        tt = []
        for i in range(0, m):
            t = []
            for j in range(0, n):
                t.append(j+1+i*n)
            tt.append(t)
        tt[m-1][n-1] = 0
        a = puzzled(tt)
        i = 0
        # 使用自定义方法打乱迷宫
        while(i < ran):
            i += 1
            a.move(random.randint(0, 4))
        return a


""" test puzzled class

p1 = puzzled.generateRandomPuzzle(3,3,4)
p1.printPuzzled()
p2 = puzzled.generateRandomPuzzle(3,3,4)
p2.printPuzzled()
print(p1.getNotInPosNum())
print(p1.getAbleMove())
print(p2.toOneDimen())
print(p2.toString())
print(p1.isEqual(p2))

[1, 2, 0]
[4, 5, 3]
[7, 8, 6]

[1, 2, 3]
[0, 5, 6]
[4, 7, 8]

3
[1, 2]
[1, 2, 3, 0, 5, 6, 4, 7, 8]
123056478
False

"""


class node:

    # 初始化
    def __init__(self, p):
        self.puzzled = p
        self.childList = []
        self.father = None

    # 增加子节点
    def addChild(self, child):
        self.childList.append(child)
        child.setFather(self)

    # 获取子节点列表
    def getChildList(self):
        return self.childList

    # 设置父节点
    def setFather(self, fa):
        self.father = fa

    # 打印从当前节点到根节点所属的步数（路径）
    def displayToRootNode(self):
        t = self
        tt = 0
        while(True):
            tt += 1
            t.puzzled.printPuzzled()
            t = t.father
            if(t == None):
                break
        print("it need "+str(tt) + " steps!")

    #
    def getFn(self):
        fn = self.getGn()+self.getHn()  # A*
        # fn=self.getHn() #贪婪
        return fn

    #
    def getHn(self):
        Hn = self.puzzled.getNotInPosDis()
        return Hn

    # 返回当前节点到根节点的距离
    def getGn(self):
        gn = 0
        t = self.father
        while(t != None):
            gn += 1
            t = t.father
        return gn

# 此节点类是trie树的节点，用来保存已经扩展过的节点


class TNode:

    def __init__(self, val=None):
        self._value = val
        self._children = {}  # [None,None,None,None,None,None,None,None,None]

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, childs):
        self._children = childs

# trie树，又称前缀树或字典树. 它利用字符串的公共前缀来节约存储空间。


class NumTree:

    # 初始化
    def __init__(self):
        self.root = TNode()

    # 插入新的节点(已扩展过的)
    def insert(self, key):      # key is of type string
        # key should be a low-case string, this must be checked here!
        tnode = self.root
        for char in key:
            if char not in tnode.children:
                # if tnode.children[char]==None:
                child = TNode(char)
                tnode.children[char] = child
                tnode = child
            else:
                tnode = tnode.children[char]
        tnode.value = key  # 叶子节点即为整字符串

    def search(self, key):
        tnode = self.root
        for char in key:
            if char not in tnode.children:
                # if tnode.children[char]==None:
                return None
            else:
                tnode = tnode.children[char]
        return tnode.value

    def display_node(self, tnode):
        if (tnode.value != None):
            print(tnode.value)
        for char in tnode.children.keys():
            if char in tnode.children:
                self.display_node(tnode.children[char])
        return

    def display(self):
        self.display_node(self.root)

    def searchAndInsert(self, m):
        if(self.search(m) == None):
            self.insert(m)
            return False
        return True


class seartchTree:

    # 初始化
    def __init__(self, root):
        self.root = root

    def __search2(self, hlist, m):  # 二分查找，经典算法，从大到小，返回位置
                                    # 若未查找到，则返回应该插入的位置
        low = 0
        high = len(hlist) - 1
        mid = -1
        while(low <= high):
            mid = (low + high)//2
            midval = hlist[mid]

            if midval > m:
                low = mid + 1
            elif midval < m:
                high = mid - 1
            else:
                return (True, mid)
        return (False, mid)

    def __sortInsert(self, hlist, m):  # 对于一个从大到小的序列，
                                       # 插入一个数，仍保持从大到小
        t = self.__search2(hlist, m)
        if(t[1] == -1):
            hlist.append(m)
            return 0
        if(m < hlist[t[1]]):
            hlist.insert(t[1]+1, m)
            return t[1]+1
        else:
            hlist.insert(t[1], m)
            return t[1]

    def breadthFirstSearch(self):  # 广度优先搜索
        if(self.root.puzzled.check() == False):
            print("there is no way to get to the goal state!")
            return
        numTree = NumTree()
        numTree.insert(self.root.puzzled.toOneDimen())
        t = [self.root]
        flag = True
        generation = 0
        while(flag):
            # print("it's the "+str(generation) + " genneration now,the total num of items is "+str(len(t)))
            tb = []
            for i in t:

                if(i.puzzled.isRight() == True):
                    # i.displayToRootNode()
                    flag = False
                    break
                else:
                    for j in i.puzzled.getAbleMove():
                        tt = i.puzzled.clone()
                        tt.move(j)
                        a = node(tt)
                        if(numTree.searchAndInsert(a.puzzled.toOneDimen()) == False):
                            i.addChild(a)
                            tb.append(a)
            t = tb
            generation += 1

    def depthFirstSearch(self):  # 深度优先搜索
        if(self.root.puzzled.check() == False):
            print("there is no way to get to the goal state!")
            return
        numTree = NumTree()
        numTree.insert(self.root.puzzled.toOneDimen())
        t = self.root
        flag = True
        gen = 0  # 深度有界，限制为6层，超过不再扩展
        while(flag):
            # print("genneration: "+str(gen))
            if(gen == 6):
                break
            if(t.puzzled.isRight() == True):
                # t.displayToRootNode()
                flag = False
                break
            else:
                f1 = True
                for j in t.puzzled.getAbleMove():
                    tt = t.puzzled.clone()
                    tt.move(j)
                    a = node(tt)
                    if(numTree.searchAndInsert(a.puzzled.toOneDimen()) == False):
                        t.addChild(a)
                        t = a
                        f1 = False
                        gen += 1
                        break
                if(f1 == True):
                    t = t.father
                    gen -= 1

    def AStarSearch(self):  # A*
        if(self.root.puzzled.check() == False):
            print("there is no way to get to the goal state!")
            return
        numTree = NumTree()
        numTree.insert(self.root.puzzled.toOneDimen())
        leaves = [self.root]
        leavesFn = [0]
        while True:
            t = leaves.pop()  # open表
            # print(leavesFn.pop())
            if(t.puzzled.isRight() == True):
                # t.displayToRootNode()
                break
            for i in t.puzzled.getAbleMove():
                tt = t.puzzled.clone()
                tt.move(i)
                a = node(tt)
                if(numTree.searchAndInsert(a.puzzled.toOneDimen()) == False):  # close表
                    t.addChild(a)
                    fnS = self.__sortInsert(leavesFn, a.getFn())
                    leaves.insert(fnS, a)


"""test NumTree class

trie = NumTree()
print(trie.searchAndInsert([8,7,6,5,4,3,2,1]))
trie.display()

"""

#p = puzzled.generateRandomPuzzle(3, 3, 5)
#p = puzzled([[2,1,3],[4,5,6],[7,8,0]])
ps = []
num = 0
for i in range(0, 300):
    p = puzzled.generateRandomPuzzle(3, 3, 5)
    if(p.check() == True):
        ps.append(p)
        num += 1
    if(num == 30):
        break

time_all_1 = 0
time_all_2 = 0
time_all_3 = 0
i = 1
for x in ps:
    # x.printPuzzled()

    root = node(x)
    s = seartchTree(root)

    time1_b = time.time()
    s.breadthFirstSearch()
    time1_e = time.time()
    time1_o = (time1_e-time1_b)
    time_all_1 += time1_o
    print("bfs %d %.2fms" % (i, time1_o*1000))

    time2_b = time.time()
    s.depthFirstSearch()
    time2_e = time.time()
    time2_o = (time2_e-time2_b)
    time_all_2 += time2_o
    print("dfs %d %.2fms" % (i, time2_o*1000))

    time3_b = time.time()
    s.AStarSearch()
    time3_e = time.time()
    time3_o = (time3_e-time3_b)
    time_all_3 += time3_o
    print("astar %d %.2fms" % (i, time3_o*1000))

    i += 1
print("avgtime:bfs %.2fms - dfs %.2fms - astar %.2fms" %
      ((time_all_1/30)*1000, (time_all_2/30)*1000, (time_all_3/30)*1000))

# p.printPuzzled()
# root = node(p)
# s = seartchTree(root)
# s.breadthFirstSearch()
# s.depthFirstSearch()
# s.AStarSearch()
