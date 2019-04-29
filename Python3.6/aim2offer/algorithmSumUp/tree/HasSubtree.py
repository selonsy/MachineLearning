'''
递归设计思路：

A、判断子树根值是否相等。

A1、如果是相等，需要一个可以判断该根左右孩子结点是否相等的递归函数。

A2、如果不等，找大树的左右孩子结点当根结点，递归到A步骤。
'''

 # -*- coding:utf-8 -*-
 # class TreeNode:
 #     def __init__(self, x):
 #         self.val = x
 #         self.left = None
 #         self.right = None
class Solution:
     def HasSubtree(self, pRoot1, pRoot2):
         # write code here
         res = False
         if pRoot1 == None or pRoot2 == None :
             #根据题干要求返回False
             return res
         if pRoot1.val == pRoot2.val:
             #如果结点相等，Subtree函数负责判断
             #这两个结点的孩子是否相等
             res = Solution.Subtree(self, pRoot1, pRoot2)
         if not res:
             #当与小树相同的子树不在根部，找大树左子树
             res = Solution.HasSubtree(self, pRoot1.left, pRoot2)
         if not res:
             #左边找到了就不需要再找了
             res = Solution.HasSubtree(self, pRoot1.right, pRoot2)
         return res
     
     def Subtree(self, pRoot1, pRoot2):
         #下面两个if位置不能反
         if pRoot2 == None:
             return True
         if pRoot1 == None:
             return False
         if pRoot1.val != pRoot2.val:
             return False
         #左右同时相等时
         return Solution.Subtree(self, pRoot1.left, pRoot2.left) and \
     Solution.Subtree(self, pRoot1.right, pRoot2.right)        
 
'''
为了简化代码，提高代码的阅读障碍，可以使用短路法

class Solution:
     def HasSubtree(self, pRoot1, pRoot2):
         # write code here
         if pRoot1 == None or pRoot2 == None :
             return False
         return Solution.Subtree(self, pRoot1, pRoot2) or \
     Solution.HasSubtree(self, pRoot1.left, pRoot2) or \
     Solution.HasSubtree(self, pRoot1.right, pRoot2)
      
     def Subtree(self, pRoot1, pRoot2):
         if pRoot2 == None:
             return True
         if pRoot1 == None:
             return False
         if pRoot1.val != pRoot2.val:
             return False
         return Solution.Subtree(self, pRoot1.left, pRoot2.left) and \
     Solution.Subtree(self, pRoot1.right, pRoot2.right)
'''
