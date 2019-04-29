'''
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。

给定 target = 20，返回 false。
'''

# 这里需要考虑一个问题，与c#等语言不同，python不能改变for循环期间的i，即定位点的序号。
# 所以使用while循环，把col和row拎出来使用。

class Solution:

    '''
    首先从行号最小列号最大（右上角）的数开始比较，若目标较小，说明当前遍历的数较大，则使列号减一，减小当前遍历的数；
    若目标较大，说明当前遍历的数较小，则使行号加一，增大当前遍历的数。
    '''
    def searchMatrix1(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        row = 0  
        maxrow = len(matrix) - 1   
        col = len(matrix[0]) - 1  
        while col>=0 and row<= maxrow:  
            if matrix[row][col]  == target:  
                return True  
            elif target > matrix[row][col]:  
                row+=1  
            else:  
                target < matrix[row][col]  
                col-=1  
        return False  
    
    '''
    首先从行号最小列号最大（右上角）的数开始比较，若目标较小，说明当前遍历的数较大，则使列号减一，减小当前遍历的数；
    若目标较大，说明当前遍历的数较小，则使行号加一，增大当前遍历的数。
    '''
    def searchMatrix2(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        col = 0  
        maxcol = len(matrix[0]) - 1   
        row = len(matrix) - 1  
        while row>=0 and col<= maxcol:  
            if matrix[row][col]  == target:  
                return True  
            elif  target > matrix[row][col]:  
                col+=1  
            else :  
                target < matrix[row][col]  
                row-=1  
        return False           

s = Solution()
_input = [
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
_output = s.searchMatrix1(_input,5)
print(_output)