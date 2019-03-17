'''
请用自己熟悉的语言实现一个方法：
solve(X,Y)

用 最小二乘法 和 梯度下降法 求解多元线性回归的参数，其中X,Y是待拟合的数据，
X是自变量，Y是因变量。

'''
 
import numpy as np
import sys, math
def solve(xArr, yArr):
        '''
        Solve multiple linear regression based on least square algorithm
    
        Inputs:
        - X： A 2-D array of shape (num_of_examples, dimension_of_x + 1)
        - Y: A 1-D array of shape (num_of_examples,)
    
        Returns:
        - coefficents: A 1-D array of shape (dimension_of_x + 1,).
        '''
        xMat=np.mat(xArr)
        yMat=np.mat(yArr).T
        xTx=xMat.T*xMat
        if np.linalg.det(xTx)==0.0:
            
            return
        ws=xTx.I*(xMat.T*yMat)
        return ws
        # return []

if __name__ == '__main__':    
    X = []
    Y = []

    while True:
            line = sys.stdin.readline().strip('\r\n')        
            if line == '':
                    break            
            nums = line.split(',')
            nums = [float(n) for n in nums]            
            X.append([1.0] + nums[:-1])
            Y.append(nums[-1])    
    coefs = solve(X, Y)    
    formatted_coefs = []
    for coef in coefs:
            coef = math.floor(coef * 100)/100
            formatted_coefs.append('%.2f' % coef)            
    print(','.join(formatted_coefs))            

# 1.0,1.0,3.5
# 2.0,2.0,6.5
# 1.0,3.0,5.5
# 1.0,6.0,4.0