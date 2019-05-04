
'''
题意：有a个金蛋，每砸一个a[i]可获得a[i - 1] * a[i] * a[i + 1]积分，
默认两端a[-1]和a[n]等于1（但非真实存在，并不参与砸蛋），求问最多可获多少积分。

样例：
输入：
3 1 5 8
输出：
167
详解：
3 1 5 8 --> 3 5 8 --> 3 8 --> 8
+ 15 + 120 + 24 + 8 == 167
'''
       
def my_func(wlist): 
    
    res,len_s,ans = [],len(wlist),[0]
     # 深度优先搜索
    def dfs(index,ans):
        if index==len_s:
            res.append(ans[0])  
            ans[0] = 0  
            return        
        for i in range(0,len_s):
            # hel = wlist[index:index+i]
            if index>=len_s:
                break
            hel = wlist[index]
            left = wlist[i-1] if i>0 else 1
            right = wlist[i+1] if i<len(wlist)-1 else 1
            x = left * hel * right
            ans[0] = ans[0] + x            
            dfs(index+i+1,ans)
    dfs(0,ans)
    return max(res)

# wlist = list(map(int,input().split()))
# print(my_func(wlist))
    # res = []
    #  # 深度优先搜索
    # def dfs(ns,res,ans):
    #     if not ns:
    #         res.append(ans[0])
    #         return        
    #     for i in range(len(ns)): 
    #         nns = ns.copy()
    #         # if not ns:
    #         #     break           
    #         left = nns[i-1] if i>0 else 1
    #         right = nns[i+1] if i<len(nns)-1 else 1
    #         x = left * nns[i] * right
    #         ans[0] = ans[0] + x
    #         nns.pop(i)
    #         dfs(nns,res,ans) 
    # ans = [0]       
    # dfs(wlist,0)
    # return res
    
wlist = [3,1]
print(my_func(wlist))


