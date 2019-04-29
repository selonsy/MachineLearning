'''
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。

示例:

输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]
'''
# class Solution:
#     def partition(self, s: str) -> List[List[str]]:
class Solution:
    def partition(self,s):
        res,len_s = [],len(s)

        # 深度优先搜索
        def dfs(re,index):
            if index==len_s:
                res.append(re)            
            for i in range(1,len_s-index+1):
                hel = s[index:index+i]
                if hel==hel[::-1]:
                    dfs(re+[hel],index+i)
        dfs([],0)
        return res
    

s = Solution()
x = s.partition('aab')
print(x)
