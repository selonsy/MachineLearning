'''
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

说明：本题中，我们将空字符串定义为有效的回文串。

示例 1:

输入: "A man, a plan, a canal: Panama"
输出: true
示例 2:

输入: "race a car"
输出: false
'''

class Solution:
    def isPalindrome1(self,s):
        '''
        厉害了，s==s[::-1]直接判断正反序列是否一致就可以了。
        filter也是厉害的一批
        '''
        s = ''.join(filter(str.isalnum,s)).lower()
        return s==s[::-1]
    def isPalindrome2(self, s):
        '''
            s: str
            rType:bool
        '''        
        res=''
        for i in s:
            if (ord(i)>=ord('A') and ord(i)<=ord('Z')) or (ord(i)>=ord('a') and ord(i)<=ord('z')):
                res += i.lower()
            if ord(i)>=ord('0') and ord(i)<=ord('9'):
                res += i
        # print(res)
        import math
        k = 0 
        l =  len(res)
        if res =='':
            return True
        while k <= math.ceil(l//2)-1:
            if res[k] != res[l-1-k]:
                return False
            k+=1            
        return True

    def isPalindrome3(self,s):
        i,j=0,len(s)-1
        while i<j:
            if s[i]!=s[j]:
                return False
            i+=1
            j-=1
        return True

        

s = Solution()
x = s.isPalindrome1('A man, a plan, a canal: Panama')
print(x)
