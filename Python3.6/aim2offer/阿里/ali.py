
''' 输入是一个数组，如 {1, 2, 5, 5, 8, 2, 2, 1}, 求这个数组的一个最短连续的子串（不需要全部找出，找到一个即可），
能包含所有元素，例如上面的数组，{1,2,5,5,8} 为正确 \
解答，包含了 1， 2， 5， 8 全部4个元素。并给出复杂度分析'''
import collections
def func(arr):
      n = len(set(arr))
      i = 0
      d = collections.deque()
      min = len(arr)
      ans = []
      dic = {}
      while i<len(arr):
            while (i+1<len(arr) and arr[i+1]>=arr[i]) and len(set(d))<=n:
                  d.append(arr[i])
                  
                  if arr[i] in dic:
                        dic[arr[i]]+=1 
                  else:
                        dic[arr[i]] = 1
                  i+=1
            if i<len(arr) and arr[i]>arr[i-1]:
                  d.append(arr[i])
                  if arr[i] in dic:
                        dic[arr[i]]+=1 
                  else:
                        dic[arr[i]] = 1
                  
            
            while len(d)>0 and len(set(d))>=n and dic[d[0]]>1:
                  dic[d[0]] -= 1
                  d.popleft()
            if len(d)<min:
                  min = len(d)
                  ans = d
            i+=1
      return ans
      
# arr = list(map(int,input().split()))
arr = [2, 1, 5, 4, 8, 2, 2, 1]
# arr = [1,1,1,2]
print(func(arr))
#print "Hello World!"
#done 时间复杂度 o(N)

