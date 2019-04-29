# 递归+深度优先，实现字符串的全排列

'''
普通递归实现全排列:

思想：
举个例子，比如你要对a,b,c三个字符进行全排列，那么它的全排列有abc,acb,bac,bca,cba,cab这六种可能，你们想想你们是如何得出这六种可能的。
没错！就是当指针指向第一个元素a时，它可以是其本身a(即和自己进行交换)，还可以和b，c进行交换，故有3种可能，
当第一个元素a确定以后，指针移向第二位置，第二个位置可以和其本身b及其后的元素c进行交换，又可以形成两种排列，
当指针指向第三个元素c的时候，这个时候其后没有元素了，此时，则确定了一组排列，输出。

###但是每次输出后要把数组恢复为原来的样子。

简单来说，它的思想即为，确定第1位，对n-1位进行全排列，确定第二位，对n-2位进行全排列。。。显然，这是一种递归的思想。

推理流程：
比如：集合{ 1,2,3}的全排列为：
{ 1 2 3} 
{ 1 3 2 }
{ 2 1 3 }
{ 2 3 1 }
{ 3 2 1 }
{ 3 1 2 }

递归思想：
取出数组中第一个元素放到最后，即a[1]与a[n]交换，然后递归求a[n-1]的全排列

1）如果数组只有一个元素n=1，a={1} 则全排列就是{1}

2）如果数组有两个元素n=2，a={1,2} 则全排列是：
{2,1}--a[1]与a[2]交换。交换后求a[2-1]={2}的全排列，归结到1)
{1,2}--a[2]与a[2]交换。交换后求a[2-1]={1}的全排列，归结到1)

3）如果数组有三个元素n=3，a={1,2,3} 则全排列是
{{2,3},1}--a[1]与a[3]交换。后求a[3-1]={2,3}的全排列，归结到2）
{{1,3},2)--a[2]与a[3]交换。后求a[3-1]={1,3}的全排列，归结到2）
{{1,2},3)--a[3]与a[3]交换。后求a[3-1]={1,2}的全排列，归结到2）
...
以此类推。

'''
def permutations(arr, position, end):
 
    if position == end:
        print(arr)
 
    else:
        for index in range(position, end):
 
            arr[index], arr[position] = arr[position], arr[index]
            permutations(arr, position+1, end)
            arr[index], arr[position] = arr[position], arr[index]
 
 
arr = ["a","b","c"]
permutations(arr, 0, len(arr))
#['a', 'b', 'c']
#['a', 'c', 'b']
#['b', 'a', 'c']
#['b', 'c', 'a']
#['c', 'b', 'a']
#['c', 'a', 'b']

'''
利用深度优先算法实现全排列
思想：深度搜索，返回时回溯
'''
visit = [True, True, True]
temp = ["" for x in range(0, 3)]
def dfs(position):
    if position == len(arr):
        print(temp)
        return
 
    for index in range(0,len(arr)):
        if visit[index] == True:
            temp[position] = arr[index]
            visit[index] = False
            dfs(position + 1)
            visit[index] = True
 
arr = ["a","b","c"]
dfs(0)
