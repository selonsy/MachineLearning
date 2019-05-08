'''
每封电子邮件都由一个本地名称和一个域名组成，以 @ 符号分隔。
例如，在 alice@leetcode.com中， alice 是本地名称，而 leetcode.com 是域名。
除了小写字母，这些电子邮件还可能包含 '.' 或 '+'。
如果在电子邮件地址的本地名称部分中的某些字符之间添加句点（'.'），则发往那里的邮件将会转发到本地名称中没有点的同一地址。例如，"alice.z@leetcode.com” 和 “alicez@leetcode.com” 会转发到同一电子邮件地址。 （请注意，此规则不适用于域名。）

如果在本地名称中添加加号（'+'），则会忽略第一个加号后面的所有内容。这允许过滤某些电子邮件，例如 m.y+name@email.com 将转发到 my@email.com。 （同样，此规则不适用于域名。）

可以同时使用这两个规则。
给定电子邮件列表 emails，我们会向列表中的每个地址发送一封电子邮件。实际收到邮件的不同地址有多少？

示例：
输入：["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
输出：2
解释：实际收到邮件的是 "testemail@leetcode.com" 和 "testemail@lee.tcode.com"。

提示：

    1 <= emails[i].length <= 100
    1 <= emails.length <= 100
    每封 emails[i] 都包含有且仅有一个 '@' 字符。

'''
import time
import copy
class Solution:    
    # def numUniqueEmails(self, emails: List[str]) -> int:
    def func1(self, emails):
        res = []
        for email in emails:
            local, area = email.split('@')            
            local = str.replace(local,'.','')
            if '+' in local:
                local = local[0:str.index(local,'+')]
            res.append(local+'@'+area)
        return len(set(res))        
     
    def getVarLen(self):
        co = self.func1.__code__        
        return co.co_argcount

ori_data = [["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]]
expect_data = [2]
assert len(ori_data) == len(expect_data)

s = Solution()
funcs = []
var_len = s.getVarLen() - 1 # exclude self
for key in Solution.__dict__.keys():
    if "func" in key:
        funcs.append(getattr(s, key)) # 反射获取实例化的func
for f in range(len(funcs)):
    func = funcs[f]
    assert func!=None,'func不能为空！'
    begin = time.time() # ToDo：这里的计时有问题，需要改进
    data_length = len(expect_data)
    input_data = copy.deepcopy(ori_data) # 前面的func可能修改测试数据，故深拷贝
    for i in range(data_length):
        if var_len == 1:
            res = func(input_data[i])
            assert res == expect_data[i], "func{0}({3}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i])
        elif var_len == 2:
            res = func(input_data[i][0],input_data[i][1])
            assert res == expect_data[i], "func{0}({3},{4}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i][0],input_data[i][1])
        elif var_len == 3:
            res = func(input_data[i][0],input_data[i][1],input_data[i][2])
            assert res == expect_data[i], "func{0}({3},{4},{5}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i][0],input_data[i][1],input_data[i][2])
        elif var_len == 4:
            res = func(input_data[i][0],input_data[i][1],input_data[i][2],input_data[i][3])
            assert res == expect_data[i], "func{0}({3},{4},{5},{6}): expected = {1}, but actually = {2}".format(f+1,expect_data[i], res,input_data[i][0],input_data[i][1],input_data[i][2],input_data[i][3])
    end = time.time()
    print("func{0} : {1:.4f} ms".format(f+1, (end-begin)*1000/data_length))
print("done")
