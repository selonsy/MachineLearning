import time

class Solution:
    def func1(self, x: int) -> int:
        pass

    def func2(self, x: int) -> int:
        pass
        
    def getVarLen(self):
        co = self.func1.__code__        
        return co.co_argcount

input_data = [0, 0, 0]
expect_data = [None, None, None]
assert len(input_data) == len(expect_data)

s = Solution()
funcs = []
func_len = 0
var_len = s.getVarLen() - 1 # exclude self
for key in Solution.__dict__.keys():
    if "func" in key:
        func_len += 1
if func_len == 1:
    funcs = [s.func1]
elif func_len == 2:
    funcs = [s.func1, s.func2]
for f in range(len(funcs)):
    func = funcs[f]
    begin = time.time() # ToDo：这里的计时有问题，需要改进
    data_length = len(expect_data)
    for i in range(data_length):
        if var_len == 1:
            res = func(input_data[i])
            assert res == expect_data[i], "func{0}({1}): expected = {2}, but actually = {3}".format(f+1,input_data[i],expect_data[i], res)
        elif var_len == 2:
            res = func(input_data[i][0],input_data[i][1])
            assert res == expect_data[i], "func{0}({1},{2}): expected = {3}, but actually = {4}".format(f+1,input_data[i][0],input_data[i][1],expect_data[i], res)
    end = time.time()
    print("func{0} : {1:.4f} ms".format(f+1, (end-begin)*1000/data_length))
print("done")
