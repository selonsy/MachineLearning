import pickle
data = {'name':'python', 'site':'pythontab.com'}
#打开文件，然后将data写入
with open('dump.data', 'wb') as f:
    pickle.dump(data, f)
#同样读取的时候也需要打开文件
with open('dump.data', 'rb') as f:
    data_load = pickle.load(f)
print(data_load)
print(type(data_load))

def test(func,func1):
    """[summary]
    
    Arguments:
        func {[type]} -- [description]
        func1 {[type]} -- [description]
    """
    pass