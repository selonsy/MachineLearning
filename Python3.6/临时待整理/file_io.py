# 打开一个文件
# fo = open("qq.txt", "a")
# fo.write( "www.runoob.com!\nVery good site!\n")
 
# # 关闭打开的文件
# fo.close()

with open('qq_word2.txt','a') as f:
    for x in range(0,10):
        f.write("www.runoob.com!\nVery good site!\n %d" %x)