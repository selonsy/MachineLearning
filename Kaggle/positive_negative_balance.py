# 处理正负样本不均匀的案例
# 有些案例中，正负样本数量相差非常大，数据严重unbalanced，这里提供几个解决的思路

# 计算正负样本比例
positive_num = df_train[df_train['label']==1].values.shape[0]
negative_num = df_train[df_train['label']==0].values.shape[0]
print(float(positive_num)/float(negative_num))


# 主要思路
# 1. 手动调整正负样本比例
# 2. 过采样 Over-Sampling
#    对训练集里面样本数量较少的类别（少数类）进行过采样，合成新的样本来缓解类不平衡，比如SMOTE算法
# 3. 欠采样 Under-Sampling
# 4. 将样本按比例一一组合进行训练，训练出多个弱分类器，最后进行集成

# 框架推荐
# Github上大神写的相关框架，专门用来处理此类问题： 
# https://github.com/scikit-learn-contrib/imbalanced-learn