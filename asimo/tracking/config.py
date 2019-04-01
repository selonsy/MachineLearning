system_type = 0   # 0 for windows and 1 for linux 

configs = [
    # 模型地址
    ['D:\\workspace\\vot\\asimo\\other\\DaSiamRPN\\selonsy\\model\\SiamRPNBIG.model',''],
    # 数据集地址
    ['D:\\workspace\\vot\\tracker_benchmark\\OTB\\',''],
    # 跟踪结果保存位置
    ['D:\\workspace\\vot\\asimo\\SiamFPN\\results\\','']
]

model_path = configs[0][system_type]
dataset_path = configs[1][system_type]
result_path = configs[2][system_type]