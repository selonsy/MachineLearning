import numpy as np
import pandas as pd
import datetime

def reduce_mem_usage(df, verbose=True):
    '''[summary]
    看起来好像是数据清洗，减少内存的使用
    Arguments:
        df {[type]} -- [description]
    
    Keyword Arguments:
        verbose {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    '''

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# parse_dates是什么意思？
new_transactions = pd.read_csv(r'D:\workspace\MachineLearning\Kaggle\Elo_Merchant_Category_Recommendation\dataset\new_merchant_transactions.csv',parse_dates=['purchase_date'])

historical_transactions = pd.read_csv(r'D:\workspace\MachineLearning\Kaggle\Elo_Merchant_Category_Recommendation\dataset\historical_transactions.csv',parse_dates=['purchase_date'])

def binarize(df):
    '''[summary]
    二值化，用于将布尔型数据进行转化。
    Arguments:
        df {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    '''
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)

# 格式化日期数据,formatting the dates
def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
# load the main files, and extracting the target
train = read_data(r'D:\workspace\MachineLearning\Kaggle\Elo_Merchant_Category_Recommendation\dataset\train.csv')
test = read_data(r'D:\workspace\MachineLearning\Kaggle\Elo_Merchant_Category_Recommendation\dataset\test.csv')

target = train['target']
del train['target']


# 特征工程 Feature engineering
historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
historical_transactions['month_diff'] += historical_transactions['month_lag']

new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30
new_transactions['month_diff'] += new_transactions['month_lag']

