import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset.DatasetReturn import data_loader
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import StringLookup, IntegerLookup, CategoryEncoding
import tensorflow as tf
from tensorflow import feature_column as fc
from typing import List, Union, Dict, Optional
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import StringLookup, IntegerLookup, CategoryEncoding



if __name__ == '__main__':
    # connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    # connection_pools.init_mysql()
    str_cols = ['title']

    dataset = data_loader()

    short_cat_cols = ['hour','dow','gender','language','channels','author_language']
    long_cat_cols = ['country_code','author_country_code']
    id_cols =  ['uid', 'author_id', 'post_id']
    num_cols = ['friends_num','binding_toys_number','hits_rate','like_rate','collect_rate','comments_rate','score']

    # 独热编码的类别字典
    one_hot_dict = {
        'hour':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
        'dow':[0,1,2,3,4,5,6],
        'gender':['男', '女', 'LGBT', '保密'],
        'language':['en','de','ja'],
        'channels':[1,2,3,4,5],
        'author_language':['en','de','ja'],
    }
    # 默认值字典
    default_values = {
        'hour': 22,
        'dow': 5,
        'gender': '保密',
        'language': 'en',
        'channels': 2,
        'author_language': 'en',
    }




    ctr_tag = ['is_hit']
    cvr_tag = ['is_like']

    # 对数据集进行编码处理
    # 初始化 OneHotEncoder
    encoders = {}
    encoded_features = []

    for col in one_hot_dict.keys():
        # 获取该列的类别列表
        categories = one_hot_dict[col]

        # 初始化 OneHotEncoder
        encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore', sparse_output=False)
        encoders[col] = encoder

        # 填充缺失值或未知值
        dataset[col] = dataset[col].fillna(default_values[col])
        dataset[col] = dataset[col].apply(lambda x: x if x in categories else default_values[col])

        # 独热编码
        encoded_col = encoder.fit_transform(dataset[[col]])
        encoded_features.append(encoded_col)

    # 拼接所有独热编码结果
    encoded_array = np.hstack(encoded_features)
    encoded_df = pd.DataFrame(encoded_array)
    print(encoded_df)
