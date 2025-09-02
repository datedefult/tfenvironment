import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    gender_str_dict = {'男': 0,'女': 1,'LGBT': 2,'保密': 3}
    language_str_dict = {'en': 1,'de': 2,'ja': 3}
    dataset = pd.read_csv('clear_data.csv')

    short_cat_cols = ['hour','dow','gender','language','channel','author_language']
    long_cat_cols = ['country_code','author_country_code']
    id_cols =  ['uid', 'post_id']
    num_cols = ['hits_rate','like_rate','collect_rate','comments_rate','score']

    ctr_tag = ['is_hit']

    dataset['gender'] = dataset['gender'].map(gender_str_dict)
    dataset['language'] = dataset['language'].map(language_str_dict)
    dataset['author_language'] = dataset['author_language'].map(language_str_dict)
    dataset = dataset[id_cols+short_cat_cols+num_cols+ctr_tag]

    print(dataset.info())

    # 类别型特征编码
    categorical_features = ['uid', 'post_id', 'gender', 'language', 'channel', 'author_language']
    for feat in categorical_features:
        lbe = LabelEncoder()
        dataset[feat] = lbe.fit_transform(dataset[feat])  # 转换为0~n的整数

    # 数值型特征归一化
    numerical_features = ['hour', 'dow', 'hits_rate', 'like_rate', 'collect_rate', 'comments_rate', 'score']
    scaler = MinMaxScaler()
    dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

    # 划分特征和标签
    X = dataset.drop('is_hit', axis=1)
    y = dataset['is_hit'].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train)
    print(y_train)
