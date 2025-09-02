import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
"""
'timestamp'：请求发生时时间戳
'uid'：发起请求用户id
'post_id'：当前请求到的内容id
'is_show'：是否看到该内容，全部为1，仅为了展示漏斗占位
'is_hit'：是否点击该内容，0表示未点击，1表示点击
'is_comment'：是否点在该内容发布评论，0表示未评论，1表示评论
'is_like'：是否点赞该内容，0表示未点赞，1表示点赞
'is_collect'：是否收藏该内容，0表示未收藏，1表示收藏
'is_play'：是否游玩该波形，0表示未游玩，1表示游玩，仅部分存在
'is_download'：是否下载该波形内容，0表示未下载，1表示下载，仅部分存在
'is_click_link'：是否点击该内容中的链接，0表示未点击，1表示点击，仅部分存在
"""

df = pd.read_csv('../tensorflow_learn/test_data.csv')
print(df.columns)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 二、特征工程
# 1. 时间特征处理
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek

# 2. 统计特征
# 用户历史点击率
user_stats = df.groupby('uid')['is_hit'].agg(['mean', 'count']).rename(
    columns={'mean': 'user_ctr', 'count': 'user_activity'})
df = df.merge(user_stats, on='uid', how='left')

# 内容热度
content_stats = df.groupby('post_id')['is_hit'].agg(['mean', 'count']).rename(
    columns={'mean': 'content_ctr', 'count': 'content_impressions'})
df = df.merge(content_stats, on='post_id', how='left')

# 三、数据预处理
# 定义特征类型
categorical_features = ['uid', 'post_id', 'hour', 'dow']
numerical_features = ['user_ctr', 'user_activity', 'content_ctr', 'content_impressions']
