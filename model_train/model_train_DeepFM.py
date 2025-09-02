import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import *
# from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate
from tensorflow.keras.models import Model

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

# df = pd.read_csv('./dataset/test_data.csv')
df = pd.read_csv('../dataset/tensorflow_dataset.csv')

print(df.columns)

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
# df.to_csv('./dataset/test_re_data.csv', index=False)
# 三、数据预处理
# 定义特征类型
categorical_features = ['uid', 'post_id', 'hour', 'dow']
numerical_features = ['user_ctr', 'user_activity', 'content_ctr', 'content_impressions']

# 类别特征编码
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 数值特征归一化
numerical_means = df[numerical_features].mean()
numerical_stds = df[numerical_features].std()
df[numerical_features] = (df[numerical_features] - numerical_means) / numerical_stds


# 四、构建DeepFM模型
def build_deepfm(categorical_features, numerical_features, emb_dim=8):
    # 输入层
    inputs = {}
    for col in categorical_features:
        inputs[col] = Input(shape=(1,), name=col)
    for col in numerical_features:
        inputs[col] = Input(shape=(1,), name=col)

    # 类别特征嵌入
    embeddings = []
    for col in categorical_features:
        vocab_size = df[col].nunique() + 1
        emb = Embedding(vocab_size, emb_dim)(inputs[col])
        emb = Reshape((emb_dim,))(emb)
        embeddings.append(emb)

    # 数值特征处理
    numerical_values = [inputs[col] for col in numerical_features]
    numerical_concat = Concatenate()(numerical_values)
    numerical_dense = Dense(16, activation='relu')(numerical_concat)

    # 修正的FM部分（使用Keras层操作）
    fm_terms = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # 使用Multiply层代替tf.multiply
            multiply = Multiply()([embeddings[i], embeddings[j]])
            # 使用Lambda层求和
            sum_product = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(multiply)
            fm_terms.append(sum_product)

    fm = Add()(fm_terms) if fm_terms else Dense(1)(numerical_dense)
    fm = Dense(1, activation=None)(fm)

    # Deep部分
    deep = Concatenate()(embeddings + [numerical_dense])
    deep = Dense(128, activation='relu')(deep)
    deep = Dense(64, activation='relu')(deep)
    deep_out = Dense(1, activation=None)(deep)

    # 合并输出
    combined = Add()([fm, deep_out])
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=list(inputs.values()), outputs=output)
    return model, inputs


# 定义一个自定义Keras回调来记录每个epoch的性能指标
class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, output_dir="performance_logs"):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.aucs = []
        self.log_losses = []
        self.accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val).flatten()

        auc = roc_auc_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred)
        acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))

        print(f"\nEpoch {epoch + 1} - AUC: {auc:.4f} - Log Loss: {logloss:.4f} - Accuracy: {acc:.4f}")

        self.aucs.append(auc)
        self.log_losses.append(logloss)
        self.accuracies.append(acc)

        # 绘制并保存图像
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.aucs, label='AUC')
        plt.title('Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('Score')

        plt.subplot(1, 3, 2)
        plt.plot(self.log_losses, label='Log Loss', color='orange')
        plt.title('Validation Log Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 3)
        plt.plot(self.accuracies, label='Accuracy', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'performance_epoch_{epoch + 1}.png'))
        plt.close()


# 五、数据准备
X = df[categorical_features + numerical_features]
y = df['is_hit'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 转换为字典格式输入
train_data = {col: X_train[col].values for col in X_train.columns}
test_data = {col: X_test[col].values for col in X_test.columns}

# # 确认并配置GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # 设置仅使用第一个GPU
#         tf.config.set_visible_devices(gpus[0], 'GPU')
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)

# 六、模型训练
model, inputs_dict = build_deepfm(categorical_features, numerical_features)
print(model.summary())
print(inputs_dict)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')])

# 创建自定义回调实例
perf_callback = PerformanceVisualizationCallback(model=model, validation_data=(test_data, y_test))

# 调整正样本权重，平衡正负样本
class_weight = {0: 1., 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}

# 早停法
early_stop = EarlyStopping(
    monitor='val_auc',
    patience=3,
    mode='max',
    restore_best_weights=True)

history = model.fit(
    train_data,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop, perf_callback])


# 七、模型评估
def print_metrics(y_true, y_pred):
    print(f"AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print(f"Log Loss: {log_loss(y_true, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, (y_pred > 0.5).astype(int)):.4f}")


# 测试集校验
y_pred = model.predict(test_data).flatten()
print("Test Metrics:")
# 测试集评估
print_metrics(y_test, y_pred)

## 将模型保存成pb格式文件
export_path = "../trained_models/saved_ctr_model/"
version = "1"  # 后续可以通过版本号进行模型版本迭代与管理

# 构建完整保存路径
full_export_path = os.path.join(export_path, version)

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(full_export_path):
    os.makedirs(full_export_path)


# 定义标签
@tf.function(input_signature=[{
    'uid': tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name='uid'),
    'post_id': tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name='post_id'),
    'hour': tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name='hour'),
    'dow': tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name='dow'),
    'user_ctr': tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name='user_ctr'),
    'user_activity': tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name='user_activity'),
    'content_ctr': tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name='content_ctr'),
    'content_impressions': tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name='content_impressions')
}])
def serving_default(inputs):
    return {'prediction': model(inputs)}


tf.saved_model.save(model, full_export_path, signatures={'serving_default': serving_default})
