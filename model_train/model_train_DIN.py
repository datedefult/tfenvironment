import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ====================== 模拟数据生成 ======================
def generate_sequential_data(num_users=1000, num_items=500, max_seq_length=20):
    np.random.seed(42)

    # 生成用户行为序列
    user_hist = {}
    for uid in range(num_users):
        seq_length = np.random.randint(5, max_seq_length)
        user_hist[uid] = {
            'post_ids': np.random.choice(num_items, seq_length),
            'timestamps': np.sort(np.random.randint(1609459200, 1640995200, seq_length))
        }

    # 生成样本数据
    samples = []
    for _ in range(10000):
        uid = np.random.choice(num_users)
        pos = np.random.choice(len(user_hist[uid]['post_ids']))
        target_item = user_hist[uid]['post_ids'][pos]

        samples.append({
            'uid': uid,
            'target_post_id': target_item,
            'hist_post_ids': user_hist[uid]['post_ids'],
            'hist_timestamps': user_hist[uid]['timestamps'],
            'is_hit': np.random.binomial(1, 0.2)
        })

    return pd.DataFrame(samples)


df = generate_sequential_data()

# ====================== 数据预处理 ======================
# 1. 类别特征编码
label_encoders = {
    'uid': LabelEncoder().fit(df['uid']),
    'post_id': LabelEncoder().fit(np.concatenate(df['hist_post_ids']))
}

# 2. 生成行为序列特征
max_seq_length = 20


def process_sequence(seq, le):
    encoded = le.transform(seq[:max_seq_length])
    return np.pad(encoded, (0, max_seq_length - len(encoded)), mode='constant')


df['hist_post_ids_encoded'] = df['hist_post_ids'].apply(
    lambda x: process_sequence(x, label_encoders['post_id']))
df['target_post_id_encoded'] = label_encoders['post_id'].transform(df['target_post_id'])
df['uid_encoded'] = label_encoders['uid'].transform(df['uid'])

# 3. 划分数据集
X = df[['uid_encoded', 'target_post_id_encoded', 'hist_post_ids_encoded']]
y = df['is_hit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# ====================== DIN模型构建 ======================
class AttentionPooling(Layer):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.W = Dense(embedding_dim, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, inputs):
        # inputs: [batch_size, seq_len, embedding_dim]
        # target: [batch_size, embedding_dim]
        target_embedding, hist_embeddings = inputs

        # 计算注意力得分
        target_expanded = tf.expand_dims(target_embedding, 1)  # [bs, 1, dim]
        attention_logits = self.q(self.W(target_expanded + hist_embeddings))  # [bs, seq, 1]
        attention_scores = tf.nn.softmax(attention_logits, axis=1)  # [bs, seq, 1]

        # 加权求和
        output = tf.reduce_sum(attention_scores * hist_embeddings, axis=1)  # [bs, dim]
        return output


def build_din_model(num_users, num_items, embedding_dim=32):
    # 输入层
    uid_input = Input(shape=(1,), name='uid')
    target_item_input = Input(shape=(1,), name='target_post_id')
    hist_items_input = Input(shape=(max_seq_length,), name='hist_post_ids')

    # Embedding层
    user_embedding = Embedding(num_users, embedding_dim)(uid_input)
    item_embedding = Embedding(num_items, embedding_dim)

    # 目标物品嵌入
    target_embedding = item_embedding(target_item_input)
    target_embedding = Reshape((embedding_dim,))(target_embedding)

    # 历史行为序列嵌入
    hist_embeddings = item_embedding(hist_items_input)  # [bs, seq_len, dim]

    # 注意力池化
    attention_output = AttentionPooling(embedding_dim)([target_embedding, hist_embeddings])

    # 合并特征
    merged = Concatenate()([
        tf.squeeze(user_embedding, axis=1),
        target_embedding,
        attention_output
    ])

    # Deep层
    deep = Dense(128, activation='relu')(merged)
    deep = Dense(64, activation='relu')(deep)

    # 输出层
    output = Dense(1, activation='sigmoid')(deep)

    model = Model(inputs=[uid_input, target_item_input, hist_items_input], outputs=output)
    return model


# ====================== 模型实例化 ======================
num_users = len(label_encoders['uid'].classes_)
num_items = len(label_encoders['post_id'].classes_)
model = build_din_model(num_users, num_items)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

# ====================== 数据输入管道 ======================
train_data = {
    'uid': X_train['uid_encoded'].values,
    'target_post_id': X_train['target_post_id_encoded'].values,
    'hist_post_ids': np.stack(X_train['hist_post_ids_encoded'].values)
}

test_data = {
    'uid': X_test['uid_encoded'].values,
    'target_post_id': X_test['target_post_id_encoded'].values,
    'hist_post_ids': np.stack(X_test['hist_post_ids_encoded'].values)
}

# ====================== 模型训练 ======================
early_stop = EarlyStopping(
    monitor='val_auc',
    patience=3,
    mode='max',
    restore_best_weights=True
)

history = model.fit(
    train_data,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    class_weight={0: 1., 1: 5.},  # 处理样本不平衡
    callbacks=[early_stop]
)

# ====================== 模型评估 ======================
y_pred = model.predict(test_data)
print(f"Test AUC: {roc_auc_score(y_test, y_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, (y_pred > 0.5).astype(int)):.4f}")

# ====================== 模型保存 ======================
tf.saved_model.save(model, 'din_model',
                    signatures={'serving_default': model.signatures['serving_default']})