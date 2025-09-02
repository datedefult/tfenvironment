import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# 自定义对比损失函数
def contrastive_loss(margin=1.0):
    def loss_function(y_true, y_pred):
        # y_true: 标签 (0 表示正样本，1 表示负样本)
        # y_pred: 用户向量和物品向量的欧几里得距离
        distance = y_pred
        positive_loss = y_true * tf.square(distance)
        negative_loss = (1 - y_true) * tf.square(tf.maximum(0.0, margin - distance))
        return tf.reduce_mean(positive_loss + negative_loss)
    return loss_function


# 定义用户塔
def build_user_tower(user_input_dim, embedding_dim):
    user_input = Input(shape=(1,), name="user_input", dtype=tf.int32)

    # 用户ID的Embedding层
    user_embedding = layers.Embedding(
        input_dim=user_input_dim,  # 用户总数
        output_dim=embedding_dim,  # Embedding维度
        input_length=1,  # 输入长度为1
        name="user_embedding"
    )(user_input)

    # 展平并添加全连接层
    user_vector = layers.Flatten()(user_embedding)
    user_vector = layers.Dense(64, activation='relu')(user_vector)
    user_vector = layers.Dense(embedding_dim, activation='relu')(user_vector)

    return Model(inputs=user_input, outputs=user_vector, name="user_tower")


# 定义物品塔
def build_item_tower(item_input_dim, embedding_dim):
    item_input = Input(shape=(1,), name="item_input", dtype=tf.int32)

    # 物品ID的Embedding层
    item_embedding = layers.Embedding(
        input_dim=item_input_dim,  # 物品总数
        output_dim=embedding_dim,  # Embedding维度
        input_length=1,  # 输入长度为1
        name="item_embedding"
    )(item_input)

    # 展平并添加全连接层
    item_vector = layers.Flatten()(item_embedding)
    item_vector = layers.Dense(64, activation='relu')(item_vector)
    item_vector = layers.Dense(embedding_dim, activation='relu')(item_vector)

    return Model(inputs=item_input, outputs=item_vector, name="item_tower")


# 定义双塔模型
def build_two_tower_model(user_input_dim, item_input_dim, embedding_dim):
    # 构建用户塔和物品塔
    user_tower = build_user_tower(user_input_dim, embedding_dim)
    item_tower = build_item_tower(item_input_dim, embedding_dim)

    # 输入
    user_input = Input(shape=(1,), name="user_input", dtype=tf.int32)
    item_input = Input(shape=(1,), name="item_input", dtype=tf.int32)

    # 获取用户和物品的向量表示
    user_vector = user_tower(user_input)
    item_vector = item_tower(item_input)

    # 计算匹配分数（点积）
    dot_product = layers.Dot(axes=1)([user_vector, item_vector])
    # distance = tf.norm(user_vector - item_vector, axis=1, keepdims=True)
    # 构建模型
    model = Model(inputs=[user_input, item_input], outputs=dot_product, name="two_tower_model")
    return model


# 参数设置
user_input_dim = 1000  # 用户总数
item_input_dim = 5000  # 物品总数
embedding_dim = 64  # Embedding维度

# 构建双塔模型
model = build_two_tower_model(user_input_dim, item_input_dim, embedding_dim)

# 编译模型
model.compile(optimizer='adam', loss=contrastive_loss(margin=1.0), metrics=['accuracy'])

# 打印模型结构
model.summary()

# 模拟数据
import numpy as np

num_samples = 1000
user_ids = np.random.randint(0, user_input_dim, size=(num_samples, 1))
item_ids = np.random.randint(0, item_input_dim, size=(num_samples, 1))
labels = np.random.randint(0, 2, size=(num_samples, 1))  # 点击标签（0或1）



# 训练模型
model.fit([user_ids, item_ids], labels, batch_size=32, epochs=5)

print(model.evaluate([user_ids, item_ids], labels))
