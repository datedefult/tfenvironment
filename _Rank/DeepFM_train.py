"""
输入层这里， 用到的特征主要是离散型特征和连续性特征， 这里不管是哪一类特征，都会过embedding层转成低维稠密的向量，是的，
连续性特征，这里并没有经过分桶离散化，而是直接走embedding。
这个是怎么做到的呢？就是就是类似于预训练时候的思路，先通过item_id把连续型特征与类别特征关联起来，
最简单的，就是把item_id拿过来，过完embedding层取出对应的embedding之后，再乘上连续值即可， 所以这个连续值事先一定要是归一化的。
当然，这个玩法，我也是第一次见。 学习到了， 所以模型整体的输入如下：
"""

from dataset.DatasetReturn import data_loader
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Reshape, Multiply, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os

def build_deepfm(categorical_features, numerical_features, emb_dim=8):
    inputs = {}
    for col in categorical_features:
        inputs[col] = Input(shape=(1,), name=col)
    for col in numerical_features:
        inputs[col] = Input(shape=(1,), name=col)

    # Embedding for categorical features
    embeddings = []
    for col in categorical_features:
        emb = Embedding(1000, emb_dim)(inputs[col])
        emb = Reshape((emb_dim,))(emb)
        embeddings.append(emb)

    # Numerical features
    numerical_values = [inputs[col] for col in numerical_features]
    numerical_concat = Concatenate()(numerical_values)
    numerical_dense = Dense(16, activation='relu')(numerical_concat)

    # FM part
    fm_terms = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            multiply = Multiply()([embeddings[i], embeddings[j]])
            sum_product = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(multiply)
            fm_terms.append(sum_product)

    fm = Add()(fm_terms) if fm_terms else Dense(1)(numerical_dense)
    fm = Dense(1, activation=None)(fm)

    # Deep part
    deep = Concatenate()(embeddings + [numerical_dense])
    deep = Dense(128, activation='relu')(deep)
    deep = Dense(64, activation='relu')(deep)
    deep_out = Dense(1, activation=None)(deep)

    # Combine outputs
    combined = Add()([fm, deep_out])
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=list(inputs.values()), outputs=output)
    return model, inputs


class PerformanceVisualizationCallback(Callback):
    def __init__(self, validation_data, output_dir="performance_logs"):
        super().__init__()
        self.validation_data = validation_data
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.aucs = []
        self.log_losses = []
        self.accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val).flatten()  # Use self.model from Callback base class

        auc = roc_auc_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred)
        acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))

        print(f"\nEpoch {epoch + 1} - AUC: {auc:.4f} - Log Loss: {logloss:.4f} - Accuracy: {acc:.4f}")

        self.aucs.append(auc)
        self.log_losses.append(logloss)
        self.accuracies.append(acc)

        # Plot and save metrics
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


if __name__ == '__main__':
    dataset = data_loader()
    dataset.to_csv('./test.csv')
    # Define feature columns
    short_cat_cols = ['gender', 'language', 'channels', 'author_language']
    long_cat_cols = ['country_code', 'author_country_code']
    id_cols = ['uid', 'author_id', 'post_id']
    num_cols = ['friends_num', 'binding_toys_number', 'hits_rate', 'like_rate', 'collect_rate', 'comments_rate', 'score']
    ctr_tag = ['is_hit']

    categorical_features = short_cat_cols + long_cat_cols + id_cols
    numerical_features = num_cols

    # Data preparation
    X = dataset[categorical_features + numerical_features]
    y = dataset[ctr_tag[0]]

    # Handle missing values
    X = X.fillna(0)  # 填充空值为0
    y = y.fillna(0)

    # Ensure correct data types
    for col in categorical_features:
        X[col] = X[col].astype('int32')  # 确保类别特征为整数类型
    for col in numerical_features:
        X[col] = X[col].astype('float32')  # 确保数值特征为浮点类型
    y = y.astype('float32')  # 确保目标变量为浮点类型

    # Balance positive and negative samples
    pos_samples = dataset[dataset[ctr_tag[0]] == 1]
    neg_samples = dataset[dataset[ctr_tag[0]] == 0].sample(len(pos_samples), random_state=42)
    balanced_dataset = pd.concat([pos_samples, neg_samples])
    X_balanced = balanced_dataset[categorical_features + numerical_features]
    y_balanced = balanced_dataset[ctr_tag[0]]

    # Handle missing values and ensure correct data types for balanced dataset
    X_balanced = X_balanced.fillna(0)
    y_balanced = y_balanced.fillna(0)
    for col in categorical_features:
        X_balanced[col] = X_balanced[col].astype('int32')
    for col in numerical_features:
        X_balanced[col] = X_balanced[col].astype('float32')
    y_balanced = y_balanced.astype('float32')

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

    # Convert to dictionary format
    train_data = {col: X_train[col].values for col in X_train.columns}
    test_data = {col: X_test[col].values for col in X_test.columns}

    # Build model
    model, inputs_dict = build_deepfm(categorical_features, numerical_features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # 损失函数需要自定义
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')])

    # Custom callback
    perf_callback = PerformanceVisualizationCallback(validation_data=(test_data, y_test))

    # Class weights for balancing
    class_weight = {0: 1., 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=3,
        mode='max',
        restore_best_weights=True)

    # Train model
    history = model.fit(
        train_data,
        y_train,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stop, perf_callback])

    # Evaluate model
    def print_metrics(y_true, y_pred):
        print(f"AUC: {roc_auc_score(y_true, y_pred):.4f}")
        print(f"Log Loss: {log_loss(y_true, y_pred):.4f}")
        print(f"Accuracy: {accuracy_score(y_true, (y_pred > 0.5).astype(int)):.4f}")

    y_pred = model.predict(test_data).flatten()
    print("Test Metrics:")
    print_metrics(y_test, y_pred)