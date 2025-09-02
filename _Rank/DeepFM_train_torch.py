from dataset.DatasetReturn import data_loader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DeepFM(nn.Module):
    def __init__(self, short_cat_dims, long_cat_dims, num_features_dim, embed_dim, dnn_hidden_units, dropout_rate=0.5):
        super(DeepFM, self).__init__()
        # Embedding layers for long categorical features
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(dim, embed_dim) for col, dim in long_cat_dims.items()
        })
        # Linear layer for one-hot encoded short categorical features
        self.linear_short_cat = nn.Linear(sum(short_cat_dims.values()), 1)
        # Linear layer for numerical features
        self.linear_num = nn.Linear(num_features_dim, 1)
        # FM interaction layer
        self.fm = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in long_cat_dims.values()
        ])
        # DNN layers with Dropout
        dnn_input_dim = len(long_cat_dims) * embed_dim + sum(short_cat_dims.values()) + num_features_dim
        dnn_layers = []
        for units in dnn_hidden_units:
            dnn_layers.append(nn.Linear(dnn_input_dim, units))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))  # 添加 Dropout
            dnn_input_dim = units
        self.dnn = nn.Sequential(*dnn_layers)
        self.dnn_output = nn.Linear(dnn_hidden_units[-1], 1)

    def forward(self, short_cat, long_cat, num_features):
        # Linear part
        linear_short = self.linear_short_cat(short_cat)
        linear_num = self.linear_num(num_features)
        linear_part = linear_short + linear_num

        # FM part
        fm_embeddings = [self.embeddings[col](long_cat[:, i]) for i, col in enumerate(self.embeddings.keys())]
        fm_sum_square = torch.sum(torch.stack(fm_embeddings), dim=0) ** 2
        fm_square_sum = torch.sum(torch.stack([x ** 2 for x in fm_embeddings]), dim=0)
        fm_part = 0.5 * (fm_sum_square - fm_square_sum).sum(dim=1, keepdim=True)

        # DNN part
        dnn_input = torch.cat([short_cat, num_features] + fm_embeddings, dim=1)
        dnn_part = self.dnn_output(self.dnn(dnn_input))

        # Output
        return linear_part + fm_part + dnn_part

# 数据预处理
def preprocess_data(dataset, short_cat_cols, long_cat_cols, num_cols):
    # One-hot encode short categorical features
    short_cat_encoder = OneHotEncoder(sparse_output=False)  # 修复参数名
    short_cat_data = short_cat_encoder.fit_transform(dataset[short_cat_cols])

    # Embedding encode long categorical features (convert to integer indices)
    long_cat_data = dataset[long_cat_cols].apply(lambda x: x.astype('category').cat.codes).values
    long_cat_dims = {col: dataset[col].nunique() for col in long_cat_cols}

    # Normalize numerical features
    scaler = MinMaxScaler()
    num_data = scaler.fit_transform(dataset[num_cols])

    return short_cat_data, long_cat_data, num_data, long_cat_dims

if __name__ == '__main__':

    dataset = data_loader()

    # 有限个类别
    short_cat_cols = ['hour','dow','gender','language','channels','author_language']
    # 几百个类别
    long_cat_cols = ['country_code','author_country_code']
    # id特征
    id_cols =  ['uid', 'author_id', 'post_id']
    # 各类数值特征
    num_cols = ['friends_num','binding_toys_number','hits_rate','like_rate','collect_rate','comments_rate','score']
    # CTR预测目标
    ctr_tag = ['is_hit']

    # 数据预处理
    short_cat_data, long_cat_data, num_data, long_cat_dims = preprocess_data(
        dataset, short_cat_cols, long_cat_cols, num_cols
    )
    short_cat_dims = {col: len(dataset[col].unique()) for col in short_cat_cols}

    # 计算正负样本权重
    positive_count = (dataset[ctr_tag[0]] == 1).sum()
    negative_count = (dataset[ctr_tag[0]] == 0).sum()
    pos_weight = negative_count / positive_count

    # 转换为Tensor
    short_cat_tensor = torch.tensor(short_cat_data, dtype=torch.float32)
    long_cat_tensor = torch.tensor(long_cat_data, dtype=torch.long)
    num_tensor = torch.tensor(num_data, dtype=torch.float32)
    target_tensor = torch.tensor(dataset[ctr_tag].values, dtype=torch.float32)

    # 数据加载
    dataset = TensorDataset(short_cat_tensor, long_cat_tensor, num_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型定义
    model = DeepFM(short_cat_dims, long_cat_dims, len(num_cols), embed_dim=8, dnn_hidden_units=[64, 32], dropout_rate=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))  # 修复权重传递
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_auc = 0.0
    train_losses = []
    val_aucs = []
    patience = 5  # 早停策略的耐心值
    patience_counter = 0

    # 训练
    for epoch in range(500):  # 增加最大训练轮数
        model.train()
        epoch_loss = 0.0
        for short_cat, long_cat, num_features, target in dataloader:
            optimizer.zero_grad()
            output = model(short_cat, long_cat, num_features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(dataloader))

        # 验证（假设有验证集 dataloader_val）
        model.eval()
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for short_cat, long_cat, num_features, target in dataloader:  # 替换为 dataloader_val
                output = model(short_cat, long_cat, num_features)
                all_outputs.extend(torch.sigmoid(output).cpu().numpy())  # 使用 Sigmoid 激活
                all_targets.extend(target.cpu().numpy())
        auc = roc_auc_score(all_targets, all_outputs)
        val_aucs.append(auc)

        # 调整学习率
        scheduler.step(auc)

        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_deepfm_model.pth")
            patience_counter = 0  # 重置早停计数器
        else:
            patience_counter += 1

        print(f"Epoch {epoch + 1}, Loss: {train_losses[-1]:.4f}, AUC: {auc:.4f}")

        # 检查早停条件
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 绘制 Loss 和 AUC 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_aucs) + 1), val_aucs, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()