import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 创建示例用户行为数据
data = {
    "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "visits": [10, 5, 15, 20, 3, 8, 12, 18, 25, 1],
    "purchases": [2, 1, 3, 5, 0, 2, 4, 6, 7, 0],
    "avg_time_spent": [120, 90, 150, 200, 60, 110, 180, 220, 250, 30]
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 2. 数据预处理
# 我们只使用行为特征列（忽略 user_id）
features = df[["visits", "purchases", "avg_time_spent"]]

# 标准化特征（K-means 对特征的尺度敏感）
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 3. 使用 K-means 进行聚类
# 选择聚类的数量 k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

# 4. 可视化聚类结果
# 将聚类结果可视化在二维平面上（降维到 2D）
from sklearn.decomposition import PCA

# 使用 PCA 将特征降到 2 维
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df["cluster"], cmap="viridis", s=100)
plt.title("User Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# 5. 输出聚类结果
print(df)