import os

import tensorflow as tf

export_path = "../models/saved_ctr_model/"
version = "2"       #后续可以通过版本号进行模型版本迭代与管理
# 构建完整保存路径
full_export_path = os.path.join(export_path, version)
# 加载模型（验证）
loaded_model = tf.saved_model.load(full_export_path)

# ====================== 使用示例 ======================
# 假设我们有10条测试数据
num_test_samples = 10

# 定义固定测试数据
test_data_tensors = {
    'uid': tf.constant([[111791],[199]], dtype=tf.int32),
    'post_id': tf.constant([[1366],[3]], dtype=tf.int32),
    'hour': tf.constant([[1],[5]], dtype=tf.int32),
    'dow': tf.constant([[1],[5]], dtype=tf.int32),
    'user_ctr': tf.constant([[0.18407348],[0.02057202]], dtype=tf.float32),
    'user_activity': tf.constant([[0.92057202],[0.92057202]], dtype=tf.float32),
    'content_ctr': tf.constant([[1.0],[0.92057202]], dtype=tf.float32),
    'content_impressions': tf.constant([[0.16254509],[0.92057202]], dtype=tf.float32)
}

# 使用模型进行预测
infer = loaded_model.signatures["serving_default"]

predictions = infer(**test_data_tensors)['prediction']
# ['prediction']
print(predictions)
print("Predictions:", predictions.numpy().flatten())