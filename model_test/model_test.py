import tensorflow as tf

loaded = tf.saved_model.load('../trained_models/linear_model/1')
infer = loaded.signatures['serving_default']
print(infer.structured_outputs)  # 查看输出签名
# 进行预测
result = infer(tf.constant([[1.0, 2.0]]))
print(result)