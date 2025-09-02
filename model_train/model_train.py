import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 必须放在导入 TensorFlow 之前！
import tensorflow as tf
from keras import models,layers,optimizers
# from tensorflow.keras import trained_models,layers,optimizers

## 样本数量
n = 800

## 生成测试用数据集
X = tf.random.uniform([n,2],minval=-10,maxval=10)
w0 = tf.constant([[2.0],[-1.0]])
b0 = tf.constant(3.0)

Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0) # @表示矩阵乘法,增加正态扰动

## 建立模型
tf.keras.backend.clear_session()
inputs = layers.Input(shape = (2,),name ="inputs") #设置输入名字为inputs
outputs = layers.Dense(1, name = "outputs")(inputs) #设置输出名字为outputs
linear = models.Model(inputs = inputs,outputs = outputs)
linear.summary()
linear.summary()
for layer in linear.layers:
    print(layer.name)

## 使用fit方法进行训练
linear.compile(optimizer="rmsprop",loss="mse",metrics=["mae"])
linear.fit(X,Y,batch_size = 8,epochs = 100)

tf.print("w = ",linear.layers[1].kernel)
tf.print("b = ",linear.layers[1].bias)

## 将模型保存成pb格式文件
export_path = "../trained_models/linear_model/"
version = "1"       #后续可以通过版本号进行模型版本迭代与管理

# 构建完整保存路径
full_export_path = os.path.join(export_path, version)

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(full_export_path):
    os.makedirs(full_export_path)

# linear.save(os.path.join(full_export_path,'test.keras'))
tf.saved_model.save(linear,full_export_path)
# linear.save(full_export_path, save_format='tf')
# docker run -t -p 8501:8501 -p 8500:8500 -v E:/pycharmPro/TFenvironment/trained_models/linear_model:/trained_models/linear_model -e MODEL_NAME=linear_model tensorflow/serving:2.18.0

# docker pull tensorflow/serving:2.18.0