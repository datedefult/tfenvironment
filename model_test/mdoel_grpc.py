import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np

# TensorFlow Serving的地址和端口
server = 'localhost:8500'  # gRPC默认监听8500端口

# 创建gRPC通道
channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# 创建预测请求
request = predict_pb2.PredictRequest()
request.model_spec.name = 'saved_ctr_model'  # 您的模型名称
request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# 添加输入数据
test_data = {
    'uid': [[111791], [199]],
    'post_id': [[1366], [3]],
    'hour': [[1], [5]],
    'dow': [[1], [5]],
    'user_ctr': [[0.18407348], [0.02057202]],
    'user_activity': [[0.92057202], [0.92057202]],
    'content_ctr': [[1.0], [0.92057202]],
    'content_impressions': [[0.16254509], [0.92057202]]
}

for feature_name, values in test_data.items():
    tensor_proto = tf.make_tensor_proto(np.array(values), dtype=tf.float32 if isinstance(values[0][0], float) else tf.int32)
    request.inputs[feature_name].CopyFrom(tensor_proto)

# 发送请求并获取响应
result_future = stub.Predict(request, 30.0)  # 30秒超时时间

# 输出结果
print("Predictions: ", result_future.outputs)