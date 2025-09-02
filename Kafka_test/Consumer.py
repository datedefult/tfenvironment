from kafka import KafkaConsumer

# 创建 Kafka 消费者
consumer = KafkaConsumer(
    'apptest',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',  # 从最早的消息开始消费
    enable_auto_commit=True,       # 自动提交偏移量
    group_id='my_group'            # 消费者组 ID
)

print("消费者已启动，正在监听消息...")

try:
    # 持续监听消息
    for message in consumer:
        print(f"收到消息: {message.value.decode('utf-8')}")
        # 模型接收数据进行训练/分发
        # celery 分发



except KeyboardInterrupt:
    print("消费者已停止")
finally:
    # 关闭消费者
    consumer.close()