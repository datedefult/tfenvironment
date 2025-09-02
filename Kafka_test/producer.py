import json

import pymysql.cursors
from kafka import KafkaProducer

from MulConnectionPool import SynDBPools



# 发送消息到 apptest 主题
def send_message(topic, message):
    producer.send(topic, value=message)
    producer.flush()  # 确保所有消息都已发送

def get_init_data(mysql_conn):
    # 获取数据库中的数据
    with mysql_conn.cursor(cursor=pymysql.cursors.SSDictCursor) as cursor:
        cursor.execute("SELECT * FROM `tensorflow_dataset_din`")
        for row in cursor:
            yield row

if __name__ == "__main__":
    # 创建 Kafka 生产者
    producer = KafkaProducer(bootstrap_servers='localhost:9092',
                             value_serializer=lambda v: json.dumps(v).encode('utf-8')  # 将消息序列化为 JSON
                             )

    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_mysql()
    recommend_conn = connection_pools.get_mysql('content_recommend')

    topic = 'apptest'
    for index, value in enumerate(get_init_data(recommend_conn)):
        print(f"Processing item {index}: {value}")  # 打印索引和值  # 打印当前值
        send_message(topic, value)  # 发送消息

    producer.close()  # 关闭生产者
