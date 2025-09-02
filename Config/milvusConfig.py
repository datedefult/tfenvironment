from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient

# ************************  milvus数据库配置
# MILVUS_HOST = '127.0.0.1'
MILVUS_HOST = '192.168.10.3'
MILVUS_PORT = '19530'
MILVUS_DB_NAME = 'default'
MILVUS_COLLECTION = 'app_posts_embedding'
# 文本嵌入维度
MILVUS_SENTENCE_DIM = 384
# 图像嵌入维度
MILVUS_IMAGE_DIM = 512
# 返回结果长度
MILVUS_LIMIT = 300
