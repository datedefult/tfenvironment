from pymongo import MongoClient, UpdateOne
import warnings

warnings.filterwarnings("ignore")

# 连接到MongoDB
MongoDB_url = "mongodb://root:123456@192.168.10.3:27017/"
MongoDB_client = MongoClient(MongoDB_url, maxPoolSize=50, minPoolSize=1)
# db = MongoDB_client["app_community"]
db = MongoDB_client["app"]
user_collection = db["app_users"]
post_collection = db["app_posts"]
deleted_collection = db["app_post_deleted"]
# content_embedding_collection = db["content_embedding"]

try:
    # 创建唯一索引
    user_collection.create_index([("uid", 1)], unique=True)
except Exception as e:
    pass

try:
    # 创建唯一索引
    post_collection.create_index([("post_id", 1)], unique=True)
except Exception as e:
    pass

try:
    # 创建唯一索引
    deleted_collection.create_index([("post_id", 1)], unique=True)
except Exception as e:
    pass
