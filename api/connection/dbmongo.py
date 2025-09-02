from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import HTTPException

# 声明全局变量
client = None
db = None

# MongoDB URI 示例：
# MONGO_URI = "mongodb://<username>:<password>@<host>:<port>"
MONGO_URI = "mongodb://root:123456@192.168.10.3:27017"
DATABASE_NAME = "app"


async def connect_to_mongo():
    global client, db
    try:
        client = AsyncIOMotorClient(MONGO_URI)
        db = client[DATABASE_NAME]
        print("✅ MongoDB 连接成功")
    except Exception as e:
        raise RuntimeError(f"❌ MongoDB 连接失败: {e}")


async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("🔌 MongoDB 连接已关闭")


def get_mongo_database():
    """提供数据库实例"""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    return db