import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from typing import Dict
import yaml
import os

_redis_clients: Dict[str, redis.Redis] = {}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "redis_config.yaml")
_is_initialized = False

def load_redis_config(path=CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Redis 配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def init_redis_pools():
    global _redis_clients, _is_initialized
    if _is_initialized:
        return
    config = load_redis_config()

    for name, conf in config.items():
        pool = ConnectionPool(
            host=conf.get("host", "localhost"),
            port=conf.get("port", 6379),
            db=conf.get("db", 0),
            decode_responses=True,
            max_connections=conf.get("max_connections", 10),
        )
        client = redis.Redis(connection_pool=pool)
        try:
            await client.ping()
            _redis_clients[name] = client
            print(f"✅ Redis[{name}] 连接成功")
            _is_initialized = True
        except Exception as e:
            print(f"❌ Redis[{name}] 连接失败: {e}")
            raise

async def close_redis_pools():
    for name, client in _redis_clients.items():
        await client.close()
        print(f"🔌 Redis[{name}] 已关闭")
    _redis_clients.clear()

def get_redis(name="default") -> redis.Redis:
    if name not in _redis_clients:
        raise ValueError(f"Redis 实例 `{name}` 未初始化")
    return _redis_clients[name]
