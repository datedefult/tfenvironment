import redis
from redis.connection import ConnectionPool
from typing import Dict
import contextlib
from utils.LogsColor import logging


class RedisPoolManager:
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}

    def init_pool(self, alias: str, config: dict):
        """
        初始化Redis连接池
        :param alias: str - 连接池别名
        :param config: dict - 配置字典，包含以下键:
            - host: str - Redis主机地址 (默认: "localhost")
            - port: int - Redis端口号 (默认: 6379)
            - db: int - 数据库编号 (默认: 0)
            - password: str - Redis密码 (可选)
            - max_connections: int - 最大连接数 (默认: 20)
            - socket_timeout: float - 套接字超时时间 (可选)
            - socket_connect_timeout: float - 套接字连接超时时间 (可选)
        :return: None
        """
        try:
            self.pools[alias] = ConnectionPool(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                db=config.get("db", 0),
                password=config.get("password"),
                max_connections=config.get("max_connections", 20),
                socket_timeout=config.get("socket_timeout"),
                socket_connect_timeout=config.get("socket_connect_timeout"),
                decode_responses=True,
            )
            logging.info(f"Redis连接池初始化成功: {alias}")
        except Exception as e:
            logging.error(f"Redis连接池初始化失败: {alias}, 错误: {e}")
            raise

    def get_connection(self, alias: str = "default") -> redis.Redis:
        """
        获取Redis连接
        :param alias: str - 连接池别名 (默认: "default")
        :return: redis.Redis - Redis连接实例
        :raises KeyError: 如果指定的连接池别名未初始化
        """
        try:
            if alias not in self.pools:
                raise KeyError(f"Redis连接池 '{alias}' 未初始化")
            return redis.Redis(connection_pool=self.pools[alias])
        except Exception as e:
            logging.error(f"获取Redis连接失败: {alias}, 错误: {e}")
            raise

    @contextlib.contextmanager
    def connection(self, alias: str = "default"):
        """
        上下文管理器获取连接
        :param alias: str - 连接池别名 (默认: "default")
        :yield: redis.Redis - Redis连接实例
        """
        conn = self.get_connection(alias)
        try:
            yield conn
        finally:
            # Redis连接池会自动管理连接
            pass

    def close_all_pools(self):
        """
        关闭所有连接池
        :return: None
        """
        try:
            for pool in self.pools.values():
                pool.disconnect()
            self.pools.clear()
            logging.info("所有Redis连接池已关闭")
        except Exception as e:
            logging.error(f"关闭Redis连接池失败: {e}")
            raise