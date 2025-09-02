from pymongo import MongoClient
from pymongo.errors import PyMongoError
from typing import Dict, Optional
import contextlib
from utils.LogsColor import logging


class MongoPoolManager:
    def __init__(self):
        self.clients: Dict[str, MongoClient] = {}

    def init_pool(self, alias: str, config: dict):
        """
        初始化MongoDB连接池
        :param alias: str - 连接池别名
        :param config: dict - 配置字典，包含以下键:
            - host: str - MongoDB主机地址 (默认: "localhost")
            - port: int - MongoDB端口号 (默认: 2706)
            - username: str - 用户名 (可选)
            - password: str - 密码 (可选)
            - maxPoolSize: int - 最大连接池大小 (默认: 10)
            - connectTimeoutMS: int - 连接超时时间 (毫秒, 默认: 20000)
        :return: None
        """
        try:
            connection_str = self._build_connection_string(config)
            self.clients[alias] = MongoClient(
                connection_str,
                maxPoolSize=config.get("maxPoolSize", 10),
                connectTimeoutMS=config.get("connectTimeoutMS", 20000),
            )
            logging.info(f"MongoDB连接池初始化成功: {alias}")
        except Exception as e:
            logging.error(f"MongoDB连接池初始化失败: {alias}, 错误: {e}")
            raise

    def _build_connection_string(self, config: dict) -> str:
        """构建MongoDB连接字符串"""
        host = config.get("host", "localhost")
        port = config.get("port", 2706)
        username = config.get("username")
        password = config.get("password")

        if username and password:
            return f"mongodb://{username}:{password}@{host}:{port}/"
        return f"mongodb://{host}:{port}"

    def get_client(self, alias: str = "default") -> MongoClient:
        """
        获取MongoDB客户端
        :param alias: str - 连接池别名 (默认: "default")
        :return: MongoClient - MongoDB客户端实例
        :raises KeyError: 如果指定的连接池别名未初始化
        """
        try:
            if alias not in self.clients:
                raise KeyError(f"MongoDB客户端 '{alias}' 未初始化")
            return self.clients[alias]
        except Exception as e:
            logging.error(f"获取MongoDB客户端失败: {alias}, 错误: {e}")
            raise

    def get_db(self, alias: str = "default", db_name: Optional[str] = None):
        """
        获取数据库实例
        :param alias: str - 连接池别名 (默认: "default")
        :param db_name: Optional[str] - 数据库名称 (如果为None，则使用默认数据库)
        :return: Database - 数据库实例
        """
        try:
            client = self.get_client(alias)
            return client.get_database(db_name)
        except Exception as e:
            logging.error(f"获取MongoDB数据库失败: {alias}, 错误: {e}")
            raise

    @contextlib.contextmanager
    def client_session(self, alias: str = "default"):
        """
        上下文管理器获取客户端会话
        :param alias: str - 连接池别名 (默认: "default")
        :yield: MongoClient - MongoDB客户端实例
        :raises PyMongoError: 如果MongoDB操作失败
        """
        client = self.get_client(alias)
        try:
            yield client
        except PyMongoError as e:
            logging.error(f"MongoDB操作失败: {e}")
            raise

    def close_all_pools(self):
        """
        关闭所有连接
        :return: None
        """
        try:
            for client in self.clients.values():
                client.close()
            self.clients.clear()
            logging.info("所有MongoDB连接已关闭")
        except Exception as e:
            logging.error(f"关闭MongoDB连接失败: {e}")
            raise