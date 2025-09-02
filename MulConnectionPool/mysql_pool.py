import threading
from collections import deque
from time import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pymysql
from utils.LogsColor import logging


@dataclass
class PoolStats:
    max_size: int
    current_size: int
    idle: int
    active: int


class MySQLConnectionPool:
    def __init__(self, **config):
        """
        基础MySQL连接池
        :param config: dict - 配置字典，包含以下键:
            - host: str - 数据库地址
            - user: str - 用户名
            - password: str - 密码
            - database: str - 数据库名
            - port: int - 端口 (默认: 3306)
            - autocommit: bool - 是否自动提交 (默认: True)
            - pool_size: int - 连接池大小 (默认: 10)
            - timeout: int - 获取连接超时 (秒, 默认: 30)
        """
        self._config = {
            "host": config.get("host"),
            "user": config.get("user"),
            "password": config.get("password"),
            "database": config.get("database"),
            "port": config.get("port", 3306),
            "autocommit": config.get("autocommit", True),
        }
        self.max_size = config.get("pool_size", 10)
        self.timeout = config.get("timeout", 30)

        self._idle_connections = deque()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._current_size = 0
        self._all_connections = set()

    def _create_connection(self):
        """创建新连接"""
        try:
            return pymysql.connect(**self._config, cursorclass=pymysql.cursors.SSCursor)
        except Exception as e:
            logging.error(f"MySQL连接创建失败: {e}")
            raise

    def _validate_connection(self, conn):
        """验证连接是否有效"""
        try:
            return conn.open
        except Exception:
            return False

    def get_connection(self):
        """获取数据库连接"""
        start_time = time()

        with self._condition:
            while True:
                try:
                    # 尝试复用空闲连接
                    if self._idle_connections:
                        conn = self._idle_connections.pop()
                        if self._validate_connection(conn):
                            return conn
                        self._current_size -= 1
                        conn.close()

                    # 创建新连接
                    if self._current_size < self.max_size:
                        conn = self._create_connection()
                        self._current_size += 1
                        self._all_connections.add(conn)
                        return conn

                    # 等待可用连接
                    timeout_left = self.timeout - (time() - start_time)
                    if timeout_left <= 0:
                        raise TimeoutError("MulConnectionPool pool exhausted")

                    if not self._condition.wait(timeout=timeout_left):
                        raise TimeoutError("MulConnectionPool wait timeout")
                except Exception as e:
                    logging.error(f"获取MySQL连接失败: {e}")
                    raise

    def release_connection(self, conn):
        """释放连接回池"""
        try:
            with self._lock:
                if conn in self._all_connections:
                    if self._validate_connection(conn):
                        self._idle_connections.append(conn)
                    else:
                        conn.close()
                        self._current_size -= 1
                        self._all_connections.remove(conn)
                    self._condition.notify()
            logging.info("MySQL连接已释放")
        except Exception as e:
            logging.error(f"释放MySQL连接失败: {e}")
            raise

    def close_all(self):
        """关闭所有连接"""
        try:
            with self._lock:
                while self._idle_connections:
                    conn = self._idle_connections.pop()
                    conn.close()
                for conn in list(self._all_connections):
                    conn.close()
                    self._all_connections.remove(conn)
                self._current_size = 0
        except Exception as e:
            logging.error(f"关闭MySQL连接池失败: {e}")
            raise

    @property
    def stats(self) -> PoolStats:
        """获取连接池状态"""
        with self._lock:
            return PoolStats(
                max_size=self.max_size,
                current_size=self._current_size,
                idle=len(self._idle_connections),
                active=self._current_size - len(self._idle_connections),
            )


class MysqlPoolManager:
    def __init__(self):
        self._pools: Dict[str, MySQLConnectionPool] = {}
        self._default_alias: Optional[str] = None
        self._lock = threading.Lock()

    def init_pool(self, alias: str, config: Dict[str, Any], is_default: bool = False) -> None:
        """
        初始化数据库连接池
        :param alias: str - 连接池别名
        :param config: Dict[str, Any] - 连接配置字典
        :param is_default: bool - 是否设为默认连接池 (默认: False)
        :return: None
        """
        with self._lock:
            if alias in self._pools:
                raise ValueError(f"连接池 '{alias}' 已存在")

            # 验证必要配置参数
            required_keys = {"host", "user", "password", "database"}
            if not required_keys.issubset(config.keys()):
                missing = required_keys - config.keys()
                raise ValueError(f"缺少必要配置参数: {missing}")

            self._pools[alias] = MySQLConnectionPool(**config)
            logging.info(f"MySQL连接池初始化成功: {alias}")
            # 设置默认连接池
            if is_default or not self._default_alias:
                self._default_alias = alias

    def get_pool(self, alias: Optional[str] = None) -> MySQLConnectionPool:
        """
        获取指定连接池实例
        :param alias: Optional[str] - 连接池别名，None表示使用默认连接池
        :return: MySQLConnectionPool - 连接池实例
        :raises ValueError: 如果指定的连接池不存在
        """
        pool_alias = alias or self._default_alias
        if pool_alias is None:
            raise ValueError("未指定连接池别名且无默认连接池")
        if pool_alias not in self._pools:
            raise ValueError(f"未配置的连接池: '{pool_alias}'")
        return self._pools[pool_alias]

    def get_connection(self, alias: Optional[str] = None):
        """
        获取数据库连接
        :param alias: Optional[str] - 连接池别名，None表示使用默认连接池
        :return: MulConnectionPool - 可用的数据库连接
        """
        return self.get_pool(alias).get_connection()

    def release_connection(self, conn, alias: Optional[str] = None) -> None:
        """
        释放连接回池
        :param conn: MulConnectionPool - 要释放的连接
        :param alias: Optional[str] - 连接池别名，None表示使用默认连接池
        :return: None
        """
        self.get_pool(alias).release_connection(conn)

    def close_all_pools(self) -> None:
        """关闭并清除所有连接池"""
        try:
            with self._lock:
                for pool in self._pools.values():
                    pool.close_all()
                self._pools.clear()
                self._default_alias = None
            logging.info("所有MySQL连接已关闭")
        except Exception as e:
            logging.error(f"关闭MySQL连接失败: {e}")
            raise

    def get_pool_stats(self, alias: Optional[str] = None) -> PoolStats:
        """
        获取指定连接池状态
        :param alias: Optional[str] - 连接池别名，None表示使用默认连接池
        :return: PoolStats - 连接池状态数据类实例
        """
        return self.get_pool(alias).stats
