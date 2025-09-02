import yaml
from pathlib import Path
from typing import Optional
from MulConnectionPool.mongo_pool import MongoPoolManager
from MulConnectionPool.mysql_pool import MysqlPoolManager
from MulConnectionPool.redis_pool import RedisPoolManager
from utils.LogsColor import logging


class SynDBPools:
    def __init__(self, config_dir: str = "MulConnectionPool/config"):
        self.mysql_manager = MysqlPoolManager()
        self.redis_manager = RedisPoolManager()
        self.mongo_manager = MongoPoolManager()
        self.config_dir = Path(config_dir)

    def _load_yaml_config(self, filename: str) -> dict:
        """加载YAML配置文件"""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                logging.info(f"加载配置文件: {config_path}")
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"加载配置文件失败: {config_path}, 错误: {e}")
            raise

    def init_mysql(self):
        """ 从YAML文件初始化MySQL连接 """
        try:
            config = self._load_yaml_config("mysql_config.yaml")
            for alias, cfg in config["mysql"].items():
                is_default = cfg.pop("is_default", False)  # 移除is_default并获取值
                if not self.mysql_manager._pools.get(alias):  # 检查是否已存在
                    self.mysql_manager.init_pool(alias, cfg, is_default=is_default)
        except Exception as e:
            logging.error(f"MySQL连接池初始化失败: {e}")
            raise

    def init_redis(self):
        """ 初始化多个Redis连接 """
        try:
            config = self._load_yaml_config("redis_config.yaml")
            for alias, cfg in config["redis"].items():
                self.redis_manager.init_pool(alias, cfg)
        except Exception as e:
            logging.error(f"Redis连接池初始化失败: {e}")
            raise

    def init_mongo(self):
        """ 初始化多个MongoDB连接 """
        try:
            config = self._load_yaml_config("mongo_config.yaml")
            for alias, cfg in config["mongo"].items():
                self.mongo_manager.init_pool(alias, cfg)
        except Exception as e:
            logging.error(f"MongoDB连接池初始化失败: {e}")
            raise

    def get_mysql(self, alias: str = 'app_community'):
        try:
            return self.mysql_manager.get_connection(alias)
        except Exception as e:
            logging.error(f"获取MySQL连接失败: {alias}, 错误: {e}")
            raise

    def get_redis(self, alias: str = 'cache'):
        """ 获取Redis连接 """
        try:
            return self.redis_manager.get_connection(alias)
        except Exception as e:
            logging.error(f"获取Redis连接失败: {alias}, 错误: {e}")
            raise

    def get_mongo(self, alias: str = 'user_data', db_name: Optional[str] = None):
        """ 获取MongoDB数据库实例 """
        try:
            return self.mongo_manager.get_db(alias, db_name)
        except Exception as e:
            logging.error(f"获取MongoDB连接失败: {alias}, 错误: {e}")
            raise

    def close(self):
        """
        关闭所有数据库连接池
        :return: None
        """
        try:
            self.mysql_manager.close_all_pools()
            self.redis_manager.close_all_pools()
            self.mongo_manager.close_all_pools()
            logging.info("所有数据库连接池已关闭")
        except Exception as e:
            logging.error(f"关闭数据库连接池失败: {e}")
            raise


