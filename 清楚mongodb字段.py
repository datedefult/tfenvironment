import pandas as pd
import pymysql.cursors

from MulConnectionPool import SynDBPools


connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
connection_pools.init_mysql()
connection_pools.init_mongo()

app_client = connection_pools.get_mongo('app', 'app')
post_collection = app_client['app_posts']


result = post_collection.update_many(
    {},  # 匹配所有文档
    {"$unset": {"record": ""}}  # 移除 record 键及其所有子字段
)

print(f"已从 {result.modified_count} 条文档中移除 record 字段")

