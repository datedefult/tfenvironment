from MulConnectionPool import SynDBPools

connection_pools = SynDBPools()
connection_pools.init_mysql()
connection_pools.init_redis()
connection_pools.init_mongo()


redis_conn = connection_pools.get_redis('recent_user')
print(redis_conn.keys('*'))

# mysqlc = connection_pools.get_mysql('app_community')

mongo_conn = connection_pools.get_mongo('app_test','app_test')
collection = mongo_conn['app_users']  # 获取集合对象
results = collection.find_one({'uid':111791})  # 查询所有文档
print(results)


with connection_pools.get_mysql('bidata') as conn:
    query = 'SELECT * FROM posts limit 3'
    cursor = conn.cursor()
    cursor.execute(query)
    print(cursor.fetchall())
