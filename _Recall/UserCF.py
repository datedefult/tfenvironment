import numpy as np
import redis
from collections import defaultdict
import multiprocessing

from Config.redisConfig import REDIS_USER_COMMUNITY_CLICK
from utils.LogsColor import logging

from MulConnectionPool import SynDBPools

if __name__ == '__main__':
    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_redis()
    redis_post_click_history = connection_pools.get_redis('post_click_history')
    user_sid_dict = {int(key): list(map(int, redis_post_click_history.lrange(key, 0, -1))) for key in
                     redis_post_click_history.keys('*')}

    print(user_sid_dict)