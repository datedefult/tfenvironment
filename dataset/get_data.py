from datetime import datetime
import redis
import pymysql
import pytz

from BaseListen.mul_base_listen import BaseMySQLListenerWithCheckpointMUL
from Config.mysqlConfig import user_event_TB,content_recommend_TB
from Config.redisConfig import REDIS_USER_COMMUNITY_PUSH
from utils.LogsColor import logging
from utils.Tools import other2int

def return_row_data():
    """

    :return:
    """
    recommend_conn = pymysql.connect(**content_recommend_TB, cursorclass=pymysql.cursors.SSCursor,autocommit=True)
    query_train_dataset = "SELECT `uid`,`post_id`,`is_show`,`is_hit` FROM `tensorflow_dataset` limit 500"
    # query_train_dataset = "SELECT * FROM `tensorflow_dataset` limit 500"
    with recommend_conn.cursor() as cursor:
        cursor.execute(query_train_dataset)
        for row in cursor:
            yield row
        # yield cursor.fetchall()
