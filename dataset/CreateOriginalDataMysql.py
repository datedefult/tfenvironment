from datetime import datetime
import redis
import pymysql
import pytz

from BaseListen.mul_base_listen import BaseMySQLListenerWithCheckpointMUL
from Config.mysqlConfig import user_event_TB,content_recommend_TB
from Config.redisConfig import REDIS_USER_COMMUNITY_PUSH
from utils.LogsColor import logging
from utils.Tools import other2int


"""
CREATE TABLE `tensorflow_dataset` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '自增',
  `timestamp` bigint DEFAULT NULL COMMENT '请求发生时间戳',
  `uid` int DEFAULT NULL COMMENT '用户id',
  `post_id` int DEFAULT NULL COMMENT '帖子id',
  `is_show` tinyint DEFAULT NULL COMMENT '是否展现',
  `is_hit` tinyint DEFAULT NULL COMMENT '是否点击',
  `is_comment` tinyint DEFAULT NULL COMMENT '是否评论',
  `is_like` tinyint DEFAULT NULL COMMENT '是否点赞',
  `is_collect` tinyint DEFAULT NULL COMMENT '是否收藏',
  `is_play` tinyint DEFAULT NULL COMMENT '是否播放波形',
  `is_download` tinyint DEFAULT NULL COMMENT '是否下载波形',
  `is_click_link` tinyint DEFAULT NULL COMMENT '是否点击链接',
  `stay_duration` int DEFAULT NULL COMMENT '帖子停留时长，ms',
  `play_duration` int DEFAULT NULL COMMENT '波形使用时长，ms',
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_index` (`timestamp`,`uid`,`post_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='深度学习基础数据';

"""



def redis_client_create(client_url):
    # 创建redis链接
    try:
        redis_client = redis.from_url(client_url)
        return redis_client
    except Exception as e:
        return



recommend_conn = pymysql.connect(**content_recommend_TB, cursorclass=pymysql.cursors.DictCursor,autocommit=True)
uid_post_client = redis_client_create(REDIS_USER_COMMUNITY_PUSH)


def make_key(uid, post_id):
    """生成唯一键"""
    return f"{uid}_{post_id}"

def set_timestamp(client,uid, post_id, timestamp):
    """设置或更新 timestamp"""
    key = make_key(uid, post_id)
    client.set(key, timestamp)

def get_timestamp(client,uid, post_id):
    """根据 uid 和 post_id 获取 timestamp"""
    key = make_key(uid, post_id)
    value = client.get(key)
    if value is None:
        return None
    else:
        return value


def return_max_timestamp(mysql_cursor,uid, post_id):
    select_query = """
    SELECT MAX(`timestamp`) AS max_timestamp
    FROM `tensorflow_dataset`
    WHERE `uid` = %s AND `post_id` = %s;
    """
    mysql_cursor.execute(select_query, (uid, post_id))
    result = mysql_cursor.fetchone()
    max_timestamp = result.get('max_timestamp')
    return max_timestamp

def deal_with_row(mysql_cursor,row):
    row_id = row['id']
    uid = int(row['uid'])
    event_type = int(row['event_type'])
    post_id = other2int(row['sid'])
    client_time_micro=row['client_time_micro']
    sid2 = row['sid2']
    duration = row['duration']

    SQL_TEMPLATES = {
        70: """INSERT INTO tensorflow_dataset (timestamp,uid, post_id, is_show,is_hit,is_comment,is_like,is_collect,is_play,is_download,is_click_link,stay_duration,play_duration) VALUES (%s, %s, %s,%s,%s, %s, %s,%s,%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE is_show = VALUES(is_show)""",
        50: """UPDATE tensorflow_dataset SET is_hit=%s,stay_duration = stay_duration + %s WHERE `timestamp`=%s AND uid=%s AND post_id=%s""",
        67: """UPDATE tensorflow_dataset SET is_comment=%s WHERE `timestamp`=%s AND uid=%s AND post_id=%s """,
        61: """UPDATE tensorflow_dataset SET is_like = is_like {} 1  WHERE `timestamp`=%s AND uid=%s AND post_id=%s""",
        62: """UPDATE tensorflow_dataset SET is_collect = is_collect {} 1 WHERE `timestamp`=%s AND uid=%s AND post_id=%s """,
        52: """UPDATE tensorflow_dataset SET is_play=%s,play_duration = play_duration + %s WHERE `timestamp`=%s AND uid=%s AND post_id=%s """,
        54: """UPDATE tensorflow_dataset SET is_download=%s WHERE `timestamp`=%s AND uid=%s AND post_id=%s""",
        51: """UPDATE tensorflow_dataset SET is_click_link=%s WHERE `timestamp`=%s AND uid=%s AND post_id=%s"""
    }

    if event_type == 70:
        mysql_cursor.execute(SQL_TEMPLATES[70], (client_time_micro,uid,post_id,1,0,0,0,0,0,0,0,0,0))
        set_timestamp(uid_post_client,uid,post_id,client_time_micro)
    elif event_type in ( 67, 54, 51):
        max_timestamp = get_timestamp(uid_post_client,uid,post_id)
        if max_timestamp is not None:  # 修正判断逻辑
            # logging.info(f'点击时间:{max_timestamp}，帖子:{post_id}，用户：{uid}，当前行号：{row_id}，处理事件：{event_type}')
            mysql_cursor.execute(SQL_TEMPLATES[event_type], (1,max_timestamp, uid, post_id))
    elif event_type in (50,52):
        max_timestamp = get_timestamp(uid_post_client,uid,post_id)
        if max_timestamp is not None:  # 修正判断逻辑
            # logging.info(f'点击时间:{max_timestamp}，帖子:{post_id}，用户：{uid}，当前行号：{row_id}，处理事件：{event_type}')
            mysql_cursor.execute(SQL_TEMPLATES[event_type], (1,duration,max_timestamp, uid, post_id))

    elif event_type in (61, 62):
        max_timestamp = get_timestamp(uid_post_client, uid, post_id)
        if max_timestamp is not None:
            # 动态生成增减语句
            # logging.info(f'点击时间:{max_timestamp}，帖子:{post_id}，用户：{uid}，当前行号：{row_id}，处理事件：{event_type}')
            operator = '+' if int(sid2) == 1 else '-'
            adjusted_sql = SQL_TEMPLATES[event_type].format(operator)
            mysql_cursor.execute(adjusted_sql, (max_timestamp, uid, post_id))





class CommunityListener(BaseMySQLListenerWithCheckpointMUL):
    def fetch_changes(self):
        """查询数据库中自上次查询以来（checkpoint 之后）有变化的记录"""
        # self.current_month = (datetime.now(pytz.timezone('America/New_York'))).strftime("%Y%m")
        # self.current_month = '202501'
        try:
            with self.connection.cursor() as cursor:
                query = f"""
                    SELECT * FROM {self.table_name}
                    WHERE event_type IN (70,50,61,62,67,51,52,54)
                    AND id > %s 
                    ORDER BY `client_time_micro` ASC
                    LIMIT 2000000;
                """

                logging.info(f"Now process :{self.table_name}, {self.checkpoint}")
                cursor.execute(query, (self.checkpoint,))
                rows = cursor.fetchall()

                if len(rows) == 0:
                    logging.info(f"{self.table_name} Not found new changes!")
                    # self.current_month = str(int(self.current_month) + 1)

                if len(rows) == 0 and self.current_month != self.previous_month:
                    self.checkpoint = 0
                    self.previous_month = self.current_month
                    self.table_name = f"{self.table_prefix}{self.current_month}"  # 动态表名
                    self.offset_file = f"./Logs/offsetLogs/offset_{self.table_name}.txt"  # 动态偏移文件名

                return rows
        except Exception as e:
            self.error_chat(e)
            logging.error(f"Error fetching changes: {e}")
            self.stop()
            return []

    def process_changes(self, rows):
        """处理查询到的数据"""
        self.err_count = 0
        if rows:
            logging.info(f"Found {len(rows)} new changes in {self.table_name} !")
            total_rows = len(rows)  # 获取总行数
            processed_count=0
            batch_size = 10000
            with recommend_conn.cursor() as cursor:
                cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
                for row in rows:
                    deal_with_row(cursor,row)
                    processed_count += 1

                    # 每处理一定数量的行打印一次进度（例如每100行）
                    if processed_count % batch_size == 0:
                        recommend_conn.commit()
                        logging.info(f"Processed {processed_count} rows: [{processed_count}/{total_rows}]")
                        recommend_conn.begin()  # 新事务
                logging.info(f"Processed {processed_count} rows: [{processed_count}/{total_rows}]")
                recommend_conn.commit()  # 提交剩余记录
            try:
                # 获取最大 id，并更新 checkpoint
                max_id = max(row['id'] for row in rows)
                if max_id > self.checkpoint:
                    self.checkpoint = max_id
                    self.write_checkpoint()  # 将新的 checkpoint 写入文件

            except Exception as e:
                logging.error(f"Error during update in {self.table_name} : {e}")





# 使用示例
if __name__ == "__main__":
    community_listener = CommunityListener(
        **user_event_TB,
        table_prefix='community',
        query_interval=300  # 每 n 秒查询一次
    )
    try:
        # 外部启动监听
        community_listener.start_listening()
        # community_listener.stop()
    except Exception as e:
        community_listener.stop()
