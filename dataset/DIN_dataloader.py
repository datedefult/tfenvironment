from collections import defaultdict, deque
from datetime import datetime
import pymysql

from MulConnectionPool import SynDBPools


class FixedLengthFIFOQueue:
    def __init__(self, length):
        # 初始化定长队列
        self.queue = deque([0] * length, maxlen=length)  # 使用deque并设置最大长度
        self.length = length

    def insert(self, value):
        # 插入新数据，如果队列已满，最早的数据会被自动移除
        self.queue.append(value)

    def get_queue(self):
        # 返回当前队列的状态
        return list(self.queue)[::-1]

    def count_nonzero(self):
        # 统计队列中非0元素的个数
        return sum(1 for item in self.queue if item != 0)



def update_mysql(mysql_conn,data_list):
    # 将数据上传至数据库，限制上限
    exe_query = """
    INSERT INTO tensorflow_dataset_din (sn,uid, hit_post_id_list, hit_len,post_id,label,timestamp) VALUES (%s, %s, %s, %s,%s,%s,%s)
    """
    cursor = mysql_conn.cursor()
    cursor.executemany(exe_query, data_list)
    mysql_conn.commit()
    cursor.close()

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}----成功上传数据！！！")




if __name__ == '__main__':
    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_mysql()

    event_conn = connection_pools.get_mysql('jh_data')
    recommend_conn = connection_pools.get_mysql('content_recommend')

    event_query = """
SELECT
	uid,
	sn,
	event_type,
    sid,
    duration,
    client_time_micro
FROM
	`community202504` 
WHERE
	( event_type = 70 AND sid3 = 1 ) 
	OR ( event_type = 50 )
    """

    MAX_CLICK_LEN = 30
    BATCH_SIZE= 10000
    # sn 字典
    user_sn_dict = defaultdict(str)
    # 用户观看历史列表
    user_queue_dict = defaultdict()

    up_list = []
    with event_conn.cursor(cursor=pymysql.cursors.SSDictCursor) as cursor:
        cursor.execute(event_query)
        for row in cursor:
            uid = row['uid']
            sn = row['sn']
            event_type = int(row['event_type'])
            sid = int(row['sid'])
            duration = row['duration']
            client_time_micro = row['client_time_micro']
            label = 0
            if event_type == 70:
                try:
                    # 获取用户的队列实例
                    user_queue = user_queue_dict[uid]
                    # 获取用户当前点击历史列表
                    now_click_list = user_queue.get_queue()
                    # 统计非零元素的数量
                    nonzero_count = user_queue.count_nonzero()

                except:
                    # 获取异常则先初始化再获取
                    user_queue_dict[uid] = FixedLengthFIFOQueue(length=MAX_CLICK_LEN)
                    now_click_list = [0] * MAX_CLICK_LEN
                    nonzero_count = 0


            elif event_type == 50:

                # 如果 sn 发生变化，重置点击列表
                if sn != user_sn_dict.get(uid):
                    user_queue_dict[uid] = FixedLengthFIFOQueue(length=MAX_CLICK_LEN)
                    user_sn_dict[uid] = sn

                # 获取用户的队列实例
                user_queue = user_queue_dict[uid]

                # 获取用户当前点击历史列表
                now_click_list = user_queue.get_queue()
                if sid in now_click_list:
                    continue
                # 统计非零元素的数量
                nonzero_count = user_queue.count_nonzero()
                # 改变点击标签
                label = 1
                # 插入新的 sid 到队列
                if sid not in now_click_list:
                    user_queue.insert(sid)



            now_click_list = ','.join(map(str, now_click_list))
            clear_row = (sn,uid,str(now_click_list),nonzero_count,sid, label,client_time_micro)
            up_list.append(clear_row)
            if len(up_list) > BATCH_SIZE:
                update_mysql(recommend_conn, up_list)
                up_list = []

        update_mysql(recommend_conn, up_list)


