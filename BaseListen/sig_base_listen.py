from datetime import datetime
import time
import threading
import pymysql
import os
import json

import requests

from utils.LogsColor import logging


class BaseMySQLListenerWithCheckpointSIG:
    """按照月进行分表的表结构数据跟踪基类"""

    def __init__(self, host, port, user, password, database, table_name, query_interval=600, max_err_count=10):
        """初始化方法：数据库/偏移文件/断点信息/控制标志/预警等信息"""
        # 初始化数据库连接参数
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name
        self.query_interval = query_interval  # 查询间隔

        self.offset_file = f"./Logs/offsetLogs/offset_{self.table_name}.txt"  # 动态偏移文件名

        # 初始化断点和控制标志
        self.checkpoint = self.read_checkpoint()
        self.stop_flag = False

        # 创建数据库连接
        self.connection = self.create_connection()

        # 初始化空值次数并设置预警
        self.err_count = 0
        self.max_err_count = max_err_count
        self.error_ = None
        # 初始化线程
        self.listen_thread = None

    def create_connection(self):
        """创建并返回持久化的数据库连接"""
        connection = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            cursorclass=pymysql.cursors.DictCursor
        )

        # 设置隔离级别为 READ COMMITTED
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
        return connection

    def read_checkpoint(self):
        """读取文件中的偏移值，如果文件不存在则返回 0"""
        if os.path.exists(self.offset_file):
            with open(self.offset_file, 'r') as f:
                checkpoint = f.read().strip()
                return int(checkpoint) if checkpoint.isdigit() else 0
        else:
            # 如果文件不存在，则创建文件并返回偏移值 0
            with open(self.offset_file, 'w') as f:
                f.write('0')
            return 0

    def write_checkpoint(self):
        """将当前的 checkpoint 写入文件"""
        with open(self.offset_file, 'w') as f:
            f.write(str(self.checkpoint))

    def set_checkpoint(self, v):
        """设置监听的断点，并将其存储"""
        self.checkpoint = v
        self.write_checkpoint()  # 可选择持久化保存到文件

    def get_checkpoint(self):
        """获取当前的断点"""
        return self.checkpoint

    def fetch_changes(self):
        """查询数据库中自上次查询以来（checkpoint 之后）有变化的记录
        自定义查询方法
        """
        self.current_month = datetime.now().strftime("%Y%m")  # 格式化为 YYYYMM
        try:
            with self.connection.cursor() as cursor:
                query = f"""
                    SELECT * FROM {self.table_name}
                    WHERE id > %s limit 100
                """
                cursor.execute(query, (self.checkpoint,))
                rows = cursor.fetchall()

                if len(rows) == 0:
                    logging.info(f"{self.table_name} Not found new changes!")

                return rows
        except Exception as e:
            logging.error(f"Error fetching changes: {e}")
            return []

    def process_changes(self, rows):
        """处理查询到的数据 自定义操作函数"""
        self.err_count = 0
        logging.info(f"Found {len(rows)} new changes!")
        # for row in rows:
        #     print(row)

        # 获取最大 id，并更新 checkpoint
        max_id = max(row['id'] for row in rows)
        if max_id > self.checkpoint:
            self.checkpoint = max_id
            self.write_checkpoint()  # 将新的 checkpoint 写入文件

    def do_listen(self):
        """监听数据库变动，并定期查询"""
        while not self.stop_flag:
            logging.info(f"Checking for new changes in table {self.table_name}...")
            rows = self.fetch_changes()
            self.process_changes(rows)
            # if rows:
            #     self.process_changes(rows)
            # else:
            #     self.err_count += 1
            #     logging.error(f"No new changes in table {self.table_name} !")
            #     if self.err_count > self.max_err_count:
            #         self.error_chat(self.error_)

            # 等待一段时间后再进行下次查询
            time.sleep(self.query_interval)

    def error_chat(self, err=''):
        """报警程序"""
        chat_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=f5b5627b-5ceb-4796-87d5-4d4ca0f94816'
        header = {
            'Content-Type': "application/json"
        }
        text = f"""
        ### 数据同步可能出现问题
        - **涉及数据：<font color=\"red\">{self.table_name}</font>**
        - **空值次数：{self.err_count}**
        - **报错下限：{self.max_err_count}**
        - **查询间隔：{self.query_interval}**
        - **报错信息：{err}**
        更多详情请查看日志文件。
        """

        body = {
            "msgtype": "markdown",
            "markdown": {
                "content": text
            }
        }
        try:
            requests.post(chat_url, headers=header, data=json.dumps(body))
        except Exception as e:
            logging.error(e)

    def stop_listening(self):
        """停止监听"""
        self.stop_flag = True
        # self.listen_thread.join()  # 等待线程结束
        if self.listen_thread:
            self.listen_thread.join()  # 等待线程结束
        else:
            logging.error(f"No listener thread running for {self.table_name}.")

    def start_listening(self):
        """启动监听线程"""
        if self.listen_thread is None or not self.listen_thread.is_alive():
            self.stop_flag = False
            self.listen_thread = threading.Thread(target=self.do_listen, name=f"Listener_{self.table_name}")
            self.listen_thread.start()
        else:
            logging.error(f"Listener for {self.table_name} is already running.")

    def close_connection(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()

    def stop(self):
        """停止监听"""
        self.stop_flag = True
        self.stop_listening()
        self.close_connection()
