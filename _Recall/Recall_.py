import os
import random
import sys
import time
from collections import defaultdict

from utils.LogsColor import logging
from utils.Tools import other2int
import concurrent.futures
from typing import Dict, List, Any
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Recall:

    def __init__(self, uid: int, connection_pools=None):
        """
        初始化Recall类，设置用户ID和数据库连接池。

        :param uid: 用户ID
        """
        self.uid = uid
        self.connection_pools = connection_pools
        self.recall_results = None
        # 如果未提供connection_pools，则初始化默认连接池
        if self.connection_pools is None:
            self._init_default_connection_pools()
    def _init_default_connection_pools(self):
        """初始化默认的数据库连接池"""
        from MulConnectionPool import SynDBPools  # 延迟导入，避免循环依赖
        print("类中初始化")
        self.connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
        self.connection_pools.init_redis()
        self.connection_pools.init_mysql()
        self.connection_pools.init_mongo()

    def return_recent_post(self, uids: List[int], n: int = 3) -> List[int]:
        """
        查找多个用户近期发帖情况。

        :param uids: 用户ID列表
        :param n: 每个用户返回的帖子数量
        :return: 用户近期发帖的ID列表
        """
        user_recall_list = []
        auther_latest_release = self.connection_pools.get_redis('author_latest_release')
        for author_uid in uids:
            latest_release = auther_latest_release.lrange(author_uid, 0, n - 1)
            author_post = [other2int(x) for x in latest_release]
            user_recall_list += author_post
        return user_recall_list

    def recent_action_author(self) -> List[int]:
        """
        召回近期与目标用户有交互的作者的最新帖子。

        :return: 近期交互作者的帖子ID列表
        """
        redis_conn = self.connection_pools.get_redis('recent_user_author')
        zset_data = redis_conn.zrange(self.uid, 0, -1, withscores=True)
        # 近期交互过的作者
        author_ = [other2int(author[0]) for author in zset_data]
        # 召回这些作者近期发布的帖子
        recent_author_recall = self.return_recent_post(author_, 8)
        return recent_author_recall

    def top_follow_author(self) -> List[int]:
        """
        召回粉丝数最多或者增长速度较快的作者的最新帖子。

        :return: 作者的帖子ID列表
        """
        try:
            app_client = self.connection_pools.get_mongo('app', 'app')
            user_collection = app_client['app_users']
            user_info = user_collection.find(
                {
                    "$or": [
                        {"fans.increase_speed": {"$gt": 10}},
                        {"fans.total_follower": {"$gt": 200}}
                    ]
                },
                {"_id": 0, "uid": 1}
            )
            speed_author = [user['uid'] for user in user_info]

            # 召回作者最新发布的5条帖子
            follow_recall_list = self.return_recent_post(speed_author, 3)
            return follow_recall_list
        except Exception as e:
            logging.error(e)
            return []
    def follow_author(self) -> List[int]:
        """
        召回目标用户关注的作者的最新帖子。

        :return: 关注作者的帖子ID列表
        """
        try:
            app_client = self.connection_pools.get_mongo('app', 'app')
            user_collection = app_client['app_users']
            user_info = user_collection.find_one({'uid': self.uid}, {"_id": 0, "app_play_list.follow_list": 1})
            user_info = user_info or {}
            follow_list = user_info.get('app_play_list', {}).get('follow_list', [])
            # 该用户关注的作者
            follow_list = [other2int(x) for x in follow_list]
            # 召回该作者最新发布的5条帖子
            follow_recall_list = self.return_recent_post(follow_list, 5)
            return follow_recall_list
        except Exception as e:
            logging.error(e)
            return []

    def sim_category(self) -> List[int]:
        """
        召回与目标用户具有相同属性的用户的帖子（冷启动）。

        :return: 相似用户的帖子ID列表
        """
        try:
            app_client = self.connection_pools.get_mongo('app', 'app')
            user_collection = app_client['app_users']
            # 目标用户及其属性
            tar_user_info = user_collection.find_one({'uid': self.uid}, {"_id": 0, "basic_information": 1})
            tar_user_info = tar_user_info or {}
            tar_gender = tar_user_info.get('basic_information', {}).get('gender', '保密')
            tar_language = tar_user_info.get('basic_information', {}).get('language', 'en')
            tar_channels = tar_user_info.get('basic_information', {}).get('channels', 'google')
            tar_continent = tar_user_info.get('basic_information', {}).get('continent', 1)

            sim_uid = user_collection.find({'basic_information.gender': tar_gender,
                                            'basic_information.language': tar_language,
                                            'basic_information.channels': tar_channels,
                                            'basic_information.continent': tar_continent
                                            }, {"_id": 0, "uid": 1})
            # 具有相同属性的用户
            sim_uid_list = sim_uid.distinct('uid')

            post_click_history = self.connection_pools.get_redis('post_click_history')
            click_user_list = post_click_history.keys('*')
            click_user_list = [other2int(uid) for uid in click_user_list]
            cover_list = list(set(click_user_list) & set(sim_uid_list))

            # 最多随机筛选出15个相似的用户
            n = min(len(cover_list), 15)
            random_elements = random.sample(cover_list, n)
            sim_user_recall_list = []
            for sim_uid in random_elements:
                sim_post = post_click_history.zrevrange(sim_uid, 0, 15)
                sim_post = [other2int(x) for x in sim_post]
                sim_user_recall_list += sim_post
            return sim_user_recall_list
        except Exception as e:
            logging.error(e)
            return []
    def itemcf(self) -> List[int]:
        """
        使用ItemCF算法召回与目标用户相关的帖子。

        :return: ItemCF召回的帖子ID列表
        """
        try:
            itemcf_list = self.connection_pools.get_redis('item_cf').lrange(self.uid, 0, -1)
            itemcf_list = [other2int(x) for x in itemcf_list]

            if not itemcf_list:
                # 若没有直接的结果，则从历史数据来进行再次召回

                post_click_history = self.connection_pools.get_redis('post_click_history')
                redis_item_sim = self.connection_pools.get_redis('item_item')
                item_total_scores = defaultdict(float)
                click_posts = post_click_history.zrevrange(self.uid, 0, 20)

                for key in click_posts:
                    sim_posts = redis_item_sim.zrange(key, 0, -1, withscores=True)
                    for item_id, score in sim_posts:
                        item_total_scores[item_id] += score  # 累加分数

                sorted_items = sorted(
                    item_total_scores.items(),
                    key=lambda x: -x[1]  # 按分数降序排序
                )[:200]  # 取前50个

                itemcf_list = [other2int(x[0]) for x in sorted_items]

            return itemcf_list
        except Exception as e:
            logging.error(e)
            return []
    def usercf(self) -> None:
        """
        使用UserCF算法召回与目标用户相关的帖子（未实现）。

        :return: None
        """
        pass

    def emd_sim_user_post(self) -> None:
        """
        根据操作历史获取嵌入，并进行嵌入返回（未实现）。

        :return: None
        """
        app_client = self.connection_pools.get_mongo('app', 'app')
        user_collection = app_client['app_users']
        # 目标用户及其属性
        tar_user_info = user_collection.find_one({'uid': self.uid}, {"_id": 0, "app_play_list.post_recommend_pushed": 1})

    def user_recommend_pushed(self) -> List[int]:
        """
        获取目标用户已推送展示的帖子ID。

        :return: 已推送展示的帖子ID列表
        """
        try:
            app_client = self.connection_pools.get_mongo('app', 'app')
            user_collection = app_client['app_users']
            # 目标用户及其属性
            tar_user_info = user_collection.find_one({'uid': self.uid}, {"_id": 0, "app_play_list.post_recommend_pushed": 1})
            tar_user_info = tar_user_info or {}
            recommend_pushed = tar_user_info.get('app_play_list', {}).get('post_recommend_pushed', [])
            recommend_pushed = [other2int(x) for x in recommend_pushed]
            return recommend_pushed
        except Exception as e:
            logging.error(e)
            return []

    def long_time_no_push(self):
        """
        召回长时间低展现的内容，但是也可能是低质量的帖子
        :return:
        """

    def popular_post_recall(self):
        """
        召回热度较高的贴子
        :return:
        """
        try:
            app_client = self.connection_pools.get_mongo('app', 'app')
            post_collection = app_client['app_posts']
            popular_post_doc = post_collection.find({"post_score": {"$gt": 25}},{"_id": 0, "post_id": 1})
            popular_post_doc = popular_post_doc or {}
            popular_post_list = [other2int(row['post_id']) for row in popular_post_doc]
            return popular_post_list
        except Exception as e:
            logging.error(e)
            return []

    def recent_create_post_recall(self):
        """
        召回最近发布的贴子
        :return:
        """
        try:
            now_timestamp = int(time.time()) - 4*24*3600

            app_client = self.connection_pools.get_mongo('app', 'app')
            post_collection = app_client['app_posts']
            recent_post_doc = post_collection.find(
                {
                    "$or": [
                        {"post_info.created_at": {"$gt": now_timestamp}},
                        {"post_info.updated_at": {"$gt": now_timestamp}}
                    ]
                },
            {"_id": 0, "post_id": 1})
            recent_post_doc = recent_post_doc or {}
            recent_post_list = [other2int(row['post_id']) for row in recent_post_doc]
            return recent_post_list
        except Exception as e:
            logging.error(e)
            return []


    def run_task_with_timeout(self, task_func: callable, timeout: int = 1, default: Any = None) -> Any:
        """
        多线程执行单个任务，带超时控制。

        :param task_func: 要执行的任务函数
        :param timeout: 超时时间（秒）
        :param default: 超时或异常时返回的默认值
        :return: 任务结果或默认值
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task_func)
            try:
                return future.result(timeout=timeout)
            except (concurrent.futures.TimeoutError, Exception) as e:
                logging.info(f"Task {task_func.__name__} timed out or failed: {e}")
                future.cancel()  # 取消任务
                return default if default is not None else []

    def run(self, pushed_tag: bool = False, timeout: int = 1) -> List[Any]:
        """
        多线程启动召回任务，支持超时控制和排除已推送的帖子。

        :param pushed_tag: 是否排除已通过推荐展示的ID，True排除，False不排除
        :param timeout: 每个任务的超时时间（秒）
        :return: 最终召回的帖子ID列表
        """
        tasks = {
            "sim_category_recall_list": self.sim_category,
            "itemcf_recall_list": self.itemcf,
            "follow_recall_list": self.follow_author,
            "recent_act_recall_list": self.recent_action_author,
            "top_follow_recall_list": self.top_follow_author,
            "popular_post_recall":self.popular_post_recall,
            "recent_create_post_recall": self.recent_create_post_recall,
        }

        recall_results = {}
        futures = {}


        # 提交所有任务到线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            for name, task in tasks.items():
                futures[name] = executor.submit(task)  # 提交任务并记录名称

            # 检查每个任务是否完成
            for name, future in futures.items():
                try:
                    recall_results[name] = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError as e:
                    logging.warning(f"任务 {name} 超时: {e}")
                    future.cancel()  # 取消超时的任务
                    recall_results[name] = []  # 超时返回空列表
                except Exception as e:
                    logging.warning(f"任务 {name} 失败: {e}")
                    recall_results[name] = []  # 失败返回空列表

        self.recall_results = recall_results

        # 排除已推送任务
        pushed_list = self.user_recommend_pushed() if pushed_tag else []

        finish_recall = []
        for name, recall_result in recall_results.items():
            finish_recall += recall_result
        finish_recall = list(set(finish_recall) - set(pushed_list))
        return finish_recall

    def close(self):
        self.connection_pools.close()

if __name__ == '__main__':
    """
    主函数，用于测试Recall类的功能。
    """
    recall_for_uid = Recall(111791)
    # recall_for_uid.top_follow_author()
    recall_result =  recall_for_uid.run(pushed_tag=True,timeout=10)

    print("Recall length:",len(recall_result))
    for k, v in recall_for_uid.recall_results.items():
        print(k, v)
    recall_for_uid.close()