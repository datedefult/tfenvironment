import numpy as np
import redis
from collections import defaultdict
import multiprocessing

from Config.redisConfig import REDIS_USER_COMMUNITY_CLICK
from utils.LogsColor import logging

from MulConnectionPool import SynDBPools

class ItemCFBatchWithPenalty:
    def __init__(self, user_item_dict):
        # 将每个用户的物品集合
        self.user_item_dict = {key: set(user_item_dict[key]) for key in user_item_dict.keys()}
        self.users_list = list(self.user_item_dict.keys())

        # 物品的用户集合
        self.item_user_dict = defaultdict(set)
        # 建立物品到用户的倒查表
        for user, items in self.user_item_dict.items():
            for item in items:
                self.item_user_dict[item].add(user)

        # 物品列表
        self.items_list = list(self.item_user_dict.keys())

        # 使用最大商品编号 + 1 来计算物品数量
        self.num_items = len(self.items_list)  # 修改为最大商品编号 + 1

        self.item_popularity = defaultdict(int)

        # 建立物品到索引的映射
        self.item_to_index = {item: idx for idx, item in enumerate(self.items_list)}
        self.index_to_item = {idx: item for idx, item in enumerate(self.items_list)}  # 反向映射

    def compute_similarity_matrix(self):
        """计算物品之间的相似度矩阵"""
        self.item_matrix = np.zeros((self.num_items, self.num_items))  # 创建稀疏矩阵

        # 计算物品之间的相似度矩阵
        for user, items in self.user_item_dict.items():
            items = list(items)
            for i, item1 in enumerate(items):
                for j in range(i + 1, len(items)):  # 只计算一半，避免重复计算
                    item2 = items[j]
                    idx1 = self.item_to_index[item1]  # 使用映射关系获取稀疏矩阵索引
                    idx2 = self.item_to_index[item2]
                    self.item_matrix[idx1, idx2] += 1
                    self.item_matrix[idx2, idx1] += 1  # 对称位置也加上

        # 物品相似度归一化
        for i in range(self.num_items):
            for j in range(self.num_items):
                if i != j and self.item_matrix[i, j] > 0:
                    self.item_matrix[i, j] /= np.sqrt(
                        len(self.item_user_dict[self.items_list[i]]) * len(self.item_user_dict[self.items_list[j]]))

        # 计算物品的流行度
        for item in self.items_list:
            self.item_popularity[item] = len(self.item_user_dict[item])

    def recommend_for_user(self, user, top_k=5):
        """为单个用户计算推荐商品"""
        user_items = self.user_item_dict[user]
        scores = np.zeros(self.num_items)

        # 计算推荐分数
        for item in user_items:
            item_idx = self.item_to_index[item]
            scores += self.item_matrix[item_idx]  # 从矩阵中提取相似度

        # 加入热门商品惩罚
        for i, item in enumerate(self.items_list):
            popularity_penalty = np.log(1 + self.item_popularity[item])
            if popularity_penalty == 0:
                popularity_penalty = 1e-6  # 设定一个最小的值，避免除零错误
            scores[i] /= popularity_penalty

        # 生成推荐列表，使用 index_to_item 将索引映射回物品
        recommended_items = [self.index_to_item[i]  # 使用映射表将索引转回物品
                                for i in np.argsort(scores)[::-1]
                                if scores[i] > 0 and self.index_to_item[i] not in user_items
                            ][:top_k]
        return recommended_items

    # 批量计算所有用户的推荐商品（使用多进程）
    def batch_recommend(self, top_k=5):
        """为所有用户计算推荐商品"""
        recommendations = {}
        # 计算每个用户的推荐商品
        for user in self.users_list:
            recommendations[user] = self.recommend_for_user(user, top_k)

        return recommendations

    # 返回物品相似度字典
    def get_item_similarity_dict(self):
        """
        返回物品相似度字典，格式为 {item1: {item2: score, item3: score}, ...}
        仅包含相似度 > 0 的物品对
        """
        similarity_dict = {}

        for i in range(self.num_items):
            item1 = self.index_to_item[i]
            similarity_dict[item1] = {}

            for j in range(self.num_items):
                if i == j:
                    continue  # 跳过自身

                item2 = self.index_to_item[j]
                similarity = self.item_matrix[i, j]

                if similarity > 0:  # 只保留有相似度的物品对
                    similarity_dict[item1][item2] = similarity

        return similarity_dict
if __name__ == '__main__':
    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_redis()
    redis_post_click_history = connection_pools.get_redis('post_click_history')
    user_sid_dict = {int(key): list(map(int, redis_post_click_history.lrange(key, 0, -1))) for key in
                     redis_post_click_history.keys('*')}


    itemcf = ItemCFBatchWithPenalty(user_sid_dict)
    itemcf.compute_similarity_matrix()

    # 直接调用带多进程支持的推荐方法
    itemcf_recommendations = itemcf.batch_recommend(top_k=200)

    redis_item_cf = connection_pools.get_redis('item_cf')
    redis_item_sim = connection_pools.get_redis('item_item')
    with redis_item_cf.pipeline() as pipe:
        for key, value_list in itemcf_recommendations.items():
            pipe.delete(key)
            if value_list:  # 确保列表不为空
                pipe.rpush(key, *value_list)  # 注意这里的 * 解包操作

        # 一次性执行所有命令
        pipe.execute()
        logging.info("ItemCF complete")

    with redis_item_sim.pipeline() as pipe:
        for itemi, value_dict in itemcf.get_item_similarity_dict().items():
            pipe.delete(itemi)
            for itemj,score in value_dict.items():
                if itemj != itemi:
                    # print(itemi, {itemj: round(score,4)})
                    pipe.zadd(itemi, {itemj: float(score)})
        # 一次性执行所有命令
        pipe.execute()
        logging.info("Item-Item matrix complete")

    connection_pools.close()