import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pymysql

# from pymongo import MongoClient
from MulConnectionPool import SynDBPools


class ContentTagAnalyzer:
    def __init__(self):
        # 初始化数据库连接
        self.connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
        self.connection_pools.init_mysql()
        self.connection_pools.init_mongo()
    
    def get_content_tags_map(self):
        """从MySQL获取标签映射表"""
        tags_map = defaultdict(str)
        bidata_conn = self.connection_pools.get_mysql('bidata')
        query = 'SELECT id, sub_tag_name FROM `post_tag_dictionary`'
        with bidata_conn.cursor(cursor=pymysql.cursors.SSDictCursor) as cursor:
            cursor.execute(query)
            for row in cursor:
                tags_map[row['id']] = row['sub_tag_name']
        return tags_map
    
    def load_post_data(self, limit=None):
        """从MongoDB加载帖子数据"""
        app_client = self.connection_pools.get_mongo('app', 'app')
        post_collection = app_client['app_posts']
        
        post_act_options = {"_id": 0, "post_id": 1, "used.opened_user_list": 1, "used.liked": 1, "used.hoard": 1,
            "used.comment": 1, "used.share": 1, "content_tags": 1, }
        # test:限制帖子长度
        if limit is not None:
            docs = list(post_collection.find({}, post_act_options).limit(limit))
        else:
            docs = list(post_collection.find({}, post_act_options))
        return docs
    
    def preprocess_tags(self, docs, tags_map):
        """预处理标签数据"""
        for doc in docs:
            content_tags = doc.get('content_tags', [])
            if content_tags:
                doc['content_tags'] = [tags_map.get(tag_id, f'unknown_{tag_id}') for tag_id in content_tags]
        return docs
    
    def calculate_tag_weights(self, docs, min_count=5):
        """计算标签权重（基于逆文档频率，亦可以自定义各个标签的权重）"""
        from collections import Counter
        
        # 统计标签出现频率
        tag_counter = Counter()
        for doc in docs:
            tags = doc.get('content_tags', [])
            tag_counter.update(tags)
        
        # 过滤低频标签
        valid_tags = {tag for tag, count in tag_counter.items() if count >= min_count}
        
        # 计算IDF权重（出现越多的标签权重越低）
        total_docs = len(docs)
        tag_weights = {tag: round(np.log(total_docs / count), 2) for tag, count in tag_counter.items() if
            tag in valid_tags and tag not in ['付费索取', '社媒引流', '品牌导流']}
        print("tag 权重数据：\n", tag_weights)
        return tag_weights
    
    def generate_balanced_user_profiles(self, docs, tag_weights):
        """生成考虑标签平衡的用户画像"""
        user_data = defaultdict(lambda: {"weighted_tag_scores": defaultdict(float), "raw_tag_counts": defaultdict(int),
            "total_interactions": 0})
        
        BEHAVIOR_WEIGHTS = {"opened": 1, "liked": 2, "hoard": 3, "comment": 4, "share": 5}
        
        def safe_user_id(user_id):
            """安全转换用户ID"""
            try:
                return int(user_id)
            except (ValueError, TypeError):
                return None
        
        for doc in docs:
            content_tags = [tag for tag in doc.get('content_tags', [])]
            if not content_tags:
                continue
            
            used = doc.get('used', {})
            
            # 处理打开行为
            opened_users = used.get('opened_user_list', {})
            for user_id_str, timestamps in opened_users.items():
                user_id = safe_user_id(user_id_str)
                if user_id is None or not isinstance(timestamps, list):
                    continue
                
                weight = BEHAVIOR_WEIGHTS['opened'] * len(timestamps)
                user_data[user_id]['total_interactions'] += len(timestamps)
                for tag in content_tags:
                    user_data[user_id]['weighted_tag_scores'][tag] += weight * tag_weights.get(tag, 1)
                    user_data[user_id]['raw_tag_counts'][tag] += len(timestamps)
            
            # 评论
            comment_users = used.get('comment', {})
            for user_id_str, timestamps in comment_users.items():
                user_id = safe_user_id(user_id_str)
                if user_id is None or not isinstance(timestamps, list):
                    continue
                
                weight = BEHAVIOR_WEIGHTS['comment'] * len(timestamps)
                user_data[user_id]['total_interactions'] += len(timestamps)
                for tag in content_tags:
                    user_data[user_id]['weighted_tag_scores'][tag] += weight * tag_weights.get(tag, 1)
                    user_data[user_id]['raw_tag_counts'][tag] += len(timestamps)
            
            # 分享
            share_users = used.get('share', {})
            for user_id_str, timestamps in share_users.items():
                user_id = safe_user_id(user_id_str)
                if user_id is None or not isinstance(timestamps, list):
                    continue
                
                weight = BEHAVIOR_WEIGHTS['share'] * len(timestamps)
                user_data[user_id]['total_interactions'] += len(timestamps)
                for tag in content_tags:
                    user_data[user_id]['weighted_tag_scores'][tag] += weight * tag_weights.get(tag, 1)
                    user_data[user_id]['raw_tag_counts'][tag] += len(timestamps)
            
            # 处理点赞和收藏行为
            for behavior in ['liked', 'hoard']:
                user_list = used.get(behavior, [])
                if not isinstance(user_list, list):
                    continue
                
                for user_id in user_list:
                    user_id = safe_user_id(user_id)
                    if user_id is None:
                        continue
                    
                    weight = BEHAVIOR_WEIGHTS[behavior]
                    # 总交互次数
                    user_data[user_id]['total_interactions'] += 1
                    for tag in content_tags:
                        # 各个tag的得分
                        user_data[user_id]['weighted_tag_scores'][tag] += weight * tag_weights.get(tag, 1)
                        # 与带有该tag内容的交互次数
                        user_data[user_id]['raw_tag_counts'][tag] += 1
        
        # 生成最终用户画像
        user_profiles = {}
        for user_id, data in user_data.items():
            total_weight = sum(data['weighted_tag_scores'].values())
            if total_weight > 0:
                user_profiles[user_id] = {"tag_preferences": {tag: round(score / total_weight, 2) for tag, score in
                    data['weighted_tag_scores'].items()}, "raw_tag_counts": dict(data['raw_tag_counts']),
                    "total_interactions": data['total_interactions']}
        
        return user_profiles
    
    def build_balanced_post_profile(self, doc, user_profiles, tag_weights):
        """构建考虑标签平衡的帖子画像"""
        used = doc.get('used', {})
        content_tags = [tag for tag in doc.get('content_tags', []) if tag in tag_weights]
        
        # ---- 核心互动统计 ----
        stats = {'total_opens': sum(len(v) for v in used.get('opened_user_list', {}).values()),
            'total_likes': len(used.get('liked', [])), 'total_hoards': len(used.get('hoard', [])), 'unique_users': len(
                set(list(used.get('opened_user_list', {}).keys()) + used.get('liked', []) + used.get('hoard', [])))}
        
        # ---- 用户偏好聚合（带标签权重） ----
        weighted_prefs = defaultdict(float)
        positive_users = set()
        
        # 处理打开用户
        opened_users = used.get('opened_user_list', {})
        for user_id_str in opened_users.keys():
            try:
                user_id = int(user_id_str)
                positive_users.add(user_id)
            except (ValueError, TypeError):
                continue
        
        # 处理点赞和收藏用户
        for behavior in ['liked', 'hoard']:
            for user_id in used.get(behavior, []):
                try:
                    positive_users.add(int(user_id))
                except (ValueError, TypeError):
                    continue
        
        for user_id in positive_users:
            if user_id in user_profiles:
                for tag, score in user_profiles[user_id]["tag_preferences"].items():
                    weighted_prefs[tag] += score * tag_weights.get(tag, 1)
        
        # 归一化
        total_score = sum(weighted_prefs.values())
        user_preferences = {tag: round(score / total_score, 2) for tag, score in
            weighted_prefs.items()} if total_score > 0 else {}
        
        return {'post_id': doc.get('post_id'), 'content_tags': content_tags, 'interaction_stats': stats,
            'user_preferences': user_preferences, 'tag_weights_applied': True}
    
    def predict_users_for_post(self, post_content_tags, user_profiles, top_n=3):
        """预测可能喜欢新帖子的用户（带标签平衡）"""
        MIN_SCORE_THRESHOLD = 0.1
        
        if not isinstance(post_content_tags, list):
            return []
        
        post_tags = set(str(tag) for tag in post_content_tags)
        user_scores = []
        
        for user_id, user_profile in user_profiles.items():
            # 计算基础标签匹配度
            base_score = sum(user_profile.get('tag_preferences', {}).get(tag, 0) for tag in post_tags)
            
            if base_score < MIN_SCORE_THRESHOLD:
                continue
            
            # 加入用户活跃度权重（对数平滑）
            activity_weight = 1 + np.log1p(user_profile.get('total_interactions', 0)) * 0.1
            weighted_score = round(base_score * activity_weight, 2)
            
            user_scores.append((user_id, weighted_score))
        
        # 按得分降序排序
        user_scores.sort(key=lambda x: x[1], reverse=True)
        return user_scores[:top_n]
    
    def save_profiles_to_json(self, user_profiles, post_profiles, prefix="profiles"):
        """保存画像数据为JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存用户画像
        user_filename = f"{prefix}_user_{timestamp}.json"
        with open(user_filename, 'w', encoding='utf-8') as f:
            json.dump(user_profiles, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存帖子画像
        post_filename = f"{prefix}_post_{timestamp}.json"
        with open(post_filename, 'w', encoding='utf-8') as f:
            json.dump(post_profiles, f, ensure_ascii=False, indent=2, default=str)
        
        return user_filename, post_filename
    
    def save_profiles_to_sql(self,user_profiles, post_profiles):
        """
        todo: 持久化存储逻辑
        """
    
    def run_full_analysis(self, sample_limit=None):
        """完整分析流程"""
        print("=== 开始分析流程 ===")
        
        # 步骤1: 加载数据
        print("\n[1/5] 加载数据...")
        tags_map = self.get_content_tags_map()
        docs = self.load_post_data(limit=sample_limit)
        docs = self.preprocess_tags(docs, tags_map)
        print(f"已加载 {len(docs)} 条帖子数据")
        
        # 步骤2: 计算标签权重
        print("\n[2/5] 计算标签权重...")
        tag_weights = self.calculate_tag_weights(docs)
        print(f"计算 {len(tag_weights)} 个标签的权重 (最大权重: {max(tag_weights.values()):.2f})")
        
        # 步骤3: 生成用户画像
        print("\n[3/5] 生成用户画像...")
        user_profiles = self.generate_balanced_user_profiles(docs, tag_weights)
        print(f"生成 {len(user_profiles)} 个用户画像")
        
        # 步骤4: 生成帖子画像
        print("\n[4/5] 生成帖子画像...")
        post_profiles = {}
        for doc in docs:
            profile = self.build_balanced_post_profile(doc, user_profiles, tag_weights)
            if profile:
                post_profiles[doc['post_id']] = profile
        print(f"生成 {len(post_profiles)} 个帖子画像")
        
        # 步骤5: 保存结果
        print("\n[5/5] 保存结果...")
        user_file, post_file = self.save_profiles_to_json(user_profiles, post_profiles)
        
        print("\n=== 分析完成 ===")
        print(f"用户画像保存至: {user_file}")
        print(f"帖子画像保存至: {post_file}")
        
        return user_profiles, post_profiles, tag_weights


if __name__ == '__main__':
    analyzer = ContentTagAnalyzer()
    
    # 运行完整分析（可设置sample_limit限制数据量）
    # user_profiles, post_profiles, tag_weights = analyzer.run_full_analysis(sample_limit=50)
    user_profiles, post_profiles, tag_weights = analyzer.run_full_analysis()
