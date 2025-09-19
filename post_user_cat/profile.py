from collections import defaultdict
import numpy as np
import re
from datetime import datetime

import pymysql

from MulConnectionPool import SynDBPools
from collections import defaultdict
import json
from datetime import datetime


def get_content_tags_map(mysql_conn):
    tags_map = defaultdict(str)
    query = 'SELECT id,sub_tag_name FROM `post_tag_dictionary`'
    with mysql_conn.cursor(cursor=pymysql.cursors.SSDictCursor) as cursor:
        cursor.execute(query)
        for row in cursor:
            tags_map[row['id']] = row['sub_tag_name']
    return tags_map




def generate_user_profiles(docs):
    """
    从帖子数据生成用户画像（改进版）
    改进点：
    1. 更严格的数据类型检查
    2. 分离权重计算逻辑
    3. 添加详细的错误处理
    """
    user_data = defaultdict(lambda: {
        "tag_weights": defaultdict(float),
        "total_weight": 0
    })
    
    BEHAVIOR_WEIGHTS = {
        "opened": 1,
        "liked": 2,
        "hoard": 3
    }
    
    def safe_user_id(user_id):
        """安全转换用户ID"""
        try:
            return int(user_id)
        except (ValueError, TypeError):
            return None
    
    for doc in docs:
        content_tags = doc.get('content_tags', [])
        if not isinstance(content_tags, list):
            continue
        
        used = doc.get('used', {})
        
        # 处理打开行为（带次数统计）
        opened_users = used.get('opened_user_list', {})
        for user_id_str, timestamps in opened_users.items():
            user_id = safe_user_id(user_id_str)
            if user_id is None or not isinstance(timestamps, list):
                continue
            
            weight = BEHAVIOR_WEIGHTS['opened'] * len(timestamps)
            user_data[user_id]['total_weight'] += weight
            for tag in content_tags:
                if isinstance(tag, (str, int)):  # 确保标签有效
                    user_data[user_id]['tag_weights'][str(tag)] += weight
        
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
                user_data[user_id]['total_weight'] += weight
                for tag in content_tags:
                    if isinstance(tag, (str, int)):
                        user_data[user_id]['tag_weights'][str(tag)] += weight
    
    # 生成最终用户画像
    user_profiles = {}
    for user_id, data in user_data.items():
        total_weight = data['total_weight']
        if total_weight > 0:
            # 归一化标签偏好并保留2位小数
            tag_preferences = {
                tag: round(weight / total_weight, 2)
                for tag, weight in data['tag_weights'].items()
                if weight > 0  # 过滤零权重
            }
            
            user_profiles[user_id] = {
                "tag_preferences": tag_preferences,
                "interaction_count": round(total_weight, 2)
            }
    
    return user_profiles


def build_post_user_profile(doc, user_profiles):
    """
    构建帖子用户画像（改进版）
    改进点：
    1. 移除无意义的avg_interaction_count
    2. 添加帖子本身的互动统计数据
    3. 更安全的用户ID处理
    """
    used = doc.get('used', {})
    
    # 获取正向用户（去重）
    positive_users = set()
    
    # 帖子互动统计（新增）
    post_stats = {
        'total_opens': 0,
        'total_likes': 0,
        'total_hoards': 0,
        'unique_users': 0
    }
    
    # 处理打开行为
    opened_users = used.get('opened_user_list', {})
    for user_id_str, timestamps in opened_users.items():
        try:
            user_id = int(user_id_str)
            positive_users.add(user_id)
            post_stats['total_opens'] += len(timestamps)
        except (ValueError, TypeError):
            continue
    
    # 处理点赞行为
    liked_users = used.get('liked', [])
    post_stats['total_likes'] = len(liked_users)
    for user_id in liked_users:
        try:
            positive_users.add(int(user_id))
        except (ValueError, TypeError):
            continue
    
    # 处理收藏行为
    hoard_users = used.get('hoard', [])
    post_stats['total_hoards'] = len(hoard_users)
    for user_id in hoard_users:
        try:
            positive_users.add(int(user_id))
        except (ValueError, TypeError):
            continue
    
    post_stats['unique_users'] = len(positive_users)
    
    if not positive_users:
        return None
    
    # 初始化画像数据
    post_profile = {
        'post_id': doc.get('post_id'),
        'content_tags': doc.get('content_tags', []),
        'interaction_stats': post_stats,  # 新增帖子互动统计
        'user_preferences': defaultdict(float)
    }
    
    # 计算用户偏好
    total_score = 0
    for user_id in positive_users:
        if user_id in user_profiles:
            user_profile = user_profiles[user_id]
            for tag, score in user_profile.get('tag_preferences', {}).items():
                post_profile['user_preferences'][tag] += score
                total_score += score
    
    # 归一化用户偏好
    if total_score > 0:
        post_profile['user_preferences'] = {
            tag: round(score / total_score, 2)
            for tag, score in post_profile['user_preferences'].items()
        }
    
    return post_profile


def predict_users_for_post(post_content_tags, user_profiles, top_n=3):
    """
    预测可能喜欢新帖子的用户（改进版）
    改进点：
    1. 更合理的权重计算
    2. 添加最低分数阈值
    """
    MIN_SCORE_THRESHOLD = 0.1  # 最低匹配分数阈值
    
    if not isinstance(post_content_tags, list):
        return []
    
    post_tags = set(str(tag) for tag in post_content_tags)
    user_scores = []
    
    for user_id, user_profile in user_profiles.items():
        # 计算基础标签匹配度
        base_score = sum(
            user_profile.get('tag_preferences', {}).get(tag, 0)
            for tag in post_tags
        )
        
        if base_score < MIN_SCORE_THRESHOLD:
            continue
        
        # 加入用户活跃度权重（对数平滑）
        activity_weight = 1 + np.log1p(user_profile.get('interaction_count', 0)) * 0.1
        weighted_score = round(base_score * activity_weight, 2)
        
        user_scores.append((user_id, weighted_score))
    
    # 按得分降序排序
    user_scores.sort(key=lambda x: x[1], reverse=True)
    return user_scores[:top_n]

def build_all_post_profiles(docs, user_profiles):
    """
    为所有帖子构建用户画像
    """
    post_profiles = {}
    for doc in docs:
        profile = build_post_user_profile(doc, user_profiles)
        if profile:
            post_profiles[doc['post_id']] = profile
    return post_profiles


def save_profiles_to_json(user_profiles, post_profiles, prefix="profiles"):
    """
    将用户画像和帖子画像保存为JSON文件
    :param user_profiles: 用户画像字典
    :param post_profiles: 帖子画像字典
    :param prefix: 文件名前缀
    """
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存用户画像
    user_filename = f"{prefix}_user_{timestamp}.json"
    with open(user_filename, 'w', encoding='utf-8') as f:
        # 处理可能的numpy类型和特殊对象
        json.dump(user_profiles, f, ensure_ascii=False, indent=2, default=str)
    print(f"用户画像已保存至: {user_filename}")
    
    # 保存帖子画像
    post_filename = f"{prefix}_post_{timestamp}.json"
    with open(post_filename, 'w', encoding='utf-8') as f:
        json.dump(post_profiles, f, ensure_ascii=False, indent=2, default=str)
    print(f"帖子画像已保存至: {post_filename}")
    
    return user_filename, post_filename


if __name__ == '__main__':
    
    # ==================== 使用示例 ====================
    # 假设你的docs数据是这样的格式

    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_mysql()
    connection_pools.init_mongo()
    bidata_conn = connection_pools.get_mysql('bidata')
    app_client = connection_pools.get_mongo('app', 'app')
    post_collection = app_client['app_posts']
    post_act_options = {
        "_id": 0,
        "post_id": 1,
        "used.opened_user_list": 1,
        "used.liked": 1,
        "used.hoard": 1,
        "content_tags": 1,
    }
    docs = post_collection.find({}, post_act_options)
    # 持久保存游标数据
    docs = list(docs)
    # 替换tag
    tags_map = get_content_tags_map(bidata_conn)
    for doc in docs:
        content_tags = doc.get('content_tags',[])
        if content_tags:
            doc['content_tags'] = [tags_map.get(tag_id) for tag_id in content_tags]
    # 生成用户画像
    print("正在生成用户画像...")
    user_profiles = generate_user_profiles(docs)
    print(f"共生成 {len(user_profiles)} 个用户画像")
    
    # 生成帖子画像
    print("\n正在生成帖子画像...")
    post_profiles = build_all_post_profiles(docs, user_profiles)
    print(f"共生成 {len(post_profiles)} 个帖子画像")
    
    # 打印部分用户画像示例
    print("\n============ 用户画像示例 ============")
    sample_count = 0
    for user_id, profile in list(user_profiles.items())[:5]:
        print(f"用户 {user_id}:")
        print(f"  互动权重: {profile['interaction_count']}")
        print(f"  标签偏好: {profile['tag_preferences']}")
        sample_count += 1
        if sample_count >= 5:
            break
    
    # 打印帖子画像
    print("\n============ 帖子画像 ============")
    for post_id, profile in list(post_profiles.items())[40000:3]:
        print(profile)
        print('--------------------------------')
        print(f"Post {post_id} Profile:")
        print(f"Tags Weight: {profile['user_preferences']}")
    
    save_profiles_to_json(user_profiles, post_profiles)
    # # 新帖子预测示例
    # print("\n============ 新帖子预测示例 ============")
    # new_post_tags = ["科技", "编程"]  # 根据实际情况调整
    # if user_profiles:  # 确保有用户画像数据
    #     recommended_users = predict_users_for_post(new_post_tags, user_profiles, top_n=5)
    #     print(f"\n对标签 {new_post_tags} 的推荐用户：")
    #     for user_id, score in recommended_users:
    #         print(f"  用户 {user_id} (匹配度: {score})")
    # else:
    #     print("暂无用户画像数据，无法进行预测")