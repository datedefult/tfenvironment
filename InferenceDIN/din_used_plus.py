import os
import time

import tensorflow as tf
from MulConnectionPool import SynDBPools  # 延迟导入，避免循环依赖
from _Recall.Recall_ import Recall

if __name__ == '__main__':
    uid = 111791

    export_path = "../trained_models/saved_din_model/"
    version = "1"  # 后续可以通过版本号进行模型版本迭代与管理

    # 构建完整保存路径
    full_export_path = os.path.join(export_path, version)

    # 默认：4
    gender_map = {'男':1,'女':2,'LGBT':3,'保密':4}
    # 默认：1
    language_map = {'en':1,'de':2,'ja':3}

    # -------------------------------连接池-------------------------------
    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_redis()
    connection_pools.init_mysql()
    connection_pools.init_mongo()
    # -------------------------------用户基础信息-------------------------------
    app_client = connection_pools.get_mongo('app', 'app')
    user_collection = app_client['app_users']
    post_collection = app_client['app_posts']

    tar_user_info = user_collection.find_one({'uid': uid}, {"_id": 0, "basic_information.gender": 1,"basic_information.language": 1})
    user_gender = gender_map.get(tar_user_info.get('basic_information',{}).get('gender','保密'),4)
    user_language = language_map.get(tar_user_info.get('basic_information',{}).get('language','en'),1)
    print(user_gender,user_language)

    # -------------------------------用户点击历史-------------------------------
    click_history_client = connection_pools.get_redis('post_click_history')

    # 前40min
    time_now = int(time.time())
    start_score = int(time.time()*1000)-2400*1000
    post_click_history = list(map(int, click_history_client.zrevrangebyscore(uid, '+inf', f'({start_score}', start=0, num=30)))  # 转换为整数

    print('近期点击历史:',post_click_history)
    post_click_history = list(dict.fromkeys(post_click_history))  # 去重，保持顺序
    post_click_history += [0] * (30 - len(post_click_history))  # 如果列表长度不足30，用0填充
    len_post_click_history = sum(1 for item in post_click_history if item != 0)

    print(len_post_click_history,post_click_history)

    # -------------------------------候选列表：即召回列表-------------------------------
    recall_for_uid = Recall(uid=uid,connection_pools=connection_pools)
    recall_result =  recall_for_uid.run(pushed_tag=True,timeout=3)
    print("召回的数量:",len(recall_result))

    # candidate_post_id_list = recall_result
    # n = len(candidate_post_id_list)
    # candidate_post_type_list = [1]*n

    query = {
        "$or": [
            # 非共玩帖条件
            {
                "post_id": {"$in": recall_result},
                "post_info.deleted_at": 0,  # deleted_at 等于 0 未删除
                "post_info.visible": 1,  # visible 等于 1 所有人可见
                "post_info.show_type": 0,  # show_type 等于 0 正常类型的帖子
                "post_info.status": 30,  # status 等于 30 审核通过的帖子
                "post_info.type": {"$ne": 4},  # post.type 不等于 4
                "post_info.released_at": {"$lt": time_now}  # released_at 小于当前时间戳
            },
            # 共玩帖条件
            {
                "post_id": {"$in": recall_result},
                "post_info.deleted_at": 0,  # deleted_at 等于 0 未删除
                "post_info.visible": 1,  # visible 等于 1 所有人可见
                "post_info.show_type": 0,  # show_type 等于 0 正常类型的帖子
                "post_info.status": 30,  # status 等于 30 审核通过的帖子
                "post_info.type": 4,  # post.type 等于 4
                "post_info.released_at": {"$lt": time_now},  # released_at 小于当前时间戳
                "post_info.created_at": {"$gt": time_now - 24 * 60 * 60},  # created_at 大于当前时间戳 - 24小时
                "post_score": {"$gt": 20}  # post_score 大于 20
            }
        ]
    }

    post_info_dict = post_collection.find(query,{"_id": 0,"post_info.id":1,"post_info.type":1})
    candidate_post_id_list = []
    candidate_post_type_list=[]
    for row in post_info_dict:
        candidate_post_id_list.append(row.get('post_info',{}).get('id',1))
        candidate_post_type_list.append(row.get('post_info',{}).get('type',1))

    n = len(candidate_post_id_list)
    print(candidate_post_type_list)
    test_input = {
        'uid': tf.constant([uid] * n, dtype=tf.int64),
        'gender': tf.constant([user_gender] * n, dtype=tf.int64),
        'language': tf.constant([user_language] * n, dtype=tf.int64),
        'hit_post_id_list': tf.constant([post_click_history] * n,dtype=tf.int64),
        'hit_len': tf.constant([len_post_click_history] * n, dtype=tf.int64),
        'post_id': tf.constant(candidate_post_id_list, dtype=tf.int64),
        'type': tf.constant(candidate_post_type_list, dtype=tf.int64)
    }

    loaded_model = tf.saved_model.load(full_export_path)
    serving_fn = loaded_model.signatures["serving_default"]

    # 预测输入
    predictions = serving_fn(
        uid=test_input['uid'],
        gender=test_input['gender'],
        language=test_input['language'],
        hit_post_id_list=test_input['hit_post_id_list'],
        hit_len=test_input['hit_len'],
        post_id=test_input['post_id'],
        type=test_input['type']
    )
    predictions_np = predictions['prediction'].numpy()

    # 将预测分数与候选ID配对
    scored_items = list(zip(candidate_post_id_list, predictions_np))

    # 按预测分数降序排序
    sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)

    # 解包结果
    sorted_ids = [item[0] for item in sorted_items]
    sorted_scores = [item[1] for item in sorted_items]

    print("精排序后的数量:",len(sorted_ids))
    print("精排序后的候选ID:", sorted_ids)
    print(sorted_scores)