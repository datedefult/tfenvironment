import os

import tensorflow as tf

if __name__ == '__main__':


    export_path = "../trained_models/saved_din_model/"
    version = "1"  # 后续可以通过版本号进行模型版本迭代与管理

    # 构建完整保存路径
    full_export_path = os.path.join(export_path, version)

    # 默认：4
    gender_map = {'男':1,'女':2,'LGBT':3,'保密':4}
    # 默认：1
    language_map = {'en':1,'de':2,'ja':3}

    # 用户信息待查询：mongodb
    # 用户点击列表：redis 12 找出不长于30 ，并去重
    # 候选列表：即召回列表

    candidate_post_id_list = [3179, 3186, 3194]
    candidate_post_type_list = [3,2,2]
    n = len(candidate_post_id_list)

    test_input = {
        'uid': tf.constant([111791] * n, dtype=tf.int64),
        'gender': tf.constant([2] * n, dtype=tf.int64),
        'language': tf.constant([1] * n, dtype=tf.int64),
        'hit_post_id_list': tf.constant(
            [[2832, 2844, 2835, 2895, 2935, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * n,  # 创建n份拷贝（形状自动变为[n, 30]）
            dtype=tf.int64
        ),
        'hit_len': tf.constant([5] * n, dtype=tf.int64),  # 实际历史长度=5
        'post_id': tf.constant(candidate_post_id_list, dtype=tf.int64),
        'type': tf.constant(candidate_post_type_list, dtype=tf.int64)
    }

    loaded_model = tf.saved_model.load(full_export_path)
    serving_fn = loaded_model.signatures["serving_default"]


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

    print("精排序后的候选ID:", sorted_ids)
    print(sorted_scores)