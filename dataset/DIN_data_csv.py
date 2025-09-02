from collections import defaultdict, deque

import pandas as pd
import pymysql

from MulConnectionPool import SynDBPools


if __name__ == '__main__':
    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_mysql()

    recommend_conn = connection_pools.get_mysql('content_recommend')
    community_conn = connection_pools.get_mysql('app_community')
    base_conn = connection_pools.get_mysql('base_data')
    # 行为数据
    with recommend_conn.cursor() as cursor:
        cursor.execute('select * from `tensorflow_dataset_din`')
        result_df = pd.DataFrame(cursor.fetchall(),columns=['id','sn','uid','hit_post_id_list','hit_len','post_id','label','timestamp'])

        post_detail_query="""
SELECT
	post_id,
	push_number,
	hits_number,
	likes,
	collect_number,
	comments_number,
	hits_rate,
	like_rate,
	collect_rate,
	comments_rate,
	average_stay_time 
FROM
	`post_score_details`"""
        cursor.execute(post_detail_query)
        post_detail_df = pd.DataFrame(cursor.fetchall(),columns=['post_id','push_number','hits_number','likes','collect_number','comments_number','hits_rate','like_rate','collect_rate','comments_rate','average_stay_time' ])

        post_score_query ="SELECT post_id,score FROM `post_score`"
        cursor.execute(post_score_query)
        post_score_df = pd.DataFrame(cursor.fetchall(),columns=['post_id','score'])



    # 帖子相关信息
    with community_conn.cursor() as cursor:
        cursor.execute('select `id` post_id,`type` from `posts`')
        post_df = pd.DataFrame(cursor.fetchall(),columns=['post_id','type'])

    # 用户相关信息
    with base_conn.cursor() as cursor:
        cursor.execute('select `用户id` uid,`性别` gender,`APP语言` `language` from `JHAD_USER`')
        user_df = pd.DataFrame(cursor.fetchall(),columns=['uid','gender','language'])

    result_df = result_df.merge(post_df,on='post_id',how='left').merge(post_detail_df,on='post_id',how='left').merge(post_score_df,on='post_id',how='left').merge(user_df,on='uid',how='left')
    result_df.to_csv('../joy_data/DIN_data_more_time.csv',sep='\t',index=False)



