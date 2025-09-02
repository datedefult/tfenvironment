import pandas as pd
import pymysql
from MulConnectionPool import SynDBPools


def fill_missing_values(df):
    """
    按照Columns类型对DataFrame的缺失值进行填补
    :param df:
    :return:
    """
    for column in df.columns:
        if df[column].dtype == 'int64':
            # 对于int64类型的列，用平均数再取整数填充NaN
            avg_int = int(df[column].mean())
            df.loc[:, column] = df[column].fillna(avg_int)
        elif df[column].dtype == 'object':
            # 对于object类型的列，用出现频率最高的元素填充NaN
            most_frequent = df[column].mode()
            if not most_frequent.empty:
                fill_value = most_frequent[0]
                df.loc[:, column] = df[column].fillna(fill_value)
            else:
                print(f"Warning: Column '{column}' has no mode.")
        elif df[column].dtype == 'float64':
            # 对于float64类型的列，用平均数填充NaN
            avg_float = df[column].mean()
            df.loc[:, column] = df[column].fillna(avg_float)
        elif df[column].dtype == 'datetime64[ns]':
            # 对于datetime64[ns]类型的列，用中位数填充NaN
            median_date = df[column].dropna().median()
            df.loc[:, column] = df[column].fillna(median_date)
        else:
            print(f"Unsupported dtype {df[column].dtype} for column {column}.")

    return df



def get_register_data(mongo_collection):
    docs = mongo_collection.find({},{"_id": 0,
                                     "basic_information.uid": 1,
                                     "basic_information.gender": 1,
                                     "basic_information.language": 1,
                                     "basic_information.register_time": 1,
                                     "basic_information.friends_num": 1,
                                     "basic_information.binding_toys_number": 1,
                                     "basic_information.country_code": 1,
                                     "basic_information.channels": 1,
                                     })
    data_list =[]
    for doc in docs:
        doc_information = doc.get("basic_information",{})
        if doc_information:
            data_list.append(doc_information)

    register_df = pd.DataFrame(data_list)
    return register_df


def get_post_data(mysql_conn):
    query = "SELECT `id` post_id,`title`,`user_id` author_id,`type`,`attachments_count`,`likes_count`,`favorites_count`,`comments_count`,`created_at` post_created_at FROM `posts`"
    with mysql_conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())


def get_post_statistics_data(mysql_conn):
    detail_query = "SELECT `post_id`, `push_number`,`hits_rate`,`like_rate`,`collect_rate`,`comments_rate`,`update_time` FROM `post_score_details` "
    score_query = "SELECT `post_id`,`score` FROM `post_score`"
    with mysql_conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
        cursor.execute(detail_query)
        detail_df = pd.DataFrame(cursor.fetchall())
        cursor.execute(score_query)
        score_df = pd.DataFrame(cursor.fetchall())

    return detail_df, score_df

def get_act_data(mysql_conn):
    """
    获取用户内容交互数据
    :param mysql_conn:
    :return:
    """
    query = """SELECT `id`,`timestamp`,`uid`,`post_id`,`is_hit`,`is_like`,`stay_duration` FROM `tensorflow_dataset` ORDER BY RAND() LIMIT 50000"""
    # query = """SELECT `id`,`timestamp`,`uid`,`post_id`,`is_hit`,`is_like`,`stay_duration` FROM `tensorflow_dataset`"""
    # query = """SELECT `id`,`timestamp`,`uid`,`post_id`,`is_hit`,`is_like`,`stay_duration` FROM `tensorflow_dataset` WHERE is_hit=1 """
    with mysql_conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
        cursor.execute(query)
        act_df = pd.DataFrame(cursor.fetchall())
        # content_stats = act_df.groupby('post_id')['is_hit'].agg(['mean', 'count']).rename(
        #     columns={'mean': 'content_ctr', 'count': 'content_impressions'})
        # user_stats = act_df.groupby('uid')['is_hit'].agg(['mean', 'count']).rename(
        #     columns={'mean': 'user_ctr', 'count': 'user_activity'})
        return act_df


def data_loader():

    connection_pools = SynDBPools('E:\pycharmPro\TFenvironment\MulConnectionPool\config')
    connection_pools.init_mysql()
    connection_pools.init_mongo()

    app_client = connection_pools.get_mongo('app','app')
    user_collection = app_client['app_users']

    recommend_conn = connection_pools.get_mysql('content_recommend')
    community_conn = connection_pools.get_mysql('app_community')


    # 用户交互表
    act_df = get_act_data(recommend_conn)
    act_df['timestamp_date'] = pd.to_datetime(act_df['timestamp'], unit='ms')
    act_df['hour'] = act_df['timestamp_date'].dt.hour
    act_df['dow'] = act_df['timestamp_date'].dt.dayofweek
    # 详情分数信息
    detail_df, score_df=get_post_statistics_data(recommend_conn)
    # 帖子信息
    post_df = get_post_data(community_conn)
    # 消费者信息
    register_df = get_register_data(user_collection)
    # 作者信息表
    author_df = register_df.copy()[['uid','language','country_code']]
    author_df.rename(columns={'uid':'author_id','language':'author_language','country_code':'author_country_code'},inplace=True)

    # 合并所有维度特征
    merged_df = act_df.merge(post_df,on='post_id',how='left').merge(detail_df,on='post_id',how='left').merge(score_df,on='post_id',how='left').merge(register_df,on='uid',how='left').merge(author_df,on='author_id',how='left')
    # 填补字段
    clear_df = fill_missing_values(merged_df)
    # 修改字段类型
    channel_str_dict = {'iOS': 1,'google': 2,'apk': 3,'三星': 4,'PC': 5,'pc': 5,}
    clear_df['channels']= clear_df['channels'].map(channel_str_dict).astype(int)
    connection_pools.close()
    return clear_df

if __name__ == '__main__':
    data_loader().to_csv('./test_loader.csv')