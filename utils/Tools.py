import json
import os
from datetime import datetime, timezone

import pytz
import requests

from utils.LogsColor import logging
from functools import wraps
import time


def timestamp_us_date(timestamp, tag='date'):
    # 创建一个带有 UTC 时区信息的时间对象
    utc_time = datetime.fromtimestamp(timestamp, timezone.utc)

    # 获取美东时区
    eastern_tz = pytz.timezone('America/New_York')
    eastern_time = utc_time.astimezone(eastern_tz)
    # 将 UTC 时间转换为美东时间
    if tag == 'date':
        eastern_time = utc_time.astimezone(eastern_tz).date()
    elif tag == 'time':
        eastern_time = utc_time.astimezone(eastern_tz).replace(tzinfo=None)

    return eastern_time


def timing_decorator(func):
    """
    时间装饰器
    :param func:
    :return:
    """
    @wraps(func)
    def time_wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算花费的时间
        logging.info(f"Function {func.__name__} took {elapsed_time:.6f} seconds to run.")
        return result
    return time_wrapper


def other2float(str_in):
    """ 将其他字符强制转为float类型，默认为0 """
    try:
        return float(str_in)
    except ValueError:
        return 0.0


def other2int(str_in):
    """ 将其他字符强制转为int类型，默认为0 """
    try:
        return int(str_in)
    except ValueError:
        return 0


def list2str_split(in_list):
    outstr = ','.join(map(str, in_list))
    return outstr


def error_chat(err, fun_name, fun_desc):
    chat_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=f5b5627b-5ceb-4796-87d5-4d4ca0f94816'
    header = {
        'Content-Type': "application/json"
    }
    text = f"""
        ### 函数调用出现问题
        - **涉及函数：<font color=\"red\">{fun_name}</font>**
        - **函数信息：<font color=\"red\">{fun_desc}</font>**
        - **报错信息：
            {err}**
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


def execute_with_error_handling(func, *args, **kwargs):
    """
    执行函数并处理异常，如果发生异常会记录错误日志。

    :param func: 需要执行的函数
    :param args: 传递给函数的位置参数
    :param kwargs: 传递给函数的关键字参数
    """
    try:
        logging.info(f"Processing {func.__name__}")
        func(*args, **kwargs)
    except Exception as e:
        error_chat(e, func.__name__, func.__code__)
        logging.error(f"Error during execution of {func.__name__}: {e}")


def merge_lists_strict(default_list, personal_list, default_step=3, person_step=2):
    """
    合并两个列表，默认推荐列表与个性化推荐列表
    :param default_list: 默认列表
    :param personal_list: 个人推荐列表
    :param default_step: 插入间隔
    :param person_step: 插入间隔

    :return: 合并后的展示列表
    """
    merge_list = []
    default_index, personal_index = 0, 0
    seen = set()
    while default_index < len(default_list) or personal_index < len(personal_list):

        a_added = 0  # 当前插入的 a 元素计数
        while default_index < len(default_list) and a_added < default_step:
            if default_list[default_index] not in seen:
                merge_list.append(default_list[default_index])
                seen.add(default_list[default_index])
                a_added += 1
            default_index += 1

        # 插入 2 个来自 b 的非重复元素
        b_added = 0  # 当前插入的 b 元素计数
        while personal_index < len(personal_list) and b_added < person_step:
            if personal_list[personal_index] not in seen:
                merge_list.append(personal_list[personal_index])
                seen.add(personal_list[personal_index])
                b_added += 1
            personal_index += 1

    return merge_list

def merge_lists_alternately(all_list_su, all_list_pro, n, m):
    """
    合并两个列表，默认推荐列表与个性化推荐列表,元素为dict

    :param all_list_su:
    :param all_list_pro:
    :param n:
    :param m:
    :return:
    """
    all_list = []
    su_len, pro_len = len(all_list_su), len(all_list_pro)
    i = j = 0

    while i < su_len or j < pro_len:
        for _ in range(n):
            if i < su_len:
                all_list.append(all_list_su[i])
                i += 1

        for _ in range(m):
            if j < pro_len:
                all_list.append(all_list_pro[j])
                j += 1

    return all_list


def refactor_new_recommendation(in_person_redict, default_redict,forbidden_list, user_dict,default_step=3, person_step=2):
    """
    将用户个性化列表与默认列表进行合并
    :param forbidden_list: 被限制的用户id
    :param user_dict: 用户属性字典
    :param in_person_redict: 用户个性化列表
    :param default_redict: 默认推荐列表
    :return:
    """
    for uid, uid_diff_dict in in_person_redict.items():
        if uid != 0 and uid != '0':
            for type, type_diff_dict in uid_diff_dict.items():
                for product_code, product_code_diff_list in type_diff_dict.items():
                    default_product_list = default_redict.get(0, {}).get(type, {}).get(product_code, [])
                    default_product_list =[item for item in default_product_list if item not in forbidden_list]
                    product_code_diff_list =[item for item in product_code_diff_list if item not in forbidden_list]

                    in_person_redict[uid][type][product_code] = merge_lists_strict(default_product_list,
                                                                                   product_code_diff_list, default_step,
                                                                                   person_step)
    return in_person_redict



def refactor_new_recommendation_new(in_person_redict, default_redict,forbidden_uid_list,forbidden_list, user_dict,post_dict,default_step=8, person_step=2):
    """
    将用户个性化列表与默认列表进行合并
        => 渠道:1-iOS,2-谷歌,3-apk,4-三星,5-pc
        => 性别:0-男,1-女,2-lgbt+,3-保密
        => 国家:2-亚洲,3-欧洲,其余为1

    :param forbidden_uid_list: 被限制的用户列表
    :param forbidden_list: 被限制的用户id 支持查看的帖子
    :param user_dict: 用户属性字典
    :param post_dict: 帖子属性字典
    :param in_person_redict: 用户个性化列表
    :param default_redict: 默认推荐列表 组
    :return:
    """

    gender_str_dict = {
        '男': 0,
        '女': 1,
        'LGBT': 2,
        '保密': 3
    }
    channel_str_dict = {
        'iOS': 1,
        'google': 2,
        'apk': 3,
        '三星': 4,
        'PC': 5,
        'pc': 5,
    }
    for uid, uid_diff_dict in in_person_redict.items():
        if uid != 0 and uid != '0' and uid not in forbidden_uid_list:
            for type, type_diff_dict in uid_diff_dict.items():
                for product_code, product_code_diff_list in type_diff_dict.items():
                    person_user_dict = user_dict.get(uid,{})
                    user_channel = channel_str_dict.get(person_user_dict.get('channels', 'google'))
                    user_continent = person_user_dict.get('continent',1)
                    user_gender = gender_str_dict.get(person_user_dict.get('gender','保密'))

                    default_product_list = default_redict.get(user_channel,{}).get(user_gender,{}).get(user_continent,{}).get(0, {}).get(type, {}).get(product_code, [])


                    result_personal_recommend_list = merge_lists_strict(default_product_list,
                                                                                   product_code_diff_list, default_step,
                                                                                   person_step)

                    # 过滤列表中被禁止查看的贴子
                    result_personal_recommend_list =[item for item in result_personal_recommend_list if item not in forbidden_list]

                    # 过滤用户该渠道不让被查看的贴子
                    result_personal_recommend_list = [item for item in result_personal_recommend_list if user_channel in post_dict.get(item,{}).get('channel',[])]

                    # 将用户适配的玩具进行提前
                    user_bind_products = set(user_dict.get(uid,{}).get("bound_toys_code",[]))

                    # 每个用户的适配列表
                    if user_bind_products:
                        post_list_for_bind = []

                        for post in result_personal_recommend_list:
                            post_productcode = post_dict.get(post,{}).get('product_code',0)

                            if post_productcode in user_bind_products:
                                post_list_for_bind.append(post)
                                user_bind_products.remove(post_productcode)
                        result_personal_recommend_list = merge_lists_strict(post_list_for_bind,result_personal_recommend_list,1,4)

                    in_person_redict[uid][type][product_code] = result_personal_recommend_list

    return in_person_redict



def read_offset(offset_file):
    """读取文件中的偏移值，如果文件不存在则返回 0"""
    if os.path.exists(offset_file):
        with open(offset_file, 'r') as f:
            checkpoint = f.read().strip()
            return int(checkpoint) if checkpoint.isdigit() else 0
    else:
        # 如果文件不存在，则创建文件并返回偏移值 0
        with open(offset_file, 'w') as f:
            f.write('0')
        return 0


def write_offset(offset_file, batch):
    """将当前的 checkpoint 写入文件"""
    with open(offset_file, 'w') as f:
        f.write(str(batch))


# 指定数据库进行数据上传
def execute_bulk_updates(collection, updates, collection_name):
    """执行批量更新并记录日志。"""
    try:
        if updates:
            result = collection.bulk_write(updates)
            logging.info(
                f"Bulk write completed in {collection_name}. "
                f"Matched: {result.matched_count}, Modified: {result.modified_count}, Upserts: {len(result.upserted_ids)}")
    except Exception as e:
        logging.error(f"Error during bulk write in {collection_name}: {e}")


def list_cut_list(list1, list2, chaos_type=1):
    """
    从第一个列表 (list1) 中移除所有出现在第二个列表 (list2) 中的元素。

    :param list1: 原始列表
    :param list2: 包含需要被移除的元素的列表
    :param chaos_type: 操作类型（1 保持顺序，2 不保持顺序）
    :return: 移除指定元素后的新列表
    """
    set2 = set(list2)

    if chaos_type == 1:
        # 保持顺序，并且保留重复项
        return [item for item in list1 if item not in set2]
    elif chaos_type == 2:
        # 不保持顺序，使用集合运算去除重复项
        return list(set(list1) - set2)
    else:
        raise ValueError("Invalid type value. Use 1 for ordered operation and 2 for unordered.")


from math import radians, sin, cos, acos


def compute_distance(slat, slon, elat, elon):
    """
    使用Haversine公式计算两个经纬度点之间的距离。

    :param slat: 起点纬度
    :param slon: 起点经度
    :param elat: 终点纬度
    :param elon: 终点经度
    :return: 两点间的距离（单位：公里）
    """
    # 将十进制度数转化为弧度
    slat, slon, elat, elon = map(radians, [slat, slon, elat, elon])

    # Haversine公式
    dist = 6371.01 * acos(sin(slat) * sin(elat) + cos(slat) * cos(elat) * cos(slon - elon))
    return dist
