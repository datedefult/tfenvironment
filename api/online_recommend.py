import time
from typing import List, Dict, Any
from pymongo.database import Database
import uvicorn
from fastapi import Depends, FastAPI
from starlette import status
from tensorflow.python.eager.context import async_wait
from redis.asyncio import Redis

from api.BaseResponse import error_response, success_response
from api.connection import connect_to_mongo, close_mongo_connection, get_mongo_database, init_redis_pools, close_redis_pools,get_redis
from api.schemas import User
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: 初始化 MongoDB
    await connect_to_mongo()
    await init_redis_pools()
    yield
    # shutdown: 关闭 MongoDB
    await close_mongo_connection()
    await close_redis_pools()


app = FastAPI(lifespan=lifespan)


#
# @app.get("/post/{post_id}")
# async def read_item(post_id: int, db=Depends(get_mongo_database)):
#     collection = db["app_posts"]
#
#     item = await collection.find_one({"post_id": post_id} ,{"_id": 0})
#
#     return item or {"message": "Not found"}
#
# # async def get_clicked_ids(redis=Depends(get_redis_client)):
# #     redis.
#
# @app.post("/")
# async def post_item(req:User, db=Depends(get_mongo_database)):
#     collection = db["app_posts"]
#     r = get_redis('pushed')
#     redis_pipe = r.pipeline()
#
#
#     time_now = int(time.time())
#     match_query ={
#                 "post_info.deleted_at": 0,  # deleted_at 等于 0 未删除
#                 "post_info.visible": 1,  # visible 等于 1 所有人可见
#                 "post_info.show_type": 0,  # show_type 等于 0 正常类型的帖子
#                 "post_info.status": 30,  # status 等于 30 审核通过的帖子
#                 "post_info.released_at": {"$lt": time_now},  # released_at 小于当前时间戳
#                 "sim_first_img": [],  # 查找没有相似首图的第一个帖子
#                 # "post_score": {"$gt": -0.0001}, # post_score 大于 0
#                 "post_info.channel":{"$in": [req.register_channel]},
#
#             }
#
#     if req.type != 0:
#         # post.type 等于 当前查询type的
#         match_query["post_info.type"] = req.type
#
#     if req.product_code is not None:
#         match_query["post_info.product_code"] = req.product_code
#
#     # if req.area is not None:
#     #     query["post_info.area"] = req.area
#
#     if req.post_language is not None:
#         match_query["post_text_language"] = req.post_language
#
#     print(match_query)
#
#     # 构建聚合管道
#     pipeline = [
#         {"$match": match_query},  # 初始查询条件
#
#         # 按 post_score 降序排序，确保每组中最大的分数排在最前面
#         {"$sort": {"post_score": -1}},
#
#         # 字段筛选：只保留你需要的字段
#         {"$project": {
#             "_id": 0,
#             "post_id": 1,
#             "post_score": 1
#         }},
#
#         # 再次排序（可选）：比如按 post_score 排序返回结果
#         {"$sort": {"post_score": -1}},
#
#         # 限制最多返回 10 个用户的结果
#         {"$limit": 100}
#     ]
#
#     items = await collection.aggregate(pipeline).to_list()
#
#     # 用户信息，用户偏好，内容信息，时间
#     pushed_key = f"pushed:{req.showcase}:{req.uid}"
#     for post in items:
#         await redis_pipe.getbit(pushed_key, post['post_id'])
#     pushed_results = await redis_pipe.execute()
#
#     res = [post_id for post_id, seen in zip(items, pushed_results) if seen == 0]
#
#     return res or {"message": "Not found"}



# 🔧 生成 Mongo 查询条件
def build_match_query(req: User, time_now: int) -> Dict[str, Any]:
    query = {
        "post_info.deleted_at": 0,
        "post_info.visible": 1,
        "post_info.show_type": 0,
        "post_info.status": 30,
        "post_info.released_at": {"$lt": time_now},
        "sim_first_img": [],
        "post_info.channel": {"$in": [req.register_channel]},
    }

    if req.type != 0:
        query["post_info.type"] = req.type

    if req.product_code is not None:
        query["post_info.product_code"] = req.product_code

    if req.post_language is not None:
        query["post_text_language"] = req.post_language

    return query


# ✅ 批量过滤已曝光内容
async def filter_unseen_posts(
    redis_conn: Redis, pushed_key: str, posts: List[Dict[str, Any]]
) -> List[Any]:
    pipe = redis_conn.pipeline()
    for post in posts:
        await pipe.getbit(pushed_key, post["post_id"])
    seen_flags = await pipe.execute()
    return [post for post, seen in zip(posts, seen_flags) if seen == 0]


async def ctr_rank_func(user_req:User,post_info_list:List[Dict[str, Any]]):
    """

    :param user_req:
    :param post_info_list:
    :return:
    """
    uid = user_req.uid
    type = user_req.type
    register_channel = user_req.register_channel
    showcase = user_req.showcase
    product_code = user_req.product_code
    gender = user_req.gender
    post_language = user_req.post_language
    area = user_req.area






# 🚀 推荐接口
@app.post("/")
async def post_item(req: User, db: Database = Depends(get_mongo_database)):
    try:
        collection = db["app_posts"]
        redis_conn = get_redis("pushed")

        time_now = int(time.time())
        match_query = build_match_query(req, time_now)

        # Mongo 聚合查询
        pipeline = [
            {"$match": match_query},
            {"$sort": {"post_score": -1}},
            {"$project": {"_id": 0, "post_id": 1, "post_score": 1}},
        ]
        posts = await collection.aggregate(pipeline).to_list()


        if not posts:
            return success_response(data=[], message="No matched posts")

        # 曝光过滤
        pushed_key = f"pushed:{req.showcase}:{req.uid}"
        unseen_posts = await filter_unseen_posts(redis_conn, pushed_key, posts)
        unseen_posts = unseen_posts[:req.limit]

        rank_recommend =  await ctr_rank_func(req,unseen_posts)

        return success_response(data=unseen_posts, message="Unseen posts retrieved",length = len(unseen_posts))

    except Exception as e:
        # raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
        return error_response(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"服务器内部错误",
            data=str(e),
        )