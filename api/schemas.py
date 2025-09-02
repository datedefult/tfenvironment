from typing import Optional

from pydantic import BaseModel




class User(BaseModel):
    # 用户uid
    uid:int
    # 帖子类型：0-无限制，1-程序，2-玩具，3-生活，4-共玩
    type:int=0
    # 用户注册渠道：1-ios，2-Google，3-apk，4-三星，5-PC
    register_channel:int = 2
    # 发起请求的频道：1-Following，2-Recommended，3-Progams，4-Latest，5-Toy，6-Life，7-Co-Play
    showcase:int = 2
    # 帖子绑定的产品
    product_code:Optional[int] = None
    # 用户性别
    gender:Optional[int] = None
    # 帖子语言：
    post_language:Optional[str] = None
    # 所在地区
    area:Optional[int] = None
    # 最大返回条数，默认为100
    limit:Optional[int] = 100
