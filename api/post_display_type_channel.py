from enum import Enum

class PostDisplayTypeChannel(Enum):
    recommend = ("推荐", None, 2)
    programs = ("程序", 1, 3)
    coplay = ("共玩", 4, 7)
    latest = ("最新", None, 4)
    toy = ("玩具", 2, 5)
    life = ("生活", 3, 6)

    def __init__(self, chinese_name: str, post_type: int, channel_id: int):
        self.chinese_name = chinese_name
        self.post_type = post_type
        self.channel_id = channel_id

    @classmethod
    def get(cls, name: str):
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f" `{name}` 不存在")

if __name__ == '__main__':
    print(PostDisplayTypeChannel.get('toy').post_type)