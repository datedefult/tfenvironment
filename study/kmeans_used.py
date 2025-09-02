import random
from collections import Counter

import joblib
import warnings
warnings.filterwarnings("ignore")

# class PredictData:
#
#     def __init__(self):
#         pass
#     def generate_single_data(self):
#         """
#         生成单组测试数据
#         """
#         gender = random.choice([0, 1, 2, 3])  # 性别：4种类别
#         language = random.choice([1, 2, 3])  # 语言：3种类别
#         channel = random.choice([1, 2, 3, 4, 5])  # 渠道：5种类别
#         author_language = random.choice([1, 2, 3])  # 作者语言：3种类别
#         country_code = random.choice([1, 2, 3])  # 国家代码：3种类别
#         author_country_code = random.choice([1, 2, 3])  # 作者国家代码：3种类别
#         friends_num = random.randint(0, 100)  # 好友数量：0~100（整数）
#         binding_toys_num = random.randint(0, 100)  # 绑定玩具数量：0~100（整数）
#         hit_rate = round(random.uniform(0, 100), 2)  # 点击率：0~100（浮点数，保留两位小数）
#         like_rate = round(random.uniform(0, 100), 2)  # 点赞率：0~100（浮点数，保留两位小数）
#         collect_rate = round(random.uniform(0, 100), 2)  # 收藏率：0~100（浮点数，保留两位小数）
#         comments_rate = round(random.uniform(0, 100), 2)  # 评论率：0~100（浮点数，保留两位小数）
#         score = random.randint(0, 100)  # 分数：0~100（整数）
#
#         # 返回单组数据
#         return [
#             gender,
#             language,
#             channel,
#             author_language,
#             country_code,
#             author_country_code,
#             friends_num,
#             binding_toys_num,
#             hit_rate,
#             like_rate,
#             collect_rate,
#             comments_rate,
#             score
#         ]
#     @classmethod
#     def generate_multiple_data(cls, num_samples=1):
#         """
#         Class method that can be called without instantiation
#         """
#         instance = cls()
#         data = []
#         for _ in range(num_samples):
#             data.append(instance.generate_single_data())
#         return data


import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn
import random


class PredictData:
    @classmethod
    def generate_multiple_data(cls, num_samples=1):
        """生成多组测试数据"""
        data = []
        for _ in range(num_samples):
            data.append(cls.generate_single_data())
        return data

    @staticmethod
    def generate_single_data():
        """生成单组测试数据"""
        return [
            random.choice([0, 1, 2, 3]),  # gender
            random.choice([1, 2, 3]),  # language
            random.choice([1, 2, 3, 4, 5]),  # channel
            random.choice([1, 2, 3]),  # author_language
            random.choice([1, 2, 3]),  # country_code
            random.choice([1, 2, 3]),  # author_country_code
            # random.randint(0, 100),  # friends_num
            # random.randint(0, 100),  # binding_toys_num
            # round(random.uniform(0, 100), 2),  # hit_rate
            # round(random.uniform(0, 100), 2),  # like_rate
            # round(random.uniform(0, 100), 2),  # collect_rate
            # round(random.uniform(0, 100), 2),  # comments_rate
            # random.randint(0, 100)  # score
        ]


def predict_batch(batch_data, scaler, model):
    """处理一批数据的预测"""
    scaled_data = scaler.transform(batch_data)
    return model.predict(scaled_data)


def parallel_predict_with_progress(total_iterations, batch_size, scaler, model, max_workers=8):
    """
    使用多线程和Rich进度条进行并行预测

    参数:
        total_iterations: 总迭代次数
        batch_size: 每批数据大小
        scaler: 数据缩放器
        model: 预测模型
        max_workers: 最大线程数
    """
    cluster_result = []

    # 创建Rich进度条
    progress_columns = [
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
    ]

    with Progress(*progress_columns) as progress:
        task = progress.add_task("[cyan]预测进度...", total=total_iterations)

        # 使用线程池
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 准备部分应用的函数
            predict_func = partial(predict_batch, scaler=scaler, model=model)

            # 提交所有任务
            futures = []
            for _ in range(total_iterations):
                data = PredictData.generate_multiple_data(batch_size)
                future = executor.submit(predict_func, data)
                futures.append(future)

            # 处理完成的任务并更新进度条
            for future in as_completed(futures):
                batch_result = future.result()
                cluster_result.extend(batch_result)
                progress.update(task, advance=1)

    return cluster_result

if __name__ == '__main__':

    # 加载模型和标准化器
    loaded_kmeans = joblib.load("./kmode/kmeans_model.joblib")
    loaded_scaler = joblib.load("./kmode/scaler.joblib")

    # cluster_result = []

    # 运行并行预测
    total_iterations = 1000
    batch_size = 200
    max_workers = 8  # 根据你的CPU核心数调整

    print("开始并行预测...")
    cluster_result = parallel_predict_with_progress(
        total_iterations=total_iterations,
        batch_size=batch_size,
        scaler=loaded_scaler,
        model=loaded_kmeans,
        max_workers=max_workers
    )

    print(f"\n预测完成! 总结果数: {len(cluster_result)}")

    counter = Counter(cluster_result)
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # 打印结果
    for item, count in sorted_counts:
        print(f"{item}: {count} 次")
