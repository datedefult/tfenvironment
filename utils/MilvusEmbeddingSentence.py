import torch
from towhee import pipe
from transformers import AutoTokenizer, AutoModel
from Config.milvusConfig import *
from utils.LogsColor import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from Connector.MilvusConnection import EMB_COLLECTION
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, db=MILVUS_DB_NAME, keep_alive=True)
milvus_collection = Collection(MILVUS_COLLECTION)

# 加载模型
model_name = "/home/ubuntu/jgs/huggingface/hub/trained_models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/ae06c001a2546bef168b9bf8f570ccb1a16aaa27"
text_tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 动态选择设备
text_emd_model = AutoModel.from_pretrained(model_name).to(device)  # 模型加载到指定设备


# 嵌入生成函数
def generate_embeddings_with_manual(texts):
    """
    使用 AutoTokenizer 和 AutoModel 生成文本嵌入。
    Args:
        texts (list[str]): 文本列表。
    Returns:
        numpy.ndarray: 文本嵌入数组。
    """
    embeddings = []
    for text in texts:
        # 编码文本
        inputs = text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)  # 移动到当前设备

        # 计算嵌入
        with torch.no_grad():
            outputs = text_emd_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 嵌入
            embeddings.append(cls_embedding.squeeze().cpu().numpy())  # 转回 CPU 并转换为 NumPy

    return embeddings


def insert_with_partition(post_ids, labels, sen_embeddings):
    for content_id, label, sen_embedding in zip(post_ids, labels, sen_embeddings):
        partition_name = label
        try:
            # 检查分区是否存在
            if not milvus_collection.has_partition(partition_name):
                milvus_collection.create_partition(partition_name)
                milvus_collection.load(partition_names=[partition_name])
        except Exception as e:
            logging.error(f"Error check data in {partition_name}: {e}")

        # 调整 img_embedding 的形状为 (1, 512)
        sen_embedding = sen_embedding.reshape(1, -1)

        data = [
            [content_id],  # ID 列
            [label],  # label 列
            sen_embedding,  # 确保形状是 (1, 512)
        ]

        try:
            milvus_collection.upsert(data, partition_name=partition_name)
            # 确保在程序结束前关闭连接
            # connections.disconnect("default")
            # milvus_collection.upsert(data)
        except Exception as e:
            logging.error(f"Error inserting data into {partition_name}: {e}")


text_pipeline = (pipe
                 .input('post_id', 'type', 'text')
                 .map('text', 'text_emb', generate_embeddings_with_manual)
                 .map(('post_id', 'type', 'text_emb'), 'mr', insert_with_partition)
                 .output('post_id', 'text_emb')
                 )


def run_text_embedding(post_id_list, type_list, text_list):
    text_embs_list = generate_embeddings_with_manual(text_list)
    insert_with_partition(post_id_list, type_list, text_embs_list)
    return post_id_list, text_embs_list


if __name__ == '__main__':
    post_id_list = [7, 8]
    type_list = ["113", '12312312312']
    texts = [
        "asdasdvefgb",
        "发生发生发生"
    ]
    print(text_pipeline(post_id_list, type_list, texts).get())

    # print(run_text_embedding(post_id_list, type_list, texts))
