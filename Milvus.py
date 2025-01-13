from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import pandas as pd
from openai import OpenAI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, db, utility

# 从 .env 文件加载环境变量
load_dotenv()

# 初始化 Embedding Client
embedding_client = OpenAI(
    # 从环境变量中获取 SiliconFlow API 密钥
    api_key=os.getenv("SILICONFLOW_API_KEY"), 
    base_url="https://api.siliconflow.cn/v1"
    )

# 使用 SQLAlchemy 加载 PostgreSQL 数据库
def load_data_from_postgres():
    # PostgreSQL 数据库连接配置（使用 SQLAlchemy）
    db_url = "postgresql+psycopg2://postgres:pwd@localhost:5432/Essentia"
    engine = create_engine(db_url)

    query = "SELECT volume, chapter, paragraph, sentence, context FROM song_shi;"
    
    # 使用 SQLAlchemy 引擎加载数据
    data = pd.read_sql(query, engine)
    
    print(f"Loaded {len(data)} rows from PostgreSQL.")
    return data

# 向量化数据
def generate_embeddings(texts, batch_size=32):
    embeddings = []

    # 如果 texts 的长度小于 batch_size，则不将输入文本分成批次
    if len(texts) <= batch_size:
        batch_text = texts
        # 使用 BAAI 的 bge-large-zh-v1.5 生成嵌入
        response = embedding_client.embeddings.create(
            model="BAAI/bge-large-zh-v1.5",
            input=batch_text
        )
        embeddings = [res.embedding for res in response.data]
    else:
        # 如果 texts 的长度大于 batch_size，则将输入文本分成批次
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]
            # 使用 BAAI 的 bge-large-zh-v1.5 生成嵌入
            response = embedding_client.embeddings.create(
                model="BAAI/bge-large-zh-v1.5",
                input=batch_text
            )
            # 提取嵌入并将其添加到结果列表中
            embeddings.extend([res.embedding for res in response.data])
    return embeddings

# 初始化 Milvus 连接
def init_milvus():
    connections.connect("default", host="localhost", port="19530")

    # 检查并切换到数据库 Essentia
    db_name = "Essentia"
    if db_name not in db.list_database(): 
        db.create_database(db_name)
    db.using_database(db_name)

    # 定义 Collection Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True), 
        FieldSchema(name="volume", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="paragraph", dtype=DataType.INT64),
        FieldSchema(name="sentence", dtype=DataType.INT64),
        FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # 嵌入维度为 1024
    ]
    schema = CollectionSchema(fields, "Song Shi embeddings collection")

    # 创建或加载 Collection
    collection_name = "song_shi"
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)

    # 如果索引不存在，则创建索引
    if not collection.has_index():
        index_params = {
            "index_type": "IVF_FLAT",  # 可选择其他类型，如 IVF_SQ8
            "metric_type": "L2",      # 使用 L2 距离或余弦相似度（IP）
            "params": {"nlist": 128}  # nlist 参数根据数据量调整
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    return collection

# 存储数据到 Milvus 数据库
def insert_into_milvus(collection, data, embeddings):
    # 组装插入数据
    insert_data = [
        list(range(len(data))), # id
        data['volume'].tolist(),
        data['chapter'].tolist(),
        data['paragraph'].tolist(),
        data['sentence'].tolist(),
        data['context'].tolist(),
        embeddings
    ]

    # 插入数据
    collection.insert(insert_data)

    # 加载索引到内存中
    collection.load()

    print("Data inserted into Milvus.")


if __name__ == "__main__":
    # Step 1: 数据加载
    data = load_data_from_postgres()

    # Step 2: 向量化
    print("Generating embeddings...")
    embeddings = generate_embeddings(data['context'].tolist())

    # Step 3: 初始化 Milvus 并存储数据
    print("Initializing Milvus...")
    collection = init_milvus()
    insert_into_milvus(collection, data, embeddings)
