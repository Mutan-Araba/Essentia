from dotenv import load_dotenv
import os
import psycopg2
from openai import OpenAI
from pymilvus import connections, Collection, db

# 从 .env 文件加载环境变量
load_dotenv()

# 初始化 DeepSeek Client
deepseek_client = OpenAI(
    # 从环境变量中获取 DeepSeek API 密钥
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com/v1"
    )

# 初始化 Embedding Client
embedding_client = OpenAI(
    # 从环境变量中获取 SiliconFlow API 密钥
    api_key=os.getenv("SILICONFLOW_API_KEY"), 
    base_url="https://api.siliconflow.cn/v1"
    )

# 初始化 PostgreSQL 数据库连接
def init_postgres_connection():
    conn = psycopg2.connect(
        dbname="Essentia",
        user="postgres",
        password="pwd",
        host="localhost",
        port="5432"
    )
    return conn

# 从 PostgreSQL 数据库中查询具体段落内容，并拼接结果
def fetch_paragraphs_from_postgres(conn, volume, chapter, paragraph):
    query = """
    SELECT context 
    FROM song_shi 
    WHERE volume = %s AND chapter = %s AND paragraph = %s;
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (volume, chapter, paragraph))
        results = cursor.fetchall()  # 获取所有匹配的行

    # 拼接所有段落内容
    if results:
        paragraphs = [row[0] for row in results]  # 提取每行的 context
        return "".join(paragraphs)  # 用换行符拼接
    else:
        return None

# 初始化 Milvus 连接
def init_milvus_connection():
    connections.connect("default", host="localhost", port="19530")
    db.using_database("Essentia")
    collection = Collection(name="song_shi")  # 替换为实际的 Collection 名称
    return collection

# 向量化查询
def generate_query_embedding(question):
    response = embedding_client.embeddings.create(
        model="BAAI/bge-large-zh-v1.5",
        input=[question]
    )
    return response.data[0].embedding

# 检索相关段落
def search_in_milvus(collection, query_embedding, top_k=4):
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["volume", "chapter", "paragraph", "context"]
    )
    return results

# 构建上下文
def build_context(results, conn):
    """
    根据 Milvus 检索结果，从 PostgreSQL 数据库中查询对应段落，构建上下文。
    检索结果以去重后的形式返回，标题格式为
    《宋史 · chapter》段 paragraph：
    paragraph_text
    （《宋史》volume）
    """
    seen_paragraphs = set()
    context_parts = []

    for res in results[0]:
        # 获取 Milvus 检索返回的字段
        volume = res.entity.get("volume")
        chapter = res.entity.get("chapter")
        paragraph = res.entity.get("paragraph")

        # 去重逻辑
        paragraph_key = (volume, chapter, paragraph)
        if paragraph_key in seen_paragraphs:
            continue
        seen_paragraphs.add(paragraph_key)

        # 查询 PostgreSQL 获取完整段落内容
        paragraph_text = fetch_paragraphs_from_postgres(conn, volume, chapter, paragraph)
        if paragraph_text:
            # 按照指定格式添加段落内容
            context_parts.append(f"""
            据《宋史 · {chapter}》段 {paragraph} 记载：
            {paragraph_text}
            (《宋史》{volume})
            """)
    
    return "\n".join(context_parts)

# 生成回答
def generate_answer(question, context):
    """
    使用 DeepSeek 模型根据上下文生成回答。

    参数:
        question (str): 用户的问题。
        context (str): 提供给模型的上下文信息。

    返回:
        str: 模型生成的回答。
    """
    system_prompt = f"""
    请你仔细阅读相关内容，结合历史资料进行回答，

    1. 如果问题有明确的史料支持，请按以下格式回答：
    据 [具体的史料名称或来源] 记载：
    [仅原样引用该史料中与问题最相关的一句原文]

    [你的回答]

    2. 如果问题无法从提供的资料中找到答案，
    请回答：“根据现有史料，无法确定答案。”

    搜索的相关历史资料如下所示：
    ---------------------
    {context}
    ---------------------"""

    user_prompt = f"""
    问题: {question}
    答案:"""

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000,
        stream=False
    )
    return response.choices[0].message.content.strip()

# 交互式命令行流程
def interactive_rag_tool():
    print("Initializing Milvus connection...")
    collection = init_milvus_connection()
    print("Milvus connection established.")
    
    print("Initializing PostgreSQL connection...")
    conn = init_postgres_connection()
    print("PostgreSQL connection established. Ready to process queries.")
    
    try:
        while True:
            question = input("\nEnter your question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                print("Exiting interactive RAG tool.")
                break
            
            print("\nGenerating query embedding...")
            query_embedding = generate_query_embedding(question)
            
            print("Searching for relevant context in Milvus...")
            results = search_in_milvus(collection, query_embedding, top_k=8)
            
            if not results[0]:
                print("No relevant results found in Milvus.")
                continue

            print("Fetching paragraphs from PostgreSQL...")
            context = build_context(results, conn)

            if not context:
                print("No valid paragraphs found in PostgreSQL.")
                continue

            print("\nGenerating answer...")
            answer = generate_answer(question, context)
            print(f"\nAnswer:\n{answer}")
    finally:
        conn.close()
        print("PostgreSQL connection closed.")

if __name__ == "__main__":
    interactive_rag_tool()
