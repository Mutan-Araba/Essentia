import re
import psycopg2

# 数据库连接设置
DB_SETTINGS = {
    'dbname': 'Essentia',
    'user': 'postgres',
    'password': 'pwd',
    'host': 'localhost',
    'port': 5432
}

def split_into_sentences(paragraph):
    """
    拆分段落为句子，遵循以下规则：
    1. 先在整个段落中标记句子结束符（。！？），并在其后加上句子分隔符 "§"。
    2. 识别「」内容，如果「前有冒号（：），则视为对话：
       - 删除对话内容中的 "§"。
       - 在对话内容最后加上 "§"。
    3. 最后根据 "§" 划分句子。
    """
    # 第一步：在句子结束符后添加分隔符 "§"
    paragraph = re.sub(r'([。！？])', r'\1§', paragraph)

    # 第二步：识别「」内容，并根据「前是否有冒号判断是否为对话
    def process_dialogue(match):
        """处理「」内容：如果是对话，删除内部的 "§" 并在最后加上 "§" """
        prefix = match.group(1)  # 「前的内容
        content = match.group(2)  # 「」之间的内容
        if prefix.endswith("："):  # 如果「前有冒号，则视为对话
            content = content.replace("§", "")  # 删除对话内容中的 "§"
            return f"{prefix}「{content}」§"  # 在对话内容最后加上 "§"
        else:
            return f"{prefix}「{content}」"  # 非对话内容，不做处理

    # 匹配「」内容，并捕获「前的内容和「」之间的内容
    paragraph = re.sub(r'([^「]*)「([^」]*)」', process_dialogue, paragraph)

    # 第三步：根据 "§" 划分句子
    sentences = [s.strip() for s in paragraph.split("§") if s.strip()]
    return sentences

def parse_text(file_path):
    data = []
    volume = file_path.split("/")[-1].split(".")[0]  # 文件名作为卷名
    chapter = None
    paragraph_counter = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 判断是否为章节名：不超过10个字
        if len(line) <= 10:
            chapter = line
            paragraph_counter = 0  # 章节切换后段落计数重置
            continue
        
        # 正常段落
        paragraph_counter += 1
        sentences = split_into_sentences(line)
        sentence_counter = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_counter += 1
            data.append({
                'volume': volume,
                'chapter': chapter,
                'paragraph': paragraph_counter,
                'sentence': sentence_counter,
                'context': sentence
            })
    return data

def insert_to_db(data):
    connection = psycopg2.connect(**DB_SETTINGS)
    cursor = connection.cursor()

    # 确保表已创建
    create_table_query = """
    CREATE TABLE IF NOT EXISTS song_shi (
        id SERIAL PRIMARY KEY,
        volume VARCHAR(255),
        chapter VARCHAR(255),
        paragraph INT,
        sentence INT,
        context TEXT
    );
    """
    cursor.execute(create_table_query)
    connection.commit()

    # 插入数据
    insert_query = """
    INSERT INTO song_shi (volume, chapter, paragraph, sentence, context)
    VALUES (%s, %s, %s, %s, %s);
    """
    for record in data:
        cursor.execute(insert_query, (
            record['volume'],
            record['chapter'],
            record['paragraph'],
            record['sentence'],
            record['context']
        ))
    connection.commit()
    cursor.close()
    connection.close()

if __name__ == "__main__":
    # 修改文件路径为你的文本文件路径
    file_path = "./卷一 本紀第一.txt"
    parsed_data = parse_text(file_path)
    insert_to_db(parsed_data)
    print(f"Inserted {len(parsed_data)} records into the database.")
