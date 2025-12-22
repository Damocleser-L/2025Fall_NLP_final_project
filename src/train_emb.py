import os
import gensim
from gensim.models import FastText
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_fasttext_model(data_path='../data/processed/', model_path='../models/embedding/', 
                         model_name='twitch_embeddings.model', 
                         vector_size=128, window=5, min_count=5, epochs=10):
    """
    训练 FastText 词嵌入模型（使用 Gensim）

    参数:
    - data_path: 训练数据路径（TXT 文件目录）
    - model_path: 模型保存路径
    - model_name: 模型文件名
    - vector_size: 向量维度
    - window: 上下文窗口大小
    - min_count: 最小词频
    - epochs: 训练轮数
    """
    # 确保模型目录存在
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        logging.info(f"创建模型目录: {model_path}")

    # 收集所有训练文件
    train_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    if not train_files:
        logging.error(f"在 {data_path} 中未找到 TXT 文件")
        return

    # 读取所有句子
    sentences = []
    for file in train_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                # 假设每行是一个句子，分割成词
                words = line.strip().split()
                if words:
                    sentences.append(words)
    
    logging.info(f"加载了 {len(sentences)} 个句子")

    # 训练模型
    logging.info("开始训练 FastText 模型...")
    model = FastText(
        sentences=sentences,
        vector_size=128,      # 适中的维度
        window=5,             # 适合短弹幕
        min_count=5,          # 保留低频但重要的 Emotes
        epochs=10,            # 迭代次数
        sg=1,                 # 使用 Skip-gram
        min_n=3,              # n-gram 最小长度
        max_n=6,              # n-gram 最大长度
    )
    logging.info("训练完成！")

    # 保存模型
    model_file = os.path.join(model_path, model_name)
    model.save(model_file)
    logging.info(f"模型已保存到: {model_file}")

    return model

if __name__ == "__main__":
    # 训练模型
    model = train_fasttext_model()
    # 进一步测试
    if model:
        print("\n=== 模型测试结果 ===")
        print(f"词汇表大小: {len(model.wv)}")
        
        # 测试词列表（基于弹幕数据选择常见词）
        test_words = ['cinema', 'bang', 'beacon', 'o7', 'beautiful', 'pog', 'wow', 'sobbing', 'noice', 'ome']
        for word in test_words:
            if word in model.wv:
                print(f"\n'{word}' 的相似词:")
                similars = model.wv.most_similar(word, topn=5)
                for sim_word, score in similars:
                    print(f"  {sim_word}: {score:.3f}")
            else:
                print(f"\n'{word}' 不在词汇表中")
        
        # 测试词向量
        if 'cinema' in model.wv and 'beautiful' in model.wv:
            similarity = model.wv.similarity('cinema', 'beautiful')
            print(f"\n'cinema' 和 'beautiful' 的余弦相似度: {similarity:.3f}")
        # 输出词汇表按出现次数排序
        vocab_counts = [(word, model.wv.get_vecattr(word, 'count')) for word in model.wv.index_to_key]
        vocab_counts.sort(key=lambda x: x[1], reverse=True)
        
        vocab_file = '../models/embedding/vocab_counts.txt'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write("词\t出现次数\n")
            for word, count in vocab_counts:
                f.write(f"{word}\t{count}\n")
        
        print(f"\n词汇表已保存到: {vocab_file}")
        print(f"前10高频词: {vocab_counts[:10]}")
