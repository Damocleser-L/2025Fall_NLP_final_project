# 2025Fall_NLP_final_project

Extracting video narratives from time-sync comments using TCN and domain-customized embeddings.

## 项目描述

本项目旨在从 Twitch 直播弹幕（时间同步评论）中提取视频叙事。通过预处理弹幕数据、使用 FastText 训练领域定制的词嵌入、拟合 IDF 权重、构建时间槽特征，并利用 Temporal Convolutional Network (TCN) 进行序列建模，实现情感分析、聊天总结或其他 NLP 下游任务。

## 主要功能

- **数据预处理**：清洗 Twitch 弹幕数据，过滤机器人、处理 emoji、URL、重复字符等，输出带时间戳的 CSV。
- **词嵌入训练**：使用 Gensim 的 FastText 训练定制词嵌入，适用于直播聊天领域。
- **IDF 拟合**：基于语料库计算词的逆文档频率，用于加权嵌入。
- **槽特征构建**：将弹幕聚合为 5 秒时间槽，计算 IDF 加权平均向量（L2 归一化）和密度特征（log(1+x) + Min-Max 缩放）。
- **下游任务**：使用 TCN 模型进行时间序列分析，如情感分类或叙事生成。
- **自动化流程**：支持批量处理、模型训练和特征验证。

## 安装和依赖

1. **克隆项目**：
   ```bash
   git clone <repository-url>
   cd 2025Fall_NLP_final_project
   ```

2. **创建虚拟环境**（推荐）：
   ```bash
   conda create -n nlp_project python=3.10
   conda activate nlp_project
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

   主要依赖：
   - pandas: 数据处理
   - emoji: Emoji 处理
   - gensim: 词嵌入训练（包含 FastText 实现）
   - scikit-learn: IDF 拟合和相似度计算
   - numpy: 数值计算
   - torch: TCN 模型（可选，需额外安装：`pip install torch`）

## 使用说明

### 1. 数据准备
- 将原始 Twitch CSV 文件放入 `data/raw/` 目录。
- CSV 格式：包含 `time`、`user_name`、`message` 等列。

### 2. 预处理数据
```bash
cd src
python preprocess.py
```
- 输入：`data/waiting/` 中的 CSV 文件（从 `data/raw/` 复制需要处理的）。
- 输出：`data/processed/` 中的 TXT 文件（每行一条清洗后的消息）和 `data/processed_with_ts/` 中的带时间戳 CSV。
- 处理完成后，`data/waiting/` 中的文件会被自动删除。

### 3. 训练词嵌入
```bash
python train_emb.py
```
- 使用 `data/processed/` 中的 TXT 文件训练 FastText 模型。
- 输出：`models/embedding/twitch_embeddings.model` 和词汇统计。

### 4. 拟合 IDF
```bash
python fit_idf.py --data_path ../data/processed/ --out_dir ../models/embedding/
```
- 基于语料库计算 IDF 映射。
- 输出：`models/embedding/idf.json` 和 `models/embedding/vectorizer.pkl`。

### 5. 构建槽特征
```bash
python build_slots.py
```
- 使用带时间戳的 CSV 和训练好的模型构建 5 秒槽特征。
- 输出：`data/slots/` 中的 `.npy` 文件，每个文件包含 (num_slots, 129) 的特征数组。

### 6. 验证槽特征
```bash
python check_slots.py
```
- 检查生成的 `.npy` 文件的形状、范数、密度分布和相邻槽相似度。

### 7. 训练 TCN 模型
```bash
python tcn_model.py
```
- 使用槽特征训练 TCN 模型进行下游任务。
- 输出：`models/tcn/` 中的模型文件。

### 8. 其他脚本
- `download_chat.py`：下载 Twitch 聊天数据。
- `summarize.py`：生成聊天总结。

## 项目结构

```
2025Fall_NLP_final_project/
├── LICENSE
├── README.md
├── requirements.txt
├── video_list.txt
├── data/
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后 TXT 数据
│   ├── processed_with_ts/ # 带时间戳的 CSV 数据
│   ├── slots/            # 槽特征 .npy 文件
│   └── waiting/          # 待处理数据（临时）
├── models/
│   ├── embedding/        # FastText 模型和 IDF 文件
│   └── tcn/              # TCN 模型
├── notebooks/            # Jupyter 实验笔记本
└── src/
    ├── preprocess.py     # 数据预处理
    ├── train_emb.py      # 词嵌入训练
    ├── fit_idf.py        # IDF 拟合
    ├── build_slots.py    # 槽特征构建
    ├── check_slots.py    # 特征验证
    ├── tcn_model.py      # TCN 模型训练
    ├── download_chat.py  # 数据下载
    └── summarize.py      # 总结生成
```

## 注意事项

- 确保数据格式正确，避免 CSV 解析错误。
- 槽特征构建假设时间戳为秒，槽持续时间为 5 秒，可通过参数调整。
- 向量已 L2 归一化，密度经 log(1+x) 和 Min-Max 缩放，适合 TCN 输入。
- 训练大模型时，建议使用 GPU（`torch` 支持 CUDA）。
- 如果遇到 emoji 处理问题，检查 `emoji` 库版本。

