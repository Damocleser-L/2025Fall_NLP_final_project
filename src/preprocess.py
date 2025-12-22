import pandas as pd
import re
import emoji
import os
import logging

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TwitchPreprocessor:
    def __init__(self, data_path='../data/waiting/', output_path='../data/processed/'):
        self.data_path = data_path
        self.output_path = output_path
        # 匹配 URL
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        # 匹配重复 3 次及以上的字符
        self.repeat_pattern = re.compile(r'(.)\1{2,}')
        # 匹配 Twitch 命令 (!command)
        self.cmd_pattern = re.compile(r'^!\w+')
        # 常见机器人名单
        self.bots = {'streamelements', 'nightbot', 'moobot', 'fossabot', 'streamelement', 'sethfromtheseedstore'}

    def clean_text(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return ""

        # 1. 尝试使用 demojize 转换已知 Emoji
        text = emoji.demojize(text, delimiters=(" EMOJI_", " "))

        # 2. 检查是否还有残留的非 ASCII 字符（如 😭）
        # 如果你决定不保留中文，可以直接用之前的 encode/decode 删掉
        # 如果要保留中文但处理掉残留 Emoji，可以使用以下正则：
        # 这个正则匹配所有非 ASCII 且非中文（\u4e00-\u9fa5）的字符
        remaining_emoji_pattern = re.compile(r'[^\x00-\x7f\u4e00-\u9fa5]+')
        
        # 选项 A：将残留的特殊表情替换为统一占位符 [UNK_EMOJI]
        text = remaining_emoji_pattern.sub(' UNK_EMOJI ', text)

        # 3. 删除 URL 和 Twitch 命令
        text = self.url_pattern.sub('', text)
        text = self.cmd_pattern.sub('', text)

        # 4. 缩减重复字符 (Hiiii -> Hii, 哈哈哈 -> 哈哈)
        # 针对 FastText，保留 2 个重复足以表达“程度”，同时减少特征爆炸
        text = self.repeat_pattern.sub(r'\1\1', text)

        # 5. 删除标点符号，但保留字母、数字、中文和空格
        # 使用 flags=re.UNICODE 确保在不同环境下都能正确保留中文字符
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)

        # 6. 清理多余空格并统一小写（中文不受 lower 影响）
        text = " ".join(text.split()).lower()
        
        return text

    def run(self):
        if not os.path.exists(self.data_path):
            logging.error(f"输入路径不存在: {self.data_path}")
            return

        # 确保输出目录存在
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            logging.info(f"创建输出目录: {self.output_path}")

        files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        if not files:
            logging.warning(f"在 {self.data_path} 中未找到 CSV 文件")
            return

        for file in files:
            file_path = os.path.join(self.data_path, file)
            try:
                # 增加 encoding='utf-8' 并处理不规则行和引号
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', quoting=3)
                
                # 检查必要的列是否存在
                if 'user_name' not in df.columns or 'message' not in df.columns:
                    logging.warning(f"文件 {file} 缺少必要列，已跳过")
                    continue

                # 过滤机器人
                df = df[~df['user_name'].str.lower().isin(self.bots)]
                
                # 清洗消息
                cleaned = df['message'].apply(self.clean_text)
                
                # 过滤掉清洗后为空的行
                cleaned = cleaned[cleaned != ""]
                
                # 生成输出文件名：与输入文件名一致，但扩展名为 .txt
                base_name = os.path.splitext(file)[0]
                output_file_path = os.path.join(self.output_path, f"{base_name}.txt")
                
                # 写入文件
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for msg in cleaned:
                        f.write(msg + '\n')
                
                logging.info(f"成功处理文件: {file} -> {output_file_path}, 有效行数: {len(cleaned)}")
                
                # 处理完成后删除 waiting 中的文件
                os.remove(file_path)
                logging.info(f"已删除处理完成的原始文件: {file_path}")

            except Exception as e:
                logging.error(f"处理文件 {file} 时出错: {e}")

# --- 执行示例 ---
if __name__ == "__main__":
    # 将需要处理的文件放入 data/waiting/ 目录，程序会自动处理并删除该目录中的文件
    # data/raw/ 中的原始数据会被保留
    processor = TwitchPreprocessor()
    processor.run()