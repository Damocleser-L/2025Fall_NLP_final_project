import argparse
import json
import logging
import os
import pickle
from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def simple_tokenizer(text: str) -> List[str]:
	"""A whitespace tokenizer that preserves emotes/short tokens (pog, o7, ome)."""
	return text.strip().split()


def iter_corpus_lines(data_path: str) -> Iterable[str]:
	"""Yield non-empty lines from all .txt files under data_path."""
	for fname in os.listdir(data_path):
		if not fname.endswith('.txt'):
			continue
		fpath = os.path.join(data_path, fname)
		with open(fpath, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if line:
					yield line


def fit_idf(data_path: str, out_dir: str) -> None:
	if not os.path.exists(data_path):
		logging.error(f"数据目录不存在: {data_path}")
		return

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
		logging.info(f"创建输出目录: {out_dir}")

	corpus_iter = list(iter_corpus_lines(data_path))
	if not corpus_iter:
		logging.error(f"在 {data_path} 未找到有效文本行")
		return

	logging.info(f"加载语料行数: {len(corpus_iter)}")

	vectorizer = TfidfVectorizer(
		tokenizer=simple_tokenizer,
		token_pattern=None,
		lowercase=False,  # 文本已在预处理阶段 lower 过
		use_idf=True,
		smooth_idf=True,
		sublinear_tf=False,
		min_df=1,
	)

	vectorizer.fit(corpus_iter)
	vocab_size = len(vectorizer.vocabulary_)
	logging.info(f"词表大小: {vocab_size}")

	# 导出词 -> IDF 映射
	idf_values = vectorizer.idf_
	idf_map = {
		term: float(idf_values[idx])
		for term, idx in vectorizer.vocabulary_.items()
	}

	idf_path = os.path.join(out_dir, 'idf.json')
	with open(idf_path, 'w', encoding='utf-8') as f:
		json.dump(idf_map, f, ensure_ascii=False, indent=2)
	logging.info(f"已保存 IDF 映射: {idf_path}")

	vec_path = os.path.join(out_dir, 'tfidf_vectorizer.pkl')
	with open(vec_path, 'wb') as f:
		pickle.dump(vectorizer, f)
	logging.info(f"已保存向量化器: {vec_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fit global IDF from processed corpus")
	parser.add_argument('--data_path', default='../data/processed/', help='预处理文本所在目录')
	parser.add_argument('--out_dir', default='../models/embedding/', help='输出 IDF 和向量化器的目录')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	fit_idf(args.data_path, args.out_dir)
