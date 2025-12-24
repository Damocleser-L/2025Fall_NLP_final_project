import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim.models import FastText

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_idf(idf_path: str) -> Dict[str, float]:
    """Load IDF mapping from JSON."""
    with open(idf_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_fasttext_model(model_path: str) -> FastText:
    """Load pre-trained FastText model."""
    return FastText.load(model_path)


def tokenize_message(message: str) -> List[str]:
    """Tokenize message using whitespace (consistent with IDF fitting)."""
    return message.strip().split()


def compute_slot_features(
    df: pd.DataFrame,
    fasttext_model: FastText,
    idf_map: Dict[str, float],
    slot_duration: int = 5
) -> np.ndarray:
    """
    Compute 5-second slot features: weighted average of word vectors + density.

    Args:
        df: DataFrame with 'time' and 'cleaned_message' columns.
        fasttext_model: Loaded FastText model.
        idf_map: Word -> IDF mapping.
        slot_duration: Slot duration in seconds.

    Returns:
        Numpy array of shape (num_slots, 129): 128-dim vector + 1 density.
    """
    if df.empty:
        return np.array([]).reshape(0, 129)

    # Ensure time is numeric (assume seconds since start)
    df = df.copy()
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])

    if df.empty:
        return np.array([]).reshape(0, 129)

    # Compute slot IDs
    df['slot_id'] = (df['time'] // slot_duration).astype(int)
    max_slot = df['slot_id'].max()
    num_slots = max_slot + 1

    # Initialize feature matrix and collect densities for scaling
    features = np.zeros((num_slots, 129))
    raw_densities = np.zeros(num_slots)

    # Group by slot and compute raw features
    for slot_id, group in df.groupby('slot_id'):
        messages = group['cleaned_message'].tolist()
        density = len(messages)  # Number of messages in slot
        raw_densities[slot_id] = density

        # Collect weighted vectors
        weighted_vectors = []
        weights = []

        for msg in messages:
            tokens = tokenize_message(msg)
            for token in tokens:
                if token in fasttext_model.wv:
                    vector = fasttext_model.wv[token]
                    idf_weight = idf_map.get(token, min(idf_map.values(), default=1.0))  # Default to min IDF
                    weighted_vectors.append(vector * idf_weight)
                    weights.append(idf_weight)

        # Compute weighted average vector
        if weighted_vectors:
            avg_vector = np.average(weighted_vectors, axis=0, weights=weights)
        else:
            avg_vector = np.zeros(128)  # Zero vector if no valid tokens

        # L2 normalize the 128-dim vector
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm

        # Store normalized vector (density will be scaled later)
        features[slot_id, :128] = avg_vector

    # Now scale densities globally for this file
    log_densities = np.log1p(raw_densities)  # log(1 + density)
    min_log_density = np.min(log_densities)
    max_log_density = np.max(log_densities)
    if max_log_density > min_log_density:
        scaled_densities = (log_densities - min_log_density) / (max_log_density - min_log_density)
    else:
        scaled_densities = np.zeros_like(log_densities)

    # Set scaled densities
    features[:, 128] = scaled_densities

    return features


def process_file(
    file_path: str,
    fasttext_model: FastText,
    idf_map: Dict[str, float],
    output_dir: str,
    slot_duration: int = 5
) -> None:
    """Process a single CSV file and save slot features."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if 'time' not in df.columns or 'cleaned_message' not in df.columns:
            logging.warning(f"文件 {file_path} 缺少必要列，已跳过")
            return

        features = compute_slot_features(df, fasttext_model, idf_map, slot_duration)

        if features.size == 0:
            logging.warning(f"文件 {file_path} 无有效特征，已跳过")
            return

        # Save as .npy
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_slots.npy")
        np.save(output_path, features)

        logging.info(f"处理完成: {file_path} -> {output_path}, 槽数: {features.shape[0]}")

    except Exception as e:
        logging.error(f"处理文件 {file_path} 时出错: {e}")


def main(data_dir: str, model_path: str, idf_path: str, output_dir: str, slot_duration: int = 5) -> None:
    if not os.path.exists(data_dir):
        logging.error(f"数据目录不存在: {data_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"创建输出目录: {output_dir}")

    # Load models
    fasttext_model = load_fasttext_model(model_path)
    idf_map = load_idf(idf_path)

    # Process each CSV
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        logging.warning(f"在 {data_dir} 中未找到 CSV 文件")
        return

    for file_path in csv_files:
        process_file(file_path, fasttext_model, idf_map, output_dir, slot_duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build 5-second slot features for TCN")
    parser.add_argument('--data_dir', default='../data/processed_with_ts/', help='带时间戳的清洗数据目录')
    parser.add_argument('--model_path', default='../models/embedding/twitch_embeddings.model', help='FastText 模型路径')
    parser.add_argument('--idf_path', default='../models/embedding/idf.json', help='IDF 映射路径')
    parser.add_argument('--output_dir', default='../data/slots/', help='槽特征输出目录')
    parser.add_argument('--slot_duration', type=int, default=5, help='槽持续时间（秒）')
    args = parser.parse_args()

    main(args.data_dir, args.model_path, args.idf_path, args.output_dir, args.slot_duration)
