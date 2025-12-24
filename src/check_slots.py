import argparse
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def check_npy_file(file_path: str) -> None:
    """检查单个 .npy 文件的合理性"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 加载数组
    features = np.load(file_path)
    print(f"\n文件: {os.path.basename(file_path)}")
    print(f"形状: {features.shape}")
    print(f"数据类型: {features.dtype}")

    if features.size == 0:
        print("数组为空")
        return

    num_slots, feature_dim = features.shape
    if feature_dim != 129:
        print(f"特征维度错误: 期望 129, 实际 {feature_dim}")
        return

    # 分离向量和密度
    vectors = features[:, :128]
    densities = features[:, 128]

    # 检查密度
    print(f"密度统计: 最小={densities.min():.0f}, 最大={densities.max():.0f}, 平均={densities.mean():.2f}")
    print(f"非零密度槽数: {np.sum(densities > 0)} / {num_slots}")

    # 向量范数
    norms = np.linalg.norm(vectors, axis=1)
    print(f"向量范数: 最小={norms.min():.4f}, 最大={norms.max():.4f}, 平均={norms.mean():.4f}")

    # 计算相邻槽的相似度（如果有多个槽）
    if num_slots > 1:
        similarities = []
        for i in range(num_slots - 1):
            vec1 = vectors[i].reshape(1, -1)
            vec2 = vectors[i+1].reshape(1, -1)
            sim = cosine_similarity(vec1, vec2)[0, 0]
            similarities.append(sim)
        similarities = np.array(similarities)
        print(f"相邻槽相似度: 最小={similarities.min():.4f}, 最大={similarities.max():.4f}, 平均={similarities.mean():.4f}")
        print(f"相似度 > 0.5 的比例: {np.mean(similarities > 0.5):.2%}")
    else:
        print("只有一个槽，无法计算相邻相似度")

    # 示例输出前几个槽
    print("\n前 3 个槽示例:")
    for i in range(min(3, num_slots)):
        print(f"  槽 {i}: 密度={densities[i]:.0f}, 向量前3维={vectors[i][:3]}")

def main(slots_dir: str) -> None:
    if not os.path.exists(slots_dir):
        print(f"目录不存在: {slots_dir}")
        return

    npy_files = [os.path.join(slots_dir, f) for f in os.listdir(slots_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"在 {slots_dir} 中未找到 .npy 文件")
        return

    for file_path in npy_files[:1]:  # 只检查第一个文件
        check_npy_file(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="检查槽特征 .npy 文件的合理性")
    parser.add_argument('--slots_dir', default='../data/slots/', help='槽特征目录')
    args = parser.parse_args()

    main(args.slots_dir)