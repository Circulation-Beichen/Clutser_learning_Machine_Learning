import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
# 生成合成数据（月牙形，含噪声）
# make_moons函数生成两个交错的半圆形数据集，适合测试非球形聚类算法
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
# 标准化数据，使各特征均值为0，方差为1，有助于提高聚类算法性能
X = StandardScaler().fit_transform(X)

# 2. OPTICS聚类
# OPTICS (Ordering Points To Identify the Clustering Structure) 是一种基于密度的聚类算法
# min_samples: 定义核心点的邻域最小样本数（相当于DBSCAN中的MinPts）
# xi: 用于提取聚类的参数，较小的值会创建更多聚类
# min_cluster_size: 定义一个聚类的最小样本比例
optics_model = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
optics_model.fit(X)

# 3. 提取聚类结果
# labels_: 每个样本的聚类标签，-1表示噪声点
labels = optics_model.labels_

# 4. 可视化聚类结果
plt.figure(figsize=(12, 10))

# 4.1 创建子图布局
plt.subplot(2, 1, 1)
# 获取唯一的标签值（包括噪声标签-1）
unique_labels = set(labels)
# 为每个聚类分配不同的颜色
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

# 绘制每个聚类
for k, col in zip(unique_labels, colors):
    if k == -1:  # 噪声点用黑色表示
        col = [0, 0, 0, 1]
    # 创建当前聚类的掩码
    class_mask = (labels == k)
    
    # 绘制聚类点
    plt.scatter(X[class_mask, 0], X[class_mask, 1],
                c=[tuple(col)], edgecolor='k', s=50, label=f'聚类 {k}')

plt.title('OPTICS聚类结果')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.legend()
plt.grid(True)

# 4.2 使用DBSCAN进行比较（因为DBSCAN有明确的核心点概念）
plt.subplot(2, 1, 2)
# 使用与OPTICS相同的参数
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X)

# 获取核心样本掩码（DBSCAN有这个属性）
core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# 获取唯一的标签值
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

# 绘制每个聚类
for k, col in zip(unique_labels, colors):
    if k == -1:  # 噪声点用黑色表示
        col = [0, 0, 0, 1]
    # 创建当前聚类的掩码
    class_mask = (dbscan_labels == k)
    
    # 绘制核心点（大标记）
    plt.scatter(X[class_mask & core_samples_mask, 0], X[class_mask & core_samples_mask, 1],
                c=[tuple(col)], edgecolor='k', s=150, label=f'聚类 {k} 核心点')
    # 绘制非核心点（小标记）
    plt.scatter(X[class_mask & ~core_samples_mask, 0], X[class_mask & ~core_samples_mask, 1],
                c=[tuple(col)], edgecolor='k', s=50)

plt.title('DBSCAN聚类结果（用于比较）')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('density_clustering.png')
plt.show()

# 5. 输出聚类统计信息
n_clusters_optics = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_optics = list(labels).count(-1)
print(f'OPTICS估计的聚类数: {n_clusters_optics}')
print(f'OPTICS估计的噪声点数: {n_noise_optics}')

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_dbscan = list(dbscan_labels).count(-1)
print(f'DBSCAN估计的聚类数: {n_clusters_dbscan}')
print(f'DBSCAN估计的噪声点数: {n_noise_dbscan}')