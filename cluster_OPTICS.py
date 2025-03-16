import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 0. 配置matplotlib支持中文显示
# 尝试使用不同的中文字体，避免字体缺失警告
try:
    # 尝试使用微软雅黑字体（Windows系统常见字体）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果设置失败，使用英文标签
    print("Warning: Failed to set Chinese fonts, will use English labels")

# 1. 数据准备
# 生成合成数据（月牙形，含噪声）
# make_moons函数生成两个交错的半圆形数据集，适合测试非球形聚类算法
# 增加样本数，减少噪声
X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
# 标准化数据，使各特征均值为0，方差为1，有助于提高聚类算法性能
X = StandardScaler().fit_transform(X)

# 2. OPTICS聚类 - 进一步优化参数
# OPTICS (Ordering Points To Identify the Clustering Structure) 是一种基于密度的聚类算法
# min_samples: 定义核心点的邻域最小样本数（相当于DBSCAN中的MinPts）
# xi: 用于提取聚类的参数，较小的值会创建更多聚类
# min_cluster_size: 定义一个聚类的最小样本比例
# 调整参数以获得更好的聚类效果
optics_model = OPTICS(min_samples=10, xi=0.1, min_cluster_size=0.02, max_eps=0.5)
optics_model.fit(X)

# 3. 提取聚类结果
# labels_: 每个样本的聚类标签，-1表示噪声点
labels = optics_model.labels_

# 4. 创建一个大图，包含多个子图
plt.figure(figsize=(15, 15))

# 4.1 绘制可达距离图（顶部子图）
plt.subplot(3, 1, 1)
# 获取可达距离和排序索引
reachability = optics_model.reachability_
ordering = optics_model.ordering_

# 绘制可达距离图
plt.plot(reachability[ordering], 'k-', alpha=0.7)
plt.ylabel('Reachability Distance')
plt.title('OPTICS Reachability Plot')

# 标记不同的聚类
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:  # 噪声点
        continue
    
    # 获取当前聚类在排序后的索引位置
    cluster_points = np.where(labels[ordering] == k)[0]
    
    # 标记聚类区域
    plt.axvspan(min(cluster_points), max(cluster_points), 
                alpha=0.2, color=col, label=f'Cluster {k}')

plt.legend()
plt.grid(True)

# 添加参数说明
plt.text(0.02, 0.95, 
         f"Parameters:\nmin_samples={optics_model.min_samples}\nxi={optics_model.xi}\n"
         f"min_cluster_size={optics_model.min_cluster_size}\nmax_eps={optics_model.max_eps}",
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

# 添加可达距离图解释
plt.text(0.7, 0.95, 
         "Reachability Plot Guide:\n"
         "- Valleys indicate clusters\n"
         "- Steep slopes indicate cluster boundaries\n"
         "- Flat high areas indicate noise",
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

# 4.2 绘制OPTICS聚类结果（中间子图）
plt.subplot(3, 1, 2)
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
                c=[tuple(col)], edgecolor='k', s=50, label=f'Cluster {k}')

plt.title('OPTICS Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# 4.3 使用DBSCAN进行比较 - 进一步优化参数
plt.subplot(3, 1, 3)
# 使用优化后的参数
# eps: 邻域半径，较小的值会创建更多聚类
# min_samples: 核心点的最小邻域样本数
dbscan = DBSCAN(eps=0.2, min_samples=10)
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
                c=[tuple(col)], edgecolor='k', s=150, label=f'Cluster {k} Core')
    # 绘制非核心点（小标记）
    plt.scatter(X[class_mask & ~core_samples_mask, 0], X[class_mask & ~core_samples_mask, 1],
                c=[tuple(col)], edgecolor='k', s=50)

plt.title('DBSCAN Clustering Result (for comparison)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('density_clustering.png')
# 不使用plt.show()，避免非交互式后端的警告

# 5. 输出聚类统计信息
n_clusters_optics = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_optics = list(labels).count(-1)
print(f'OPTICS estimated number of clusters: {n_clusters_optics}')
print(f'OPTICS estimated number of noise points: {n_noise_optics}')

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_dbscan = list(dbscan_labels).count(-1)
print(f'DBSCAN estimated number of clusters: {n_clusters_dbscan}')
print(f'DBSCAN estimated number of noise points: {n_noise_dbscan}')

# 6. 打印参数设置，便于调整
print("\nParameters used:")
print(f"OPTICS: min_samples={optics_model.min_samples}, xi={optics_model.xi}, min_cluster_size={optics_model.min_cluster_size}, max_eps={optics_model.max_eps}")
print(f"DBSCAN: eps={dbscan.eps}, min_samples={dbscan.min_samples}")

# 7. 参数调整指南
print("\nOPTICS Parameter Tuning Guide:")
print("1. min_samples: 控制点被视为核心点所需的邻居数")
print("   - 增大: 减少噪声敏感性，但可能导致较小的聚类被忽略")
print("   - 减小: 可以检测较小的聚类，但可能增加噪声敏感性")
print("   - 建议范围: 数据集大小的1%-5%")
print("\n2. xi: 控制聚类提取的陡度阈值")
print("   - 增大: 提取更少的聚类")
print("   - 减小: 提取更多的聚类")
print("   - 建议范围: 0.01-0.1")
print("\n3. min_cluster_size: 控制被视为聚类的最小样本比例")
print("   - 增大: 忽略较小的聚类")
print("   - 减小: 允许较小的聚类")
print("   - 建议范围: 0.01-0.05")
print("\n4. max_eps: 控制最大可达距离")
print("   - 增大: 可能连接更多的点，减少噪声")
print("   - 减小: 可能增加噪声点数量，但聚类更紧凑")
print("   - 建议: 如果不确定，可以设置为无穷大(np.inf)")
print("\n可达距离图解读:")
print("- 图中的'山谷'表示潜在的聚类")
print("- 陡峭的'山坡'表示聚类边界")
print("- 高处的'平台'通常表示噪声点")
print("- 调整参数时，观察可达距离图的变化，寻找清晰的'山谷'结构")