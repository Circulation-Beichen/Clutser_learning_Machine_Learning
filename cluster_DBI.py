from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 0. 配置matplotlib支持中文显示
# 尝试使用不同的中文字体，避免字体缺失警告
try:
    # 尝试使用微软雅黑字体（Windows系统常见字体）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    # 测试中文显示
    mpl.use('Agg')  # 使用非交互式后端，避免一些显示问题
except:
    # 如果设置失败，使用英文标签
    print("警告: 中文字体设置失败，将使用英文标签")

# 1. 数据准备
# make_blobs函数生成用于聚类的多维高斯分布数据
# n_samples: 样本总数
# centers: 聚类中心数量
# random_state: 随机种子，确保结果可重复
X, _ = make_blobs(n_samples=500, centers=3, random_state=42)

# 2. K-Means聚类
# 使用K-Means算法将数据分为3个簇
# n_clusters: 指定聚类数量
# random_state: 随机种子，确保结果可重复
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
# 获取每个数据点的聚类标签
labels = kmeans.labels_

# 3. 计算Davies-Bouldin指数(DBI)
# DBI是一种内部评估指标，用于评估聚类的质量
# 它衡量的是"聚类内部的紧密度"与"聚类之间的分离度"的比值
# 值越小越好，表示聚类效果越好
dbi = davies_bouldin_score(X, labels)
print(f"DBI = {dbi:.3f}")  # 值越小越好

# 4. 比较不同K值的聚类效果
# 测试从2到9的不同聚类数
k_range = range(2, 10)
dbi_scores = []  # 存储不同k值的DBI
silhouette_scores = []  # 存储不同k值的轮廓系数
ch_scores = []  # 存储不同k值的Calinski-Harabasz指数

# 创建一个大图，包含多个子图
plt.figure(figsize=(15, 10))

# 4.1 创建第一个子图（左上）用于DBI曲线
plt.subplot(2, 2, 1)

# 对每个k值进行聚类并计算评估指标
for k in k_range:
    # 使用k个聚类中心进行K-Means聚类
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    
    # 计算各种评估指标
    # Davies-Bouldin指数: 值越小越好
    dbi_score = davies_bouldin_score(X, labels)
    # 轮廓系数: 值越大越好，范围在[-1, 1]之间
    silhouette = silhouette_score(X, labels)
    # Calinski-Harabasz指数: 值越大越好
    calinski = calinski_harabasz_score(X, labels)
    
    # 存储计算结果
    dbi_scores.append(dbi_score)
    silhouette_scores.append(silhouette)
    ch_scores.append(calinski)
    
    # 打印每个k值的评估结果
    print(f"Clusters k={k}, DBI={dbi_score:.3f}, Silhouette={silhouette:.3f}, CH={calinski:.3f}")

# 4.2 绘制不同k值的DBI变化曲线
plt.plot(k_range, dbi_scores, 'o-', color='blue')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index for Different k\n(Lower is better)')
plt.grid(True)

# 4.3 绘制轮廓系数变化曲线（右上子图）
plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-', color='green')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k\n(Higher is better)')
plt.grid(True)

# 4.4 绘制Calinski-Harabasz指数变化曲线（左下子图）
plt.subplot(2, 2, 3)
plt.plot(k_range, ch_scores, 'o-', color='red')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Index')
plt.title('Calinski-Harabasz Index for Different k\n(Higher is better)')
plt.grid(True)

# 5. 可视化最佳聚类结果 (k=3)（右下子图）
plt.subplot(2, 2, 4)
# 使用k=3进行K-Means聚类
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
labels = kmeans.labels_

# 5.1 绘制聚类结果
# 为每个数据点根据其聚类标签分配颜色
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
# 绘制聚类中心点（用红色X标记）
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='X', s=200)
plt.title(f'K-Means Clustering (k=3)\nDBI={davies_bouldin_score(X, labels):.3f}')
plt.grid(True)

# 6. 调整布局并保存图像
plt.tight_layout()
plt.savefig('cluster_evaluation.png')
# 不使用plt.show()，避免非交互式后端的警告

# 7. 输出DBI指标的详细解释
print("\nDavies-Bouldin Index (DBI) Explanation:")
print("- DBI measures the ratio of 'within-cluster scatter' to 'between-cluster separation'")
print("- Lower values are better, indicating compact clusters that are well-separated")
print("- It calculates the similarity between each pair of clusters and takes the average of the maximum values")
print("- Formula: DBI = (1/n) * Σ max_j≠i ((Si + Sj) / Mij)")
print("  where Si is the average distance within cluster i, Mij is the distance between cluster centers i and j")