from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

# 0. 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
    print(f"聚类数k={k}, DBI={dbi_score:.3f}, 轮廓系数={silhouette:.3f}, CH指数={calinski:.3f}")

# 4.2 绘制不同k值的DBI变化曲线
plt.plot(k_range, dbi_scores, 'o-', color='blue')
plt.xlabel('聚类数 (k)')
plt.ylabel('Davies-Bouldin指数')
plt.title('不同聚类数的Davies-Bouldin指数\n(值越小越好)')
plt.grid(True)

# 4.3 绘制轮廓系数变化曲线（右上子图）
plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-', color='green')
plt.xlabel('聚类数 (k)')
plt.ylabel('轮廓系数')
plt.title('不同聚类数的轮廓系数\n(值越大越好)')
plt.grid(True)

# 4.4 绘制Calinski-Harabasz指数变化曲线（左下子图）
plt.subplot(2, 2, 3)
plt.plot(k_range, ch_scores, 'o-', color='red')
plt.xlabel('聚类数 (k)')
plt.ylabel('Calinski-Harabasz指数')
plt.title('不同聚类数的Calinski-Harabasz指数\n(值越大越好)')
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
plt.title(f'K-Means聚类 (k=3)\nDBI={davies_bouldin_score(X, labels):.3f}')
plt.grid(True)

# 6. 调整布局并保存图像
plt.tight_layout()
plt.savefig('cluster_evaluation.png')
plt.show()

# 7. 输出DBI指标的详细解释
print("\nDavies-Bouldin指数 (DBI) 解释:")
print("- DBI衡量的是聚类的'紧密度'与'分离度'的比值")
print("- 值越小越好，表示聚类内部紧密，聚类之间分离良好")
print("- 它计算每对聚类中心之间的相似度，然后取所有聚类对的最大值的平均值")
print("- 计算公式: DBI = (1/n) * Σ max_j≠i ((Si + Sj) / Mij)")
print("  其中Si是聚类i内部的平均距离，Mij是聚类i和j中心之间的距离")