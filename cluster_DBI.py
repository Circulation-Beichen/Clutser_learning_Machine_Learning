from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 生成示例数据
X, _ = make_blobs(n_samples=500, centers=3, random_state=42)

# 1. 使用 K-Means 聚类并计算DBI
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
labels = kmeans.labels_
dbi = davies_bouldin_score(X, labels)
print(f"DBI = {dbi:.3f}")  # 值越小越好

# 2. 比较不同K值的DBI
k_range = range(2, 10)
dbi_scores = []
silhouette_scores = []
ch_scores = []

plt.figure(figsize=(15, 10))

# 创建子图布局
plt.subplot(2, 2, 1)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    
    # 计算各种评估指标
    dbi_score = davies_bouldin_score(X, labels)
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    
    dbi_scores.append(dbi_score)
    silhouette_scores.append(silhouette)
    ch_scores.append(calinski)
    
    print(f"聚类数k={k}, DBI={dbi_score:.3f}, 轮廓系数={silhouette:.3f}, CH指数={calinski:.3f}")

# 绘制不同k值的DBI变化曲线
plt.plot(k_range, dbi_scores, 'o-', color='blue')
plt.xlabel('聚类数 (k)')
plt.ylabel('Davies-Bouldin指数')
plt.title('不同聚类数的Davies-Bouldin指数\n(值越小越好)')
plt.grid(True)

# 3. 绘制轮廓系数变化曲线
plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-', color='green')
plt.xlabel('聚类数 (k)')
plt.ylabel('轮廓系数')
plt.title('不同聚类数的轮廓系数\n(值越大越好)')
plt.grid(True)

# 4. 绘制Calinski-Harabasz指数变化曲线
plt.subplot(2, 2, 3)
plt.plot(k_range, ch_scores, 'o-', color='red')
plt.xlabel('聚类数 (k)')
plt.ylabel('Calinski-Harabasz指数')
plt.title('不同聚类数的Calinski-Harabasz指数\n(值越大越好)')
plt.grid(True)

# 5. 可视化最佳聚类结果 (k=3)
plt.subplot(2, 2, 4)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
labels = kmeans.labels_

# 为每个聚类分配不同的颜色
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='X', s=200)
plt.title(f'K-Means聚类 (k=3)\nDBI={davies_bouldin_score(X, labels):.3f}')
plt.grid(True)

plt.tight_layout()
plt.savefig('cluster_evaluation.png')
plt.show()

# 6. 解释DBI指标
print("\nDavies-Bouldin指数 (DBI) 解释:")
print("- DBI衡量的是聚类的'紧密度'与'分离度'的比值")
print("- 值越小越好，表示聚类内部紧密，聚类之间分离良好")
print("- 它计算每对聚类中心之间的相似度，然后取所有聚类对的最大值的平均值")