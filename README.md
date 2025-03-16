# 聚类算法学习项目
聚类方法学习——机器学习

这个项目包含了多种聚类算法的实现和评估方法，用于学习和比较不同聚类技术的效果。

## 项目内容

1. **K-Means聚类与Davies-Bouldin指数(DBI)评估**
   - 文件: `cluster_DBI.py`
   - 功能: 使用K-Means算法进行聚类，并通过DBI、轮廓系数和Calinski-Harabasz指数评估不同K值的聚类效果
   - 输出: 生成`cluster_evaluation.png`图像，展示不同评估指标随K值变化的曲线和最佳聚类结果

2. **基于密度的聚类算法(OPTICS和DBSCAN)**
   - 文件: `cluster_OPTICS.py`
   - 功能: 使用OPTICS和DBSCAN算法对非球形数据进行聚类，展示密度聚类的效果
   - 输出: 生成`density_clustering.png`图像，对比两种密度聚类算法的结果

## 聚类评估指标

### Davies-Bouldin指数(DBI)
- 衡量聚类的"紧密度"与"分离度"的比值
- 值越小越好，表示聚类内部紧密，聚类之间分离良好
- 计算公式: DBI = (1/n) * Σ max_j≠i ((Si + Sj) / Mij)
  - Si是聚类i内部的平均距离
  - Mij是聚类i和j中心之间的距离

### 轮廓系数(Silhouette Score)
- 衡量样本与自己所在聚类的相似度与其他聚类的相似度之比
- 值越大越好，范围在[-1, 1]之间
- 适合评估各种形状的聚类

### Calinski-Harabasz指数(CH指数)
- 也称为方差比准则(Variance Ratio Criterion)
- 计算聚类间离散度与聚类内离散度的比值
- 值越大越好
- 对球形聚类效果较好

## 使用方法

1. **环境设置**
   ```bash
   # 创建并激活虚拟环境
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   
   # 安装依赖
   pip install scikit-learn matplotlib numpy
   ```

2. **运行K-Means聚类与DBI评估**
   ```bash
   python cluster_DBI.py
   ```

3. **运行密度聚类算法**
   ```bash
   python cluster_OPTICS.py
   ```

## 结果解读

1. **K-Means聚类结果**
   - DBI值最小的K值通常是最佳聚类数
   - 不同评估指标可能给出不同的最佳聚类数
   - 在实际应用中，应结合业务需求和数据可视化结果来确定最终的聚类数

2. **密度聚类结果**
   - OPTICS和DBSCAN适合发现非球形、不规则形状的聚类
   - 这些算法能够自动识别噪声点
   - 参数设置（如MinPts和eps）对结果有显著影响

## 参考资料

- [scikit-learn文档: 聚类](https://scikit-learn.org/stable/modules/clustering.html)
- [Davies-Bouldin指数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
- [OPTICS算法](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
- [DBSCAN算法](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
