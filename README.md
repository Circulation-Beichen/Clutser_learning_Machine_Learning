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

3. **OPTICS参数调整工具**
   - 文件: `optics_parameter_tuning.py`
   - 功能: 提供命令行接口，方便尝试不同的OPTICS参数组合
   - 输出: 生成`optics_tuning_result.png`图像，展示可达距离图和聚类结果

4. **OPTICS最佳参数示例**
   - 文件: `optics_best_params.py`
   - 功能: 展示不同数据集上OPTICS的最佳参数设置
   - 输出: 生成`optics_best_results.png`图像，展示三种不同数据集的聚类结果

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

## OPTICS参数调整指南

### 可达距离图(Reachability Plot)解读
- 图中的"山谷"表示潜在的聚类
- 陡峭的"山坡"表示聚类边界
- 高处的"平台"通常表示噪声点
- 调整参数时，应寻找清晰的"山谷"结构

### 关键参数
1. **min_samples**
   - 控制点被视为核心点所需的邻居数
   - 增大: 减少噪声敏感性，但可能导致较小的聚类被忽略
   - 减小: 可以检测较小的聚类，但可能增加噪声敏感性
   - 建议范围: 数据集大小的1%-5%

2. **xi**
   - 控制聚类提取的陡度阈值
   - 增大: 提取更少的聚类
   - 减小: 提取更多的聚类
   - 建议范围: 0.01-0.1

3. **min_cluster_size**
   - 控制被视为聚类的最小样本比例
   - 增大: 忽略较小的聚类
   - 减小: 允许较小的聚类
   - 建议范围: 0.01-0.05

4. **max_eps**
   - 控制最大可达距离
   - 增大: 可能连接更多的点，减少噪声
   - 减小: 可能增加噪声点数量，但聚类更紧凑
   - 建议: 如果不确定，可以设置为无穷大(np.inf)

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

4. **使用OPTICS参数调整工具**
   ```bash
   # 基本用法
   python optics_parameter_tuning.py
   
   # 指定参数
   python optics_parameter_tuning.py --min_samples=15 --xi=0.03 --min_cluster_size=0.02
   
   # 使用不同数据集
   python optics_parameter_tuning.py --dataset=circles --noise=0.1
   ```

5. **查看不同数据集的最佳参数**
   ```bash
   python optics_best_params.py
   ```

## 参考资料

- [scikit-learn文档: 聚类](https://scikit-learn.org/stable/modules/clustering.html)
- [Davies-Bouldin指数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
- [OPTICS算法](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
- [DBSCAN算法](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [可达距离图解读](https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html)


- 为什么OPTICS效果不如DBSCAN及如何改进
   - 根据我们的实验结果，我可以分析为什么OPTICS在某些情况下效果不如DBSCAN，以及如何使OPTICS的效果更接近DBSCAN。
   - 为什么OPTICS效果不如DBSCAN
- 聚类提取方法的差异：
   - 默认情况下，OPTICS使用基于xi的提取方法，它寻找可达距离图中的"陡峭下降"区域
   - DBSCAN直接基于距离阈值eps进行聚类，这在某些数据集上更直观
- 参数敏感性：
   - OPTICS的xi参数对聚类结果影响很大。当xi=0.01时，我们得到了21个聚类和203个噪声点；当xi=0.1时，得到了14个聚类和300个噪声点
   - 而DBSCAN只需要调整eps和min_samples两个参数，在月牙形数据集上表现稳定（2个聚类，0个噪声点）
- 噪声点处理：
   - 我们看到OPTICS倾向于将更多的点标记为噪声（使用xi方法时）
   - DBSCAN在eps选择合适的情况下，能更好地包含所有点
- 聚类数量：
   - 使用xi方法的OPTICS倾向于产生过多的小聚类，而不是几个大的、有意义的聚类
   - DBSCAN在月牙形数据集上正确识别出了2个聚类