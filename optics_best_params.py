import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

# 生成合成数据集
def generate_datasets(n_samples=500, noise=0.05, random_state=42):
    """生成不同类型的数据集用于聚类演示"""
    # 半月形
    moons_X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    moons_X = StandardScaler().fit_transform(moons_X)
    
    # 高斯分布
    blobs_X, _ = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    blobs_X = StandardScaler().fit_transform(blobs_X)
    
    # 同心圆
    circles_X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    circles_X = StandardScaler().fit_transform(circles_X)
    
    return {
        'moons': moons_X,
        'blobs': blobs_X,
        'circles': circles_X
    }

# 不同数据集的最佳参数
best_params = {
    'moons': {
        'min_samples': 5,
        'xi': 0.01,
        'min_cluster_size': 0.03,
        'max_eps': np.inf
    },
    'blobs': {
        'min_samples': 5,
        'xi': 0.01,
        'min_cluster_size': 0.05,
        'max_eps': np.inf
    },
    'circles': {
        'min_samples': 5,
        'xi': 0.01,
        'min_cluster_size': 0.02,
        'max_eps': np.inf
    }
}

# 使用最佳参数运行OPTICS
def run_optics_with_best_params(datasets):
    """使用每个数据集的最佳参数运行OPTICS"""
    results = {}
    
    for dataset_name, X in datasets.items():
        params = best_params[dataset_name]
        
        # 使用最佳参数创建OPTICS模型
        optics_model = OPTICS(
            min_samples=params['min_samples'],
            xi=params['xi'],
            min_cluster_size=params['min_cluster_size'],
            max_eps=params['max_eps']
        )
        
        # 拟合模型
        optics_model.fit(X)
        
        # 存储结果
        results[dataset_name] = {
            'model': optics_model,
            'X': X,
            'labels': optics_model.labels_,
            'reachability': optics_model.reachability_,
            'ordering': optics_model.ordering_,
            'params': params
        }
    
    return results

# 可视化结果
def visualize_results(results):
    """可视化所有数据集的OPTICS聚类结果"""
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 5 * len(results)))
    
    for i, (dataset_name, result) in enumerate(results.items()):
        X = result['X']
        labels = result['labels']
        reachability = result['reachability']
        ordering = result['ordering']
        params = result['params']
        
        # 绘制可达距离图
        ax1 = axes[i, 0]
        ax1.plot(reachability[ordering], 'k-', alpha=0.7)
        ax1.set_ylabel('Reachability Distance')
        ax1.set_title(f'OPTICS Reachability Plot - {dataset_name.capitalize()}')
        
        # 在可达距离图中标记聚类
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:  # 跳过噪声
                continue
            
            # 获取排序中的聚类点
            cluster_points = np.where(labels[ordering] == k)[0]
            
            # 标记聚类区域
            if len(cluster_points) > 0:  # 只有当聚类有点时
                ax1.axvspan(min(cluster_points), max(cluster_points), 
                           alpha=0.2, color=col, label=f'Cluster {k}')
        
        ax1.legend()
        ax1.grid(True)
        
        # 添加参数信息
        ax1.text(0.02, 0.95, 
                 f"Parameters:\nmin_samples={params['min_samples']}\n"
                 f"xi={params['xi']}\nmin_cluster_size={params['min_cluster_size']}\n"
                 f"max_eps={params['max_eps']}",
                 transform=ax1.transAxes, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # 绘制聚类结果
        ax2 = axes[i, 1]
        
        # 绘制每个聚类
        for k, col in zip(unique_labels, colors):
            if k == -1:  # 噪声点用黑色表示
                col = [0, 0, 0, 1]
            
            # 创建当前聚类的掩码
            class_mask = (labels == k)
            
            # 绘制聚类点
            ax2.scatter(X[class_mask, 0], X[class_mask, 1],
                       c=[tuple(col)], edgecolor='k', s=50, label=f'Cluster {k}')
        
        ax2.set_title(f'OPTICS Clustering - {dataset_name.capitalize()}')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.legend()
        ax2.grid(True)
        
        # 添加统计信息
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        ax2.text(0.02, 0.95, 
                 f"Statistics:\nClusters: {n_clusters}\n"
                 f"Noise points: {n_noise}\n"
                 f"Noise ratio: {n_noise/len(labels):.2%}",
                 transform=ax2.transAxes, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('optics_best_results.png')
    print("结果已保存到 optics_best_results.png")

# 主函数
def main():
    # 生成数据集
    datasets = generate_datasets(n_samples=500, noise=0.05)
    
    # 使用最佳参数运行OPTICS
    results = run_optics_with_best_params(datasets)
    
    # 可视化结果
    visualize_results(results)
    
    # 打印参数调整指南
    print("\nOPTICS 参数调整指南:")
    print("1. min_samples:")
    print("   - 控制点被视为核心点所需的邻居数")
    print("   - 增大: 减少噪声敏感性，但可能忽略较小的聚类")
    print("   - 减小: 可以检测较小的聚类，但可能增加噪声敏感性")
    print("   - 建议范围: 数据集大小的1%-5%")
    
    print("\n2. xi:")
    print("   - 控制聚类提取的陡度阈值")
    print("   - 增大: 提取更少的聚类")
    print("   - 减小: 提取更多的聚类")
    print("   - 建议范围: 0.01-0.1")
    
    print("\n3. min_cluster_size:")
    print("   - 控制被视为聚类的最小样本比例")
    print("   - 增大: 忽略较小的聚类")
    print("   - 减小: 允许较小的聚类")
    print("   - 建议范围: 0.01-0.05")
    
    print("\n4. max_eps:")
    print("   - 控制最大可达距离")
    print("   - 增大: 可能连接更多的点，减少噪声")
    print("   - 减小: 可能增加噪声点数量，但聚类更紧凑")
    print("   - 建议: 如果不确定，设置为无穷大(np.inf)")
    
    print("\n可达距离图解读:")
    print("- 图中的'山谷'表示潜在的聚类")
    print("- 陡峭的'山坡'表示聚类边界")
    print("- 高处的'平台'通常表示噪声点")
    print("- 调整参数时，寻找清晰的'山谷'结构")

if __name__ == "__main__":
    main() 