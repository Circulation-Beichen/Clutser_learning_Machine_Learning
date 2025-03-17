import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import argparse

def generate_dataset(dataset_type='moons', n_samples=500, noise=0.05, random_state=42):
    """生成不同类型的数据集"""
    if dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'blobs':
        X, _ = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    # 标准化数据
    X = StandardScaler().fit_transform(X)
    return X

def run_optics(X, min_samples=5, xi=0.01, min_cluster_size=0.05, max_eps=np.inf):
    """运行OPTICS算法并返回结果"""
    optics_model = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        max_eps=max_eps
    )
    optics_model.fit(X)
    return optics_model

def plot_results(X, optics_model, output_file=None):
    """绘制OPTICS结果，包括可达距离图和聚类结果"""
    # 获取结果
    labels = optics_model.labels_
    reachability = optics_model.reachability_
    ordering = optics_model.ordering_
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 1. 绘制可达距离图
    ax1.plot(reachability[ordering], 'k-', alpha=0.7)
    ax1.set_ylabel('Reachability Distance')
    ax1.set_title('OPTICS Reachability Plot')
    
    # 标记不同的聚类
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:  # 噪声点
            continue
        
        # 获取当前聚类在排序中的点
        cluster_points = np.where(labels[ordering] == k)[0]
        
        # 标记聚类区域
        ax1.axvspan(min(cluster_points), max(cluster_points), 
                   alpha=0.2, color=col, label=f'Cluster {k}')
    
    ax1.legend()
    ax1.grid(True)
    
    # 添加参数描述
    ax1.text(0.02, 0.95, 
             f"Parameters:\nmin_samples={optics_model.min_samples}\nxi={optics_model.xi}\n"
             f"min_cluster_size={optics_model.min_cluster_size}\nmax_eps={optics_model.max_eps}",
             transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # 添加可达距离图解释
    ax1.text(0.7, 0.95, 
             "Reachability Plot Interpretation:\n"
             "- Valleys indicate clusters\n"
             "- Steep slopes indicate boundaries\n"
             "- High plateaus indicate noise",
             transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # 2. 绘制聚类结果
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:  # 噪声点用黑色表示
            col = [0, 0, 0, 1]
        # 创建当前聚类的掩码
        class_mask = (labels == k)
        
        # 绘制聚类点
        ax2.scatter(X[class_mask, 0], X[class_mask, 1],
                   c=[tuple(col)], edgecolor='k', s=50, label=f'Cluster {k}')
    
    ax2.set_title('OPTICS Clustering Result')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True)
    
    # 统计信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ax2.text(0.02, 0.95, 
             f"Statistics:\nClusters: {n_clusters}\nNoise points: {n_noise}\nNoise ratio: {n_noise/len(labels):.2%}",
             transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图像
    if output_file:
        plt.savefig(output_file)
        print(f"结果已保存到 {output_file}")
    
    # 返回统计信息
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise/len(labels)
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OPTICS 聚类参数调整工具')
    parser.add_argument('--dataset', type=str, default='moons', choices=['moons', 'blobs', 'circles'],
                        help='数据集类型: moons, blobs, circles (默认: moons)')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='样本数量 (默认: 500)')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='噪声水平 (默认: 0.05)')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='OPTICS min_samples参数 (默认: 5)')
    parser.add_argument('--xi', type=float, default=0.01,
                        help='OPTICS xi参数 (默认: 0.01)')
    parser.add_argument('--min_cluster_size', type=float, default=0.05,
                        help='OPTICS min_cluster_size参数 (默认: 0.05)')
    parser.add_argument('--max_eps', type=float, default=np.inf,
                        help='OPTICS max_eps参数 (默认: np.inf)')
    parser.add_argument('--output', type=str, default='optics_tuning_result.png',
                        help='输出图像文件名 (默认: optics_tuning_result.png)')
    
    args = parser.parse_args()
    
    # 生成数据集
    X = generate_dataset(
        dataset_type=args.dataset,
        n_samples=args.n_samples,
        noise=args.noise
    )
    
    # 运行OPTICS
    optics_model = run_optics(
        X,
        min_samples=args.min_samples,
        xi=args.xi,
        min_cluster_size=args.min_cluster_size,
        max_eps=args.max_eps
    )
    
    # 绘制结果
    stats = plot_results(X, optics_model, args.output)
    
    # 打印参数调整指南
    print("\nOPTICS 参数调整指南:")
    print("1. min_samples: 控制点被视为核心点所需的邻居数")
    print(f"   - 当前值: {args.min_samples}")
    print("   - 增大: 减少噪声敏感性，但可能忽略较小的聚类")
    print("   - 减小: 可以检测较小的聚类，但可能增加噪声敏感性")
    print("   - 建议范围: 数据集大小的1%-5%")
    
    print("\n2. xi: 控制聚类提取的陡度阈值")
    print(f"   - 当前值: {args.xi}")
    print("   - 增大: 提取更少的聚类")
    print("   - 减小: 提取更多的聚类")
    print("   - 建议范围: 0.01-0.1")
    
    print("\n3. min_cluster_size: 控制被视为聚类的最小样本比例")
    print(f"   - 当前值: {args.min_cluster_size}")
    print("   - 增大: 忽略较小的聚类")
    print("   - 减小: 允许较小的聚类")
    print("   - 建议范围: 0.01-0.05")
    
    print("\n4. max_eps: 控制最大可达距离")
    print(f"   - 当前值: {args.max_eps}")
    print("   - 增大: 可能连接更多的点，减少噪声")
    print("   - 减小: 可能增加噪声点数量，但聚类更紧凑")
    print("   - 建议: 如果不确定，设置为无穷大(np.inf)")
    
    print("\n聚类结果统计:")
    print(f"- 聚类数量: {stats['n_clusters']}")
    print(f"- 噪声点数量: {stats['n_noise']}")
    print(f"- 噪声比例: {stats['noise_ratio']:.2%}")
    
    print("\n尝试不同参数的示例命令:")
    print(f"python optics_parameter_tuning.py --min_samples=15 --xi=0.03 --min_cluster_size=0.02")
    print(f"python optics_parameter_tuning.py --dataset=circles --noise=0.1 --min_samples=5")

if __name__ == "__main__":
    main() 