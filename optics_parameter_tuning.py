import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import argparse

def generate_dataset(dataset_type='moons', n_samples=500, noise=0.05, random_state=42):
    """Generate different types of datasets"""
    if dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'blobs':
        X, _ = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Standardize data
    X = StandardScaler().fit_transform(X)
    return X

def run_optics(X, min_samples=10, xi=0.05, min_cluster_size=0.05, max_eps=1.0):
    """Run OPTICS algorithm and return results"""
    optics_model = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        max_eps=max_eps
    )
    optics_model.fit(X)
    return optics_model

def plot_results(X, optics_model, output_file=None):
    """Plot OPTICS results, including reachability plot and clustering results"""
    # Get results
    labels = optics_model.labels_
    reachability = optics_model.reachability_
    ordering = optics_model.ordering_
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 1. Plot reachability distance
    ax1.plot(reachability[ordering], 'k-', alpha=0.7)
    ax1.set_ylabel('Reachability Distance')
    ax1.set_title('OPTICS Reachability Plot')
    
    # Mark different clusters
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:  # Noise points
            continue
        
        # Get current cluster points in ordering
        cluster_points = np.where(labels[ordering] == k)[0]
        
        # Mark cluster regions
        ax1.axvspan(min(cluster_points), max(cluster_points), 
                   alpha=0.2, color=col, label=f'Cluster {k}')
    
    ax1.legend()
    ax1.grid(True)
    
    # Add parameter description
    ax1.text(0.02, 0.95, 
             f"Parameters:\nmin_samples={optics_model.min_samples}\nxi={optics_model.xi}\n"
             f"min_cluster_size={optics_model.min_cluster_size}\nmax_eps={optics_model.max_eps}",
             transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # Add reachability plot explanation
    ax1.text(0.7, 0.95, 
             "Reachability Plot Guide:\n"
             "- Valleys indicate clusters\n"
             "- Steep slopes indicate boundaries\n"
             "- High plateaus indicate noise",
             transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # 2. Plot clustering results
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:  # Noise points in black
            col = [0, 0, 0, 1]
        # Create mask for current cluster
        class_mask = (labels == k)
        
        # Plot cluster points
        ax2.scatter(X[class_mask, 0], X[class_mask, 1],
                   c=[tuple(col)], edgecolor='k', s=50, label=f'Cluster {k}')
    
    ax2.set_title('OPTICS Clustering Result')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True)
    
    # Statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ax2.text(0.02, 0.95, 
             f"Statistics:\nClusters: {n_clusters}\nNoise points: {n_noise}\nNoise ratio: {n_noise/len(labels):.2%}",
             transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save image
    if output_file:
        plt.savefig(output_file)
        print(f"Results saved to {output_file}")
    
    # Return statistics
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise/len(labels)
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OPTICS Clustering Parameter Tuning Tool')
    parser.add_argument('--dataset', type=str, default='moons', choices=['moons', 'blobs', 'circles'],
                        help='Dataset type: moons, blobs, circles (default: moons)')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of samples (default: 500)')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Noise level (default: 0.05)')
    parser.add_argument('--min_samples', type=int, default=10,
                        help='OPTICS min_samples parameter (default: 10)')
    parser.add_argument('--xi', type=float, default=0.05,
                        help='OPTICS xi parameter (default: 0.05)')
    parser.add_argument('--min_cluster_size', type=float, default=0.05,
                        help='OPTICS min_cluster_size parameter (default: 0.05)')
    parser.add_argument('--max_eps', type=float, default=1.0,
                        help='OPTICS max_eps parameter (default: 1.0)')
    parser.add_argument('--output', type=str, default='optics_tuning_result.png',
                        help='Output image filename (default: optics_tuning_result.png)')
    
    args = parser.parse_args()
    
    # Generate dataset
    X = generate_dataset(
        dataset_type=args.dataset,
        n_samples=args.n_samples,
        noise=args.noise
    )
    
    # Run OPTICS
    optics_model = run_optics(
        X,
        min_samples=args.min_samples,
        xi=args.xi,
        min_cluster_size=args.min_cluster_size,
        max_eps=args.max_eps
    )
    
    # Plot results
    stats = plot_results(X, optics_model, args.output)
    
    # Print parameter tuning guide
    print("\nOPTICS Parameter Tuning Guide:")
    print("1. min_samples: Controls the number of neighbors required for a point to be a core point")
    print(f"   - Current value: {args.min_samples}")
    print("   - Increase: Reduces noise sensitivity, but may ignore smaller clusters")
    print("   - Decrease: Can detect smaller clusters, but may increase noise sensitivity")
    print("   - Recommended range: 1%-5% of dataset size")
    
    print("\n2. xi: Controls the steepness threshold for cluster extraction")
    print(f"   - Current value: {args.xi}")
    print("   - Increase: Extracts fewer clusters")
    print("   - Decrease: Extracts more clusters")
    print("   - Recommended range: 0.01-0.1")
    
    print("\n3. min_cluster_size: Controls the minimum sample ratio to be considered a cluster")
    print(f"   - Current value: {args.min_cluster_size}")
    print("   - Increase: Ignores smaller clusters")
    print("   - Decrease: Allows smaller clusters")
    print("   - Recommended range: 0.01-0.05")
    
    print("\n4. max_eps: Controls the maximum reachability distance")
    print(f"   - Current value: {args.max_eps}")
    print("   - Increase: May connect more points, reducing noise")
    print("   - Decrease: May increase noise points, but clusters are more compact")
    print("   - Recommendation: If unsure, set to infinity (np.inf)")
    
    print("\nClustering Result Statistics:")
    print(f"- Number of clusters: {stats['n_clusters']}")
    print(f"- Number of noise points: {stats['n_noise']}")
    print(f"- Noise ratio: {stats['noise_ratio']:.2%}")
    
    print("\nExample commands to try different parameters:")
    print(f"python optics_parameter_tuning.py --min_samples=15 --xi=0.03 --min_cluster_size=0.02")
    print(f"python optics_parameter_tuning.py --dataset=circles --noise=0.1 --min_samples=20")

if __name__ == "__main__":
    main() 