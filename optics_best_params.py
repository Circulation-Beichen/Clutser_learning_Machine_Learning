import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

# Generate synthetic datasets
def generate_datasets(n_samples=500, noise=0.05, random_state=42):
    """Generate different types of datasets for clustering demonstration"""
    # Half-moon shapes
    moons_X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    moons_X = StandardScaler().fit_transform(moons_X)
    
    # Gaussian blobs
    blobs_X, _ = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    blobs_X = StandardScaler().fit_transform(blobs_X)
    
    # Concentric circles
    circles_X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    circles_X = StandardScaler().fit_transform(circles_X)
    
    return {
        'moons': moons_X,
        'blobs': blobs_X,
        'circles': circles_X
    }

# Best parameters for different datasets
best_params = {
    'moons': {
        'min_samples': 15,
        'xi': 0.02,
        'min_cluster_size': 0.03,
        'max_eps': 0.5
    },
    'blobs': {
        'min_samples': 10,
        'xi': 0.05,
        'min_cluster_size': 0.05,
        'max_eps': 1.0
    },
    'circles': {
        'min_samples': 10,
        'xi': 0.02,
        'min_cluster_size': 0.02,
        'max_eps': 0.5
    }
}

# Run OPTICS with best parameters
def run_optics_with_best_params(datasets):
    """Run OPTICS with best parameters for each dataset"""
    results = {}
    
    for dataset_name, X in datasets.items():
        params = best_params[dataset_name]
        
        # Create OPTICS model with best parameters
        optics_model = OPTICS(
            min_samples=params['min_samples'],
            xi=params['xi'],
            min_cluster_size=params['min_cluster_size'],
            max_eps=params['max_eps']
        )
        
        # Fit model
        optics_model.fit(X)
        
        # Store results
        results[dataset_name] = {
            'model': optics_model,
            'X': X,
            'labels': optics_model.labels_,
            'reachability': optics_model.reachability_,
            'ordering': optics_model.ordering_,
            'params': params
        }
    
    return results

# Visualize results
def visualize_results(results):
    """Visualize OPTICS clustering results for all datasets"""
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 5 * len(results)))
    
    for i, (dataset_name, result) in enumerate(results.items()):
        X = result['X']
        labels = result['labels']
        reachability = result['reachability']
        ordering = result['ordering']
        params = result['params']
        
        # Plot reachability
        ax1 = axes[i, 0]
        ax1.plot(reachability[ordering], 'k-', alpha=0.7)
        ax1.set_ylabel('Reachability Distance')
        ax1.set_title(f'OPTICS Reachability Plot - {dataset_name.capitalize()}')
        
        # Mark clusters in reachability plot
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:  # Skip noise
                continue
            
            # Get cluster points in ordering
            cluster_points = np.where(labels[ordering] == k)[0]
            
            # Mark cluster regions
            if len(cluster_points) > 0:  # Only if cluster has points
                ax1.axvspan(min(cluster_points), max(cluster_points), 
                           alpha=0.2, color=col, label=f'Cluster {k}')
        
        ax1.legend()
        ax1.grid(True)
        
        # Add parameter info
        ax1.text(0.02, 0.95, 
                 f"Parameters:\nmin_samples={params['min_samples']}\n"
                 f"xi={params['xi']}\nmin_cluster_size={params['min_cluster_size']}\n"
                 f"max_eps={params['max_eps']}",
                 transform=ax1.transAxes, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # Plot clustering results
        ax2 = axes[i, 1]
        
        # Plot each cluster
        for k, col in zip(unique_labels, colors):
            if k == -1:  # Noise points in black
                col = [0, 0, 0, 1]
            
            # Create mask for current cluster
            class_mask = (labels == k)
            
            # Plot cluster points
            ax2.scatter(X[class_mask, 0], X[class_mask, 1],
                       c=[tuple(col)], edgecolor='k', s=50, label=f'Cluster {k}')
        
        ax2.set_title(f'OPTICS Clustering - {dataset_name.capitalize()}')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.legend()
        ax2.grid(True)
        
        # Add statistics
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
    print("Results saved to optics_best_results.png")

# Main function
def main():
    # Generate datasets
    datasets = generate_datasets(n_samples=500, noise=0.05)
    
    # Run OPTICS with best parameters
    results = run_optics_with_best_params(datasets)
    
    # Visualize results
    visualize_results(results)
    
    # Print parameter tuning guide
    print("\nOPTICS Parameter Tuning Guide:")
    print("1. min_samples:")
    print("   - Controls the number of neighbors required for a point to be a core point")
    print("   - Increase: Reduces noise sensitivity, but may ignore smaller clusters")
    print("   - Decrease: Can detect smaller clusters, but may increase noise sensitivity")
    print("   - Recommended range: 1%-5% of dataset size")
    
    print("\n2. xi:")
    print("   - Controls the steepness threshold for cluster extraction")
    print("   - Increase: Extracts fewer clusters")
    print("   - Decrease: Extracts more clusters")
    print("   - Recommended range: 0.01-0.1")
    
    print("\n3. min_cluster_size:")
    print("   - Controls the minimum sample ratio to be considered a cluster")
    print("   - Increase: Ignores smaller clusters")
    print("   - Decrease: Allows smaller clusters")
    print("   - Recommended range: 0.01-0.05")
    
    print("\n4. max_eps:")
    print("   - Controls the maximum reachability distance")
    print("   - Increase: May connect more points, reducing noise")
    print("   - Decrease: May increase noise points, but clusters are more compact")
    print("   - Recommendation: If unsure, set to infinity (np.inf)")
    
    print("\nReachability Plot Interpretation:")
    print("- Valleys in the plot indicate potential clusters")
    print("- Steep slopes indicate cluster boundaries")
    print("- High plateaus typically represent noise points")
    print("- When tuning parameters, look for clear valley structures")

if __name__ == "__main__":
    main() 