#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import pairwise_distances


# 1. Data Loading and Preprocessing

data = pd.read_excel('data/StoresData.xlsx')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())



# Select only numeric columns for clustering
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"\nNumeric columns to use: {numeric_cols}")

# Create feature matrix
X = data[numeric_cols].copy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-medoids Clustering
print("\n--- K-medoids Clustering ---")

# Function to find optimal number of clusters using silhouette score
def find_optimal_k_medoids(X, k_range):
    silhouette_scores = []
    for k in k_range:
        kmedoids = KMedoids(n_clusters=k, metric='euclidean', random_state=42)
        cluster_labels = kmedoids.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"K = {k}, Silhouette Score = {score:.4f}")
    return silhouette_scores

# Try different numbers of clusters
k_range = range(2, 6)
silhouette_scores = find_optimal_k_medoids(X_scaled, k_range)

# Visualize the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method For Optimal k (K-medoids)')
plt.grid(True)
plt.savefig('kmedoids_silhouette_scores.png')

# Get the optimal k value
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters for K-medoids: {optimal_k}")

# Apply K-medoids with optimal k
kmedoids = KMedoids(n_clusters=optimal_k, metric='euclidean', random_state=42)
kmedoids_labels = kmedoids.fit_predict(X_scaled)

# Add the cluster labels to the data
data['KMedoids_Cluster'] = kmedoids_labels

# 3. Hierarchical Clustering
print("\n--- Hierarchical Clustering ---")

# Compute the linkage matrix
Z = linkage(X_scaled, method='ward')

# Plot dendrogram to help determine the number of clusters
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig('hierarchical_dendrogram.png')

# Extract clusters from the hierarchical clustering (using the same k as K-medoids for comparison)
hierarchical_labels = fcluster(Z, optimal_k, criterion='maxclust') - 1  # Subtract 1 to start from 0

# Add the cluster labels to the data
data['Hierarchical_Cluster'] = hierarchical_labels

# 4. Visualization with PCA for dimensionality reduction
print("\n--- Visualizations ---")

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for easy plotting
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'KMedoids_Cluster': kmedoids_labels,
    'Hierarchical_Cluster': hierarchical_labels
})

# Plot K-medoids results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for cluster in range(optimal_k):
    cluster_data = pca_df[pca_df['KMedoids_Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
    
    # Mark the medoids
    medoid_idx = kmedoids.medoid_indices_[cluster]
    plt.scatter(X_pca[medoid_idx, 0], X_pca[medoid_idx, 1], 
                s=200, facecolors='none', edgecolors='black', linewidths=2,
                label=f'Medoid {cluster}' if cluster == 0 else "")

plt.title('K-medoids Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Plot Hierarchical clustering results
plt.subplot(1, 2, 2)
for cluster in range(optimal_k):
    cluster_data = pca_df[pca_df['Hierarchical_Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')

plt.title('Hierarchical Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.tight_layout()
plt.savefig('clustering_comparison.png')

# 5. Evaluate and Compare the Clustering Results
print("\n--- Evaluation ---")

# Compute silhouette scores for both methods
kmedoids_silhouette = silhouette_score(X_scaled, kmedoids_labels)
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)

print(f"K-medoids Silhouette Score: {kmedoids_silhouette:.4f}")
print(f"Hierarchical Silhouette Score: {hierarchical_silhouette:.4f}")

# Additional simple evaluation metrics
print("\n--- Additional Evaluation Metrics ---")

# Calculate inertia (sum of distances to closest centroid) for K-medoids
kmedoids_inertia = sum(np.min(pairwise_distances(X_scaled, kmedoids.cluster_centers_), axis=1))
print(f"K-medoids Inertia: {kmedoids_inertia:.4f}")

# Calculate within-cluster variance for each method
kmedoids_variances = []
for i in range(optimal_k):
    cluster_points = X_scaled[kmedoids_labels == i]
    if len(cluster_points) > 0:  # Ensure cluster is not empty
        variance = np.mean(np.var(cluster_points, axis=0))
        kmedoids_variances.append(variance)
print(f"K-medoids Average Within-Cluster Variance: {np.mean(kmedoids_variances):.4f}")

hierarchical_variances = []
for i in range(optimal_k):
    cluster_points = X_scaled[hierarchical_labels == i]
    if len(cluster_points) > 0:  # Ensure cluster is not empty
        variance = np.mean(np.var(cluster_points, axis=0))
        hierarchical_variances.append(variance)
print(f"Hierarchical Average Within-Cluster Variance: {np.mean(hierarchical_variances):.4f}")

# Calculate between-cluster separation
def calculate_between_cluster_distance(centers):
    n_centers = centers.shape[0]
    distances = []
    for i in range(n_centers):
        for j in range(i+1, n_centers):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)
    return np.mean(distances) if distances else 0

# Calculate cluster centers for hierarchical clustering
hierarchical_centers = np.array([X_scaled[hierarchical_labels == i].mean(axis=0) for i in range(optimal_k)])

# Calculate between-cluster distances
kmedoids_between = calculate_between_cluster_distance(kmedoids.cluster_centers_)
hierarchical_between = calculate_between_cluster_distance(hierarchical_centers)

print(f"K-medoids Between-Cluster Average Distance: {kmedoids_between:.4f}")
print(f"Hierarchical Between-Cluster Average Distance: {hierarchical_between:.4f}")

# Calculate percentage of variance explained by clustering
def calculate_variance_explained(X, labels):
    # Total variance
    total_variance = np.sum(np.var(X, axis=0))
    
    # Within-cluster variance
    within_variance = 0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        within_variance += np.sum(np.var(cluster_points, axis=0)) * len(cluster_points)
    within_variance /= len(X)
    
    # Between-cluster variance is the difference
    between_variance = total_variance - within_variance
    
    # Percentage explained
    return (between_variance / total_variance) * 100 if total_variance > 0 else 0

kmedoids_explained = calculate_variance_explained(X_scaled, kmedoids_labels)
hierarchical_explained = calculate_variance_explained(X_scaled, hierarchical_labels)

print(f"K-medoids Percentage of Variance Explained: {kmedoids_explained:.2f}%")
print(f"Hierarchical Percentage of Variance Explained: {hierarchical_explained:.2f}%")

# 6. Analysis of Clusters
print("\n--- Cluster Analysis ---")

# Analyze cluster sizes
print("\nCluster sizes:")
print("K-medoids:")
print(pd.Series(kmedoids_labels).value_counts().sort_index())
print("\nHierarchical:")
print(pd.Series(hierarchical_labels).value_counts().sort_index())

# Analyze cluster centers (profiles)
print("\nCluster Profiles (K-medoids):")
cluster_centers = kmedoids.cluster_centers_
cluster_centers_df = pd.DataFrame(scaler.inverse_transform(cluster_centers), 
                        columns=numeric_cols)
print(cluster_centers_df)

# Save the results to a CSV file
data.to_csv('clustering_results.csv', index=False)

print("\nAnalysis complete! Results and visualizations have been saved.") 