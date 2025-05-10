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

# Convert categorical variables to numeric if needed
# This step depends on your specific dataset structure
# For demonstration, we'll assume we need to preprocess some columns

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