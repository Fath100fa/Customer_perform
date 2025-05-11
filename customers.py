# ================================================
# Customer Preferences Clustering Analysis Project
# Dataset: FoodMart (StoresData.xlsx)
# Objective: Discover customer preference clusters
# Techniques used: K-Medoids and Hierarchical Clustering
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# --------------------------------------
# 1. Load Data and Initial Inspection
# --------------------------------------
data = pd.read_excel('data/StoresData.xlsx')

# Display basic dataset info
print("\nDataset Information:")
print(f"Shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Select numeric columns for clustering
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"\nNumeric columns to use: {numeric_cols}")

# --------------------------------------
# 2. Preprocessing
# --------------------------------------
X = data[numeric_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Visualize correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X_scaled, columns=numeric_cols).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# --------------------------------------
# 3. K-Medoids Clustering
# --------------------------------------
print("\n--- K-Medoids Clustering ---")

def find_optimal_k_medoids(X, k_range):
    silhouette_scores = []
    for k in k_range:
        kmedoids = KMedoids(n_clusters=k, metric='euclidean', random_state=42)
        cluster_labels = kmedoids.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"K = {k}, Silhouette Score = {score:.4f}")
    return silhouette_scores

k_range = range(2, 6)
silhouette_scores = find_optimal_k_medoids(X_scaled, k_range)

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k (K-Medoids)')
plt.grid(True)
plt.savefig('kmedoids_silhouette_scores.png')

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

kmedoids = KMedoids(n_clusters=optimal_k, metric='euclidean', random_state=42)
kmedoids_labels = kmedoids.fit_predict(X_scaled)
data['KMedoids_Cluster'] = kmedoids_labels

# --------------------------------------
# 4. Hierarchical Clustering
# --------------------------------------
print("\n--- Hierarchical Clustering ---")
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('hierarchical_dendrogram.png')

hierarchical_labels = fcluster(Z, optimal_k, criterion='maxclust') - 1
data['Hierarchical_Cluster'] = hierarchical_labels

# --------------------------------------
# 5. Dimensionality Reduction for Visualization
# --------------------------------------
print("\n--- Visualization ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'KMedoids_Cluster': kmedoids_labels,
    'Hierarchical_Cluster': hierarchical_labels
})

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for cluster in range(optimal_k):
    cluster_data = pca_df[pca_df['KMedoids_Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
    medoid_idx = kmedoids.medoid_indices_[cluster]
    plt.scatter(X_pca[medoid_idx, 0], X_pca[medoid_idx, 1],
                s=200, facecolors='none', edgecolors='black', linewidths=2,
                label=f'Medoid {cluster}' if cluster == 0 else "")
plt.title('K-Medoids Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.subplot(1, 2, 2)
for cluster in range(optimal_k):
    cluster_data = pca_df[pca_df['Hierarchical_Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
plt.title('Hierarchical Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.tight_layout()
plt.savefig('clustering_comparison.png')

# --------------------------------------
# 6. Evaluation
# --------------------------------------
print("\n--- Evaluation ---")
kmedoids_silhouette = silhouette_score(X_scaled, kmedoids_labels)
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
print(f"K-Medoids Silhouette Score: {kmedoids_silhouette:.4f}")
print(f"Hierarchical Silhouette Score: {hierarchical_silhouette:.4f}")

# --------------------------------------
# 7. Cluster Analysis
# --------------------------------------
print("\n--- Cluster Analysis ---")
print("\nCluster sizes:")
print("K-Medoids:")
print(pd.Series(kmedoids_labels).value_counts().sort_index())
print("\nHierarchical:")
print(pd.Series(hierarchical_labels).value_counts().sort_index())

print("\nCluster Profiles (K-Medoids):")
cluster_centers = kmedoids.cluster_centers_
cluster_centers_df = pd.DataFrame(scaler.inverse_transform(cluster_centers), columns=numeric_cols)
print(cluster_centers_df)

# --------------------------------------
# 8. Save Results
# --------------------------------------
data.to_csv('clustering_results.csv', index=False)
print("\nAnalysis complete! Results and visualizations have been saved.")