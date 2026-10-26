import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load real-world dataset (Wine Dataset)
wine = load_wine()
X = wine.data

# 2. Preprocess Data: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Reduce dimensions to 2D for visualization using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4. K-Means Clustering Evaluation
print("--- K-Means Evaluation ---")
for k in range(2, 6):
  km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
  labels_temp = km_temp.fit_predict(X_scaled)
  score = silhouette_score(X_scaled, labels_temp)
  print(f"K = {k} | Silhouette Score: {score:.4f}")

# Train optimal K-Means model (k=3 for Wine dataset)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

# 5. Agglomerative Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = hierarchical.fit_predict(X_scaled)
hier_silhouette = silhouette_score(X_scaled, hier_labels)

print("\n--- Model Comparison (k=3) ---")
print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Hierarchical Clustering Silhouette Score: {hier_silhouette:.4f}")

# 6. Visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot K-Means Clusters (using PCA coordinates)
axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.8
)
centers_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Centroids',
)
axes[0].set_title(
    f'K-Means Clustering (k=3)\nSilhouette: {kmeans_silhouette:.3f}'
)
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend()

# Plot Hierarchical Clusters (using PCA coordinates)
axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap='plasma', s=50, alpha=0.8
)
axes[1].set_title(
    f'Hierarchical Clustering (k=3)\nSilhouette: {hier_silhouette:.3f}'
)
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')

# Plot Hierarchical Dendrogram
linked = linkage(X_scaled, method='ward')
dendrogram(linked, ax=axes[2], truncate_mode='lastp', p=15, leaf_rotation=45)
axes[2].set_title('Hierarchical Clustering Dendrogram')
axes[2].set_xlabel('Cluster Size / Sample Index')
axes[2].set_ylabel('Euclidean Distance')

plt.tight_layout()
plt.show()