import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Data points
X = np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4]])

# Perform hierarchical clustering (Euclidean distance)
Z = linkage(X, method='complete', metric='euclidean')

# Cut dendrogram into 2 clusters
clusters = fcluster(Z, t=2, criterion='maxclust')

# Plot dendrogram
plt.figure(figsize=(8,5))
dendrogram(Z, labels=[f'P{i+1}' for i in range(len(X))])
plt.title("Agglomerative Hierarchical Clustering Dendrogram (Euclidean Distance)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# Print cluster assignments
for i, point in enumerate(X):
    print(f"Point P{i+1} {point} → Cluster {clusters[i]}")