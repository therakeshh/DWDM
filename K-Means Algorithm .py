# k-Means Clustering with k=3
import numpy as np
import matplotlib.pyplot as plt

# Points
X = np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]])

# Initial centroids: choose 3 points (P1, P4, P7 for example)
centroids = np.array([[2,10],[5,8],[1,2]])

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:,None] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Run 3 iterations
for _ in range(3):
    labels = assign_clusters(X, centroids)
    centroids = np.array([X[labels==i].mean(axis=0) for i in range(3)])

# Plot result
colors = ['blue','red','green']
for i in range(3):
    plt.scatter(X[labels==i,0], X[labels==i,1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:,0], centroids[:,1], c='yellow', marker='X', s=200, label='Centroids')

# Label points
for idx, point in enumerate(X):
    plt.text(point[0]+0.3, point[1], f'P{idx+1}')

plt.title("K-Means Clustering (K=3) - After 3 Iterations")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()