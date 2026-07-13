import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([[2,6],[3,4],[3,8],[4,7],[6,2],[6,4],[7,3],[7,4],[8,5],[7,6]])

# Initial medoids
medoids = np.array([[2,6],[3,4]])

def manhattan(a,b): return np.sum(np.abs(a-b))
def assign(X, medoids):
    return np.argmin([[manhattan(x,m) for m in medoids] for x in X], axis=1)
def update(X, labels, k):
    return np.array([min(X[labels==i], key=lambda p: sum(manhattan(p,q) for q in X[labels==i])) for i in range(k)])

# Run PAM
k=2
for _ in range(5):
    labels = assign(X, medoids)
    medoids = update(X, labels, k)

# Plot
colors=['blue','red']
for i in range(k):
    plt.scatter(X[labels==i,0], X[labels==i,1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(medoids[:,0], medoids[:,1], c='yellow', marker='X', s=200, label='Medoids')
for idx,p in enumerate(X): plt.text(p[0]+0.2,p[1],f'P{idx+1}')
plt.title("K-Medoids (PAM) with Manhattan Distance (k=2)")
plt.legend(); plt.grid(True); plt.show()