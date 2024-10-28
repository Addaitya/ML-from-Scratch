import numpy as np

class Kmeans:
    def __init__(self, clusters:int):
        self._clusters= clusters
        self.centroids = None
        self.labels = None
    
    def _init_centroids(self, X: np.ndarray)->np.ndarray:
        m = X.shape[0]
        dp_ids = np.random.randint(0, m, size=self._clusters)
        centroids = X[dp_ids, :]
        return centroids

    def _closest_centroid(self, X:np.ndarray, centroids: np.ndarray)->np.ndarray:
        m = X.shape[0]
        min_dists = np.ones(m) * np.inf
        curr_labels = np.zeros(m)

        for i,C in enumerate(centroids):
            curr_dists = np.sum((X-C)**2, axis=1)
            gt = min_dists > curr_dists
            curr_labels[gt] = i
            min_dists[gt] = curr_dists[gt]
        
        return curr_labels

    def _update_centroids(self, X:np.ndarray, centroids:np.ndarray, curr_labels: np.ndarray)->np.ndarray:
        comp_centroids = np.zeros_like(centroids, dtype=np.float64)
        for i in range(self._clusters):
            ids = (curr_labels == i)
            ith_cluster = X[ids, :]
            comp_centroids[i] = np.average(ith_cluster, axis=0)
        
        return comp_centroids
    
    def fit(self, X: np.ndarray, *, iter:int=10):
        centroids = self._init_centroids(X)
        labels = self._closest_centroid(X, centroids)

        for _ in range(iter):
            centroids = self._update_centroids(X, centroids, labels)
            labels = self._closest_centroid(X, centroids)


        self.centroids = centroids
        self.labels = labels
    
    def predict(self, X_test: np.ndarray):
        try:
            y_test = self._closest_centroid(X_test, self.centroids)
            return y_test
        except TypeError:
            print("Call fit() for training first")
            raise


