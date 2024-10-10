from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans


class KClusterCentroids(ClusterCentroids):
    def __init__(self, *, k=8, random_state=None, **kwargs):
        self.k = k
        estimator = KMeans(n_clusters=k, random_state=random_state)
        super().__init__(random_state=random_state, estimator=estimator, **kwargs)

