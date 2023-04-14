import numpy as np

class Unsupervised:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
    
    def fit(self, X: np.ndarray) -> "Unsupervised":
        # Compute mean of the input data and center it
        self.mean_ = X.mean(axis=0)
        X = X - self.mean_
        # Compute the covariance matrix of the centered data
        self.cov_ = np.cov(X.T)
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        self.eigval_, self.eigvec_ = np.linalg.eig(self.cov_)
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Fit the input data and then transform it to the new space
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Transform the input data to the new space
        X = X - self.mean_
        return np.dot(X, self.eigvec_[:, :self.n_components])
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # Transform the data back to the original space
        return np.dot(X, self.eigvec_[:, :self.n_components].T) + self.mean_
    
class tSNE(Unsupervised):
    def __init__(self, n_components: int = 2, perplexity: float = 30.0, early_exaggeration: float = 12.0, learning_rate: float = 200.0, n_iter: int = 1000):
        super().__init__(n_components=n_components)
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    @staticmethod
    def _binary_search(perplexity, dist, tol=1e-5, max_iter=1000):
        # Perform binary search to find the optimal sigma values
        lo = np.ones(dist.shape[0]) * -np.inf
        hi = np.ones(dist.shape[0]) * np.inf
        for i in range(dist.shape[0]):
            perp = perplexity
            sigma = 1.0
            for j in range(max_iter):
                # Compute probabilities based on current sigma value
                p = np.exp(-dist[i] ** 2 / (2 * sigma ** 2))
                sum_p = np