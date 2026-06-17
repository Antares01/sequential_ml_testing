import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression, LassoCV



class Sampler(ABC):
    def __init__(self, j):
        self.j = j

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def sample(self, X):
        """Return an (n,) vector of sampled values for feature j."""
        pass


class GaussianSampler(Sampler):

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigma = np.cov(X, rowvar=False)

        mask = np.arange(X.shape[1]) != self.j

        self.mu_minus = self.mu[mask]
        self.sigma_j_minus = self.sigma[self.j, mask]
        self.sigma_minus_j = self.sigma[mask, self.j]
        self.sigma_minus = self.sigma[np.ix_(mask, mask)]

        self.sigma_minus_inv = np.linalg.inv(self.sigma_minus) # We save it to avoid recomputing it b_resampling times

        return self

    def sample(self, X):
        mask = np.arange(X.shape[1]) != self.j
        X_minus = X[:, mask]

        mu_cond = (
            self.mu[self.j]
            + (X_minus - self.mu_minus)
            @ self.sigma_minus_inv
            @ self.sigma_minus_j
        )

        sigma_cond = (
            self.sigma[self.j, self.j]
            - self.sigma_j_minus
            @ self.sigma_minus_inv
            @ self.sigma_minus_j
        )

        return np.random.normal(mu_cond, np.sqrt(sigma_cond))



class ProbaSampler(Sampler):

    def fit(self, X):

        mask = np.arange(X.shape[1]) != self.j

        X_other = X[:, mask]
        y = X[:, self.j]

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_other, y)

        return self

    def sample(self, X):

        mask = np.arange(X.shape[1]) != self.j
        X_other = X[:, mask]

        p = self.model.predict_proba(X_other)[:, 1]

        return np.random.binomial(1, p)
    

class RegressorSampler(Sampler):

    def fit(self, X):

        mask = np.arange(X.shape[1]) != self.j

        X_other = X[:, mask]
        y = X[:, self.j]

        self.model = LassoCV()
        self.model.fit(X_other, y)

        self.residuals = y - self.model.predict(X_other)

        return self

    def sample(self, X):

        mask = np.arange(X.shape[1]) != self.j

        X_other = X[:, mask]

        fitted = self.model.predict(X_other)

        residuals = np.random.choice(
            self.residuals,
            size=fitted.shape[0],
            replace=True  
        )

        return fitted + residuals
    

class DefaultSampler(Sampler):

    def fit(self, X):

        xj = X[:, self.j]

        if np.unique(xj).size > 10:
            self.sampler = RegressorSampler(self.j)
        else:
            self.sampler = ProbaSampler(self.j)

        self.sampler.fit(X)

        return self

    def sample(self, X):
        return self.sampler.sample(X)

                
