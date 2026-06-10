import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class BettingStrategy(ABC):
    @abstractmethod
    def __call__(self, a, b):
        pass

    def update(self):
        pass

class SignBet(BettingStrategy):
    def __call__(self, a, b):
        return np.sign(b - a)

class TanhBet(BettingStrategy):
    def __init__(self, scale=20):
        self.scale = scale

    def __call__(self, a, b):
        return np.tanh(self.scale * (b - a) / np.max((a, b)))

class ExponentialBet(BettingStrategy):
    def __init__(self, eta_values = np.linspace(0, 10, 1001, endpoint=False)[1:]):
        self.is_exponential_bet = True
        self.eta_values = eta_values
        self.history = np.ones_like(self.eta_values)
    def __call__(self, a, b):
        '''Inputs:
         - a: scalar, average prediction loss over batch 
         - b: 1d-array, average prediction loss over batch with sampled data.
         Returns:
         - 1d array of length len(eta_values) representing the Exp e-variable for different eta parameter choices
         '''
        c = b-a
        d = self.eta_values[:,np.newaxis] * c[np.newaxis,:]
        e = np.mean(np.exp(-d),axis=1)
        return 1/e


class CoinBetting(BettingStrategy):
    def __init__(self, M_values = np.linspace(0.01, 10, 100)):
        self.M_values = M_values

    def __call__(self, a, b):
        return np.clip(b - a, -self.M_values, self.M_values) / self.M_values
        
        

class LossEstimationBet(BettingStrategy):
    pass


class TestStatistic(Enum):
    mse = lambda a, b: ((a - b) ** 2).mean()


def get_martingale_values(martingale_dict):
    b_last_used_list = []
    st_list = []
    for b in martingale_dict.keys():
        st_list.append(martingale_dict[b]["St"])
        b_last_used_list.append(martingale_dict[b]["last_used_idx"])
    return np.array(st_list).mean(), np.array(b_last_used_list).max()


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def lasso_cv_online_learning(X, y, models_dict, val_prcg=0.2):
    """
    Online hyper-parameter tuning using ensemble of Lasso models
    :param X: The data matrix with size (n, d).
    :param y: A vector of labels with size (n, 1).
    :param models_dict: A dictionary contains M models.
    The keys are the values of the tuned parameter (the regularization constant the multiplies the L1 term
    in the loss function).
    The values in the dictionary are the lasso models with the corresponding parameter.
    :param val_prcg: The percentage of data to be used for validation.
    :return: The regularization constant of the model that got the best score on the validation data.
    """
    train_idx = int(len(y) * (1-val_prcg))
    X_train = X[:train_idx, :]
    y_train = y[:train_idx]
    X_val = X[train_idx:, :]
    y_val = y[train_idx:]
    alpha_vec = list(models_dict.keys())
    score = -1
    best_alpha = alpha_vec[0]
    for alpha in alpha_vec:
        models_dict[alpha].fit(X_train, y_train.ravel())
        model_score = models_dict[alpha].score(X_val, y_val.ravel())
        if model_score > score:
            best_alpha = alpha
            score = model_score
    return best_alpha
