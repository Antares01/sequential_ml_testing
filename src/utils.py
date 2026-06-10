import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, log_loss


class BettingStrategy(ABC):
    history = None
    @abstractmethod
    def __init__(self, loss = mean_squared_error):
        self.loss = loss

    def __call__(self, a, b):
        pass

    def update(self):
        pass


class AntisymmetricBet(BettingStrategy):
    
    def __init__(self):
        self.history = None

    @abstractmethod
    def g_func(self, a, b):
        pass

    def get_statistic(self, m, x, y, z):
        y_pred = m.predict([x, z])
        return self.loss(y, y_pred)
    
    def get_one_plus_g_func(self, m, x, x_tilde, y, z):
        q = self.get_statistic(m, x, y, z)
        q_tilde = self.get_statistic(m, x_tilde, y, z)
        return 1 + self.g_func(q, q_tilde)
    
    def derandomized_bet(self, m, x, y, z, x_tilde):
        bet = 0
        for k in range(x_tilde.shape[1]):
            bet += self.get_one_plus_g_func(m, x, x_tilde[:, k], y, z)
        return bet / x_tilde.shape[1]


class SignBet(AntisymmetricBet):

    def __init__(self, lambd = 1, lambd_values = np.linspace(0.01, 2, 100)):
        self.lambd = lambd
        self.lambd_values = lambd_values
        self.history = np.ones_like(lambd_values)

    def g_func(self, a, b):
        return np.sign(b - a)


class TanhBet(AntisymmetricBet):
    def __init__(self, lambd = 1, scale=20):
        self.scale = scale
        self.lambd = lambd  

    def g_func(self, a, b):
        return np.tanh(self.scale * (b - a) / np.max((a, b)))

class CoinBetting(AntisymmetricBet):
    def __init__(self, lambd = 1, M_values = np.linspace(0.01, 10, 100)):
        self.M_values = M_values
        self.lambd = lambd

    def g_func(self, a, b):
        return np.clip(b - a, -self.M_values, self.M_values) / self.M_values
        
class LossEstimationBet(AntisymmetricBet):
    pass

class ExponentialBet(BettingStrategy):
    pass



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
