import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from sklearn.neighbors import KernelDensity



def quadratic_loss(y_pred, y_true):
    return (y_pred-y_true)**2

def log_loss(y_pred, y_true, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class BettingStrategy(ABC):
    
    def __init__(self, loss = quadratic_loss, prequential = True, parameters = None, proba = False):
        self.loss = loss
        self.parameters = parameters
        self.past_martingales = np.ones(len(parameters))
        self.prequential = prequential
        self.proba = proba
    def __call__(self, a, b):
        pass
    
    @abstractmethod
    def wealth(self, model, x, x_tilde, y, z):
        pass

    def e_value(self, model, x, x_tilde, y, z):
        wealths = self.wealth(model, x, x_tilde, y, z)
        if self.prequential:
            best_parameter_hat = np.argmax(self.past_martingales)
            e_value = wealths[best_parameter_hat]
        else:
            e_value = np.mean(wealths)
        self.past_martingales *= wealths
        return e_value
    
    @abstractmethod
    def update(self, model, x, x_tilde, y, z):
        pass


class AntisymmetricBet(BettingStrategy):

    def __init__(self, g_family, update_g_func, parameters, loss=quadratic_loss, prequential=True, proba = False):
        super().__init__(
            loss=loss,
            prequential=prequential,
            parameters=parameters,
            proba=proba,
        )
        self.g_family = g_family
        self.update_g_func = update_g_func
        self.past_qs = []
        

    def g_func(self, q, q_tilde): # Antisymetric function: takes batch qs and returns the g_func of each one of the parameters
        g_value = np.zeros((q.shape[0],len(self.parameters)))
        for i, param in enumerate(self.parameters):
            g_value[:, i] = self.g_family(q, q_tilde, param)
        return np.mean(g_value, axis=0)

    def get_statistic(self, model, x, y, z):
        X = np.concatenate([x, z], axis=1)
        if self.proba:
            y_pred = model.predict_proba(X)
        else: 
            y_pred = model.predict(X)
        return self.loss(y_pred, y)
    
    def get_one_plus_g_func(self, model, x, x_tilde, y, z):
        q = self.get_statistic(model, x, y, z)
        q_tilde = self.get_statistic(model, x_tilde, y, z)
        return 1 + self.g_func(q, q_tilde)
    
    def derandomized_bet(self, model, x, x_tildes, y, z):  # X_tilde is batch x B resample
        bet = 0
        for k in range(x_tildes.shape[1]):
            bet += self.get_one_plus_g_func(model, x, x_tildes[:, k], y, z)
        return bet / x_tildes.shape[1]
    
    def wealth(self, model, x, x_tilde, y, z):
        return self.derandomized_bet(model, x, x_tilde, y, z)

    def update(self, q, q_tildes):
        self.past_qs.append((q, q_tildes))
        self.g_family = self.update_g_func(self.past_qs, self.parameters)



def update_g_func_kernel_density(history, parameters, resamplings = 10):
    X = []

    for q, q_tildes in history:

        B = min(resamplings, q_tildes.shape[1])

        for i in range(B):

            pairs = np.column_stack([
                q,
                q_tildes[:, i]
            ])

            X.append(pairs)

    X = np.vstack(X)

    kdes = {}
    for param in parameters:
        key = (param["kernel"], param["bandwidth"])
        kdes[key] = KernelDensity(kernel=param['kernel'], bandwidth=param['bandwidth']).fit(X)

    def new_g(q, q_tilde, param):
        points = np.column_stack([q, q_tilde])
        points_inverse = np.column_stack([q_tilde, q])
        key = (param["kernel"], param["bandwidth"])
        q_orig = np.exp(kdes[key].score_samples(points))
        q_inverse = np.exp(kdes[key].score_samples(points_inverse))
        eps = 1e-12
        return (q_orig - q_inverse)/(q_orig + q_inverse + eps)

    return new_g

class SignBet(AntisymmetricBet):

    def __init__(self, lambd = 0.99, lambd_values = np.linspace(0.01, 0.99, 100)):
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
