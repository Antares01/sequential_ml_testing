#from utils import quadratic_loss
from sklearn.metrics import mean_squared_error, log_loss
from BettingStrategy import BettingStrategy
import numpy as np

class AntisymmetricBet(BettingStrategy):

    def __init__(self, g_family, update_g_func, parameters, past_qs=None, loss=mean_squared_error, prequential=True, proba = False):
        super().__init__(
            loss=loss,
            prequential=prequential,
            parameters=parameters,
            proba=proba,
        )
        self.g_family = g_family
        self.update_g_func = update_g_func
        if past_qs is not None:
            self.past_qs = past_qs
            self.g_family = self.update_g_func(self.past_qs, self.parameters, self.g_family)
        else:
            self.past_qs = []
        self.current_qs = None

    def g_func(self, q, q_tilde):
        g_value = np.zeros(len(self.parameters))

        for i, param in enumerate(self.parameters):
            g_value[i] = np.mean(
                self.g_family(q, q_tilde, param)
            )

        return g_value

    def get_statistic(self, model, X, y):
        if self.proba:
            y_pred = model.predict_proba(X)
        else: 
            y_pred = model.predict(X)
        return self.loss(y_pred, y)
    
    
    def derandomized_bet(self, model, X, X_j_tildes, y, j):  # X_tilde is batch x B resample
        bet = 0
        q = self.get_statistic(model, X, y)
        q_tildes = []
        for k in range(X_j_tildes.shape[1]):
            X_k = X.copy()
            X_k[:, j] = X_j_tildes[:, k]
            q_tilde = self.get_statistic(model,  X_k, y)
            q_tildes.append(q_tilde)
            bet += 1 + self.g_func(q, q_tilde)
        q_tildes = np.asarray(q_tildes)
        self.current_qs = (q, q_tildes)
        return bet / X_j_tildes.shape[1]
    
    def wealth(self, model, x, x_j_tildes, y, j):
        self.wealths = self.derandomized_bet(model, x, x_j_tildes, y, j)
        return self.wealths

    def update(self):
        self.past_qs.append(self.current_qs)
        self.g_family = self.update_g_func(self.past_qs, self.parameters, self.g_family)
        self.past_martingales *= self.wealths
        #should this also update self.past_martingales?


