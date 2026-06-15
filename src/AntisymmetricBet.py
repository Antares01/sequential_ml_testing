from utils import quadratic_loss
from BettingStrategy import BettingStrategy
import numpy as np

class AntisymmetricBet(BettingStrategy):

    def __init__(self, g_family, update_g_func, parameters, past_qs=None, loss=quadratic_loss, prequential=True, proba = False):
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
        

    def g_func(self, q, q_tilde):
        g_value = np.zeros(len(self.parameters))

        for i, param in enumerate(self.parameters):
            g_value[i] = np.mean(
                self.g_family(q, q_tilde, param)
            )

        return g_value

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
    
    def wealth(self, model, x, x_tildes, y, z):
        return self.derandomized_bet(model, x, x_tildes, y, z)

    def update(self, model, x, x_tildes, y, z):
        q = self.get_statistic(model, x, y, z)# Not sure if this is eficient to recompute that many times the q and q_tildes
        q_tildes = np.asarray([self.get_statistic(model, x_tildes[:,k], y, z) for k in range(x_tildes.shape[1])])
        self.past_qs.append((q, q_tildes))
        self.g_family = self.update_g_func(self.past_qs, self.parameters, self.g_family)

        #should this also update self.past_martingales?


