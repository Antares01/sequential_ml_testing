from utils import quadratic_loss
from BettingStrategy import BettingStrategy
import numpy as np

class ExponentialBet(BettingStrategy):

    def __init__(self, parameters, loss=quadratic_loss, prequential=True, proba = False, exact = True):
        super().__init__(
            loss=loss,
            prequential=prequential,
            parameters=parameters,
            proba=proba,
        )
        self.parameters = np.array(self.parameters)
        self.wealths = None
        self.exact = exact  # exact = TRUE -> adapt the exp e-value for safety under finite derandomization samples

        # self.parameters and self.past_martingales already defined in the parent class

    def get_statistic(self, model, x, y, z):
        X = np.concatenate([x, z], axis=1)
        if self.proba:
            y_pred = model.predict_proba(X)
        else: 
            y_pred = model.predict(X)
        return self.loss(y_pred, y)
    
    def wealth(self, model, x, x_tildes, y, z):
        q = self.get_statistic(model, x, y, z)
        q_tildes = np.asarray([self.get_statistic(model, x_tildes[:,k], y, z) for k in range(x_tildes.shape[1])])

        # 1/exp_e_value calculation
        c = q_tildes - q
        d = self.parameters[:,np.newline]*c[np.newline,:]
        e = np.mean(np.exp(-d), axis=1)
        if self.exact:
            K = q_tildes.shape[0]
            e = (K*e + 1)/(K + 1)

        self.wealths = 1/e  # to avoid recomputing in the future, mb add self.wealths to base class?
        return 1/e

    # Update function does not take any arguments, no recomputation of wealth needed
    def update(self):
        self.past_martingales *= self.wealths
        self.wealths = None


