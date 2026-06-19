from sklearn.metrics import mean_squared_error, log_loss
from BettingStrategy import BettingStrategy
import numpy as np

class ExponentialBet(BettingStrategy):

    def __init__(self, parameters, loss=mean_squared_error, prequential=True, proba = False, exact = True):
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

    def get_statistic(self, model, X, y):
        if self.proba:
            y_pred = model.predict_proba(X)
        else: 
            y_pred = model.predict(X)
        return self.loss(y_pred, y)
    
    def wealth(self, model, x, x_j_tildes, y, j):
        q = self.get_statistic(model, x, y)

        q_tildes = np.asarray([
            self.get_statistic(
                model,
                np.column_stack([
                    x[:, :j],               
                    x_j_tildes[:, k],        
                    x[:, j+1:]               
                ]),
                y
            )
            for k in range(x_j_tildes.shape[1])
        ])

        # 1/exp_e_value calculation
        c = q_tildes - q
        d = self.parameters[:,np.newaxis]*c[np.newaxis,:]
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


