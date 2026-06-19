from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, log_loss
import numpy as np

class BettingStrategy(ABC):
    
    def __init__(self, loss = mean_squared_error, prequential = True, parameters = None, proba = False):
        self.loss = loss
        self.parameters = parameters
        self.past_martingales = np.ones(len(parameters))
        self.prequential = prequential
        self.proba = proba
    def __call__(self, a, b):
        pass
    
    @abstractmethod
    def wealth(self, model, x, x_j_tildes, y, j):
        pass

    def e_value(self, model, x, x_j_tildes, y, j):
        wealths = self.wealth(model, x, x_j_tildes, y, j)
        if self.prequential:
            best_parameter_hat = np.argmax(self.past_martingales)
            e_value = wealths[best_parameter_hat]
        else:
            e_value = np.mean(wealths)
        return e_value
    
    @abstractmethod
    def update(self):
        pass
