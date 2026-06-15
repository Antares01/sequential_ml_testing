from abc import ABC, abstractmethod
from utils import quadratic_loss
import numpy as np

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
    def wealth(self, model, x, x_tildes, y, z):
        pass

    def e_value(self, model, x, x_tildes, y, z):
        wealths = self.wealth(model, x, x_tildes, y, z)
        if self.prequential:
            best_parameter_hat = np.argmax(self.past_martingales)
            e_value = wealths[best_parameter_hat]
        else:
            e_value = np.mean(wealths)
        return e_value
    
    @abstractmethod
    def update(self):
        pass
