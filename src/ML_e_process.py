import numpy as np
from copy import deepcopy
from AntisymmetricBet import AntisymmetricBet



import json
import os
from sklearn.linear_model import LassoCV, Lasso
from Sampler import get_data_statistics, sample_from_gaussian
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from Sampler import DefaultSampler

from src.utils import LossEstimationBet, SignBet, ExponentialBet, CoinBetting, BettingStrategy, TanhBet, TestStatistic, default, lasso_cv_online_learning
simplefilter("ignore", category=ConvergenceWarning)


class ML_e_process:
    """
    Conditional Testing with Machine Learning e-values
    """

    def __init__(self, batch_list=[2, 5, 10], n_init=50, b_resamplings=20, study_j=[0],
                 betting_strategies=None, model=LassoCV(),
                 #learn_conditional_distribution=get_data_statistics,
                 samplers=None, #sampling_args={}
                 learn_conditional_distribution=True,
                 optional_stopping=True, 
                 ):
        """

        optional_stopping: if the martingale is over the 1/alpha, then it stops, otherwise it continues computing the martingale



        :param batch_list: A list of batch sizes for the batch-ensemble.
        All batches must be divisors of the maximal one.
        :param n_init: The number of samples for the initial training.
        :param K: The de-randomization parameter. Number of dummy copies to be used for the wealth computation.
        :param j: The index of the tested feature. If you wish to test a different feature,
        you should create a new instance.
        :param g_func: The betting score function. Must be antisymmetric.
        :param test_statistic: The test statistic function, used to compare between the original and the dummy features.
        :param offline: Train offline LassoCV instead of online Lasso.
        :param path: Folder path to save and load martingales data.
        :param load_name: File name to load old martingales data.
        If given, the online updates start from the last saved wealth, and the last used point.
        If not given, the test starts from initial wealth 1.
        If you choose to load previous data, make sure to run with the same batch list, and on the same feature j.
        :param save_name: File name to save martingales data.
        :param learn_conditional_distribution: This function get X, the dataset,
        and returns learned arguments that are needed for the sampling of X_tilde.
        The returned arguments are saved to the dictionary sampling_args, and passed to the sampling function.
        :param sampling_func: This function gets X, j, and the additional arguments in sampling_args,
        and returns the dummy features X_tilde.
        :param sampling_args: A dictionary with all the non-learned arguments to pass to the sampling functions.
        """
        max_b = np.max(batch_list)
        for b in batch_list:
            assert max_b % b == 0
        self.batch_list = batch_list
        self.n_init = n_init
        self.b_resamplings = b_resamplings
        self.study_j = study_j
        if betting_strategies==None:
            self.betting_strategies = {"kernel_bet": AntisymmetricBet()} #TODO: define default betting strategy
            
        else:
            self.betting_strategies = betting_strategies
        self.model = model
        self.models_dict = {} # For the online Lasso
        if samplers == None:
            self.samplers = {j: DefaultSampler(j) for j in study_j} 
        else:
            self.samplers = samplers
        self.learn_conditional_distribution = learn_conditional_distribution
        self.optional_stopping = optional_stopping
        self.bets_js_bs = {
                j: {
                    b: {
                        key: deepcopy(strategy)
                        for key, strategy in self.betting_strategy_classes.items()
                    }
                    for b in self.list_bs
                }
                for j in self.list_js
            }

    def _sample_conditionals(self, X, feature_j):
        X_j_tildes = np.empty((X.shape[0], self.b_resamplings))
        sampler = self.sampler[feature_j]

        for b in range(self.b_resamplings):
            X_j_tildes[:, b] = sampler.sample(X)

        return X_j_tildes


    

    def martingales(self, X, y, start_idx=None, alpha=0.05):
        """
        :param X: The data matrix with size (n, d).
        :param y: A vector of labels with size (n, 1)
        :param start_idx: The first sample that will be used to update the martingales.
        All points before it will be used for training only. If None, the first sample will be n_init.
        :param alpha: The target level. The null will be rejected when the martingale will reach 1/alpha.
        :return: Whether the null is rejected or not, i.e., whether the tested feature is important or not.
        """
        martingales = {
            strategy: {
                b: np.ones((len(self.study_j), X.shape[0]))
                for b in self.list_bs
            }
            for strategy in self.betting_strategies
        }

        if start_idx is None:
            start_idx = self.n_init
        # Train the model on the available data points, that are not used for the martingales update.
        # If you wish to use a different predictive model, please replace the Lasso model here.
        n = X.shape[0]
        # Run the sequential updates
        update_points = sorted({
            t
            for batch in self.batch_list
            for t in range(start_idx + batch, n + 1, batch)
        })
        for new_points in update_points:
            # We first update the model and the conditional sampler
            self.model = self.model.fit(X[:new_points, :], y[:new_points].ravel())
            for sampler in self.samplers:
                sampler.fit(X[:new_points, :])
            for b in self.batch_list:
                if new_points % b == 0:
                    end = min(new_points + b, n)
                    for j in self.study_j:
                        X_j_tildes = self._sample_conditionals(X[new_points:end], j)
                        for strategy in self.betting_strategies:
                            martingales[strategy][b][j, new_points:]*=self.bets_js_bs[j][b][strategy].e_value( self.model, X[new_points:end, :], X_j_tildes, y, j)
                            self.bets_js_bs[j][b][strategy].update()
        return martingales

